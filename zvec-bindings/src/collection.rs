use std::ffi::CString;
use std::os::raw::c_int;
use std::path::Path;
use std::ptr;

use crate::doc::{Doc, DocList, DocMap, WriteResults};
use crate::error::{check_error, Result};
use crate::ffi;
use crate::query::{GroupByVectorQuery, GroupResults, VectorQuery};
use crate::schema::{CollectionSchema, FieldSchema};
use crate::types::{IndexType, MetricType, QuantizeType};

/// Collection statistics.
///
/// In zvec v0.5.0 the stats object is opaque; `memory_usage` and
/// `json_details` are no longer exposed by the upstream C API.
pub struct CollectionStats {
    ptr: *mut ffi::zvec_collection_stats_t,
}

impl CollectionStats {
    pub fn doc_count(&self) -> u64 {
        unsafe { ffi::zvec_collection_stats_get_doc_count(self.ptr) }
    }

    pub fn index_count(&self) -> usize {
        unsafe { ffi::zvec_collection_stats_get_index_count(self.ptr) }
    }

    pub fn index_name(&self, index: usize) -> Option<&str> {
        unsafe {
            let p = ffi::zvec_collection_stats_get_index_name(self.ptr, index);
            if p.is_null() {
                None
            } else {
                std::ffi::CStr::from_ptr(p).to_str().ok()
            }
        }
    }

    pub fn index_completeness(&self, index: usize) -> f32 {
        unsafe { ffi::zvec_collection_stats_get_index_completeness(self.ptr, index) }
    }
}

impl Drop for CollectionStats {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_collection_stats_destroy(self.ptr) };
        }
    }
}

/// A collection of documents with vector search capabilities.
///
/// A Collection is the main entry point for working with zvec. It represents
/// a collection of documents that can be searched using vector similarity.
///
/// # Example
///
/// ```rust,no_run
/// use zvec_bindings::{create_and_open, CollectionSchema, Doc, VectorQuery, VectorSchema};
///
/// # fn main() -> zvec_bindings::Result<()> {
/// let mut schema = CollectionSchema::new("my_collection");
/// schema.add_field(VectorSchema::fp32("embedding", 128))?;
///
/// let collection = create_and_open("./my_db", schema)?;
///
/// // Insert documents
/// let mut doc = Doc::id("doc_1");
/// doc.set_vector("embedding", &[0.1, 0.2, 0.3])?;
/// collection.insert(&[doc])?;
///
/// // Search
/// let query = VectorQuery::new("embedding").topk(10).vector(&[0.1, 0.2, 0.3])?;
/// let results = collection.query(query)?;
/// # Ok(())
/// # }
/// ```
pub struct Collection {
    ptr: *mut ffi::zvec_collection_t,
    /// Filesystem path the collection was opened with. Preserved because
    /// upstream v0.5.0 does not expose a `zvec_collection_path` accessor.
    path: Option<String>,
}

impl Collection {
    pub fn create_and_open<P: AsRef<Path>>(path: P, schema: CollectionSchema) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        let path_c = CString::new(path_str.as_str())
            .map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))?;

        let mut handle: *mut ffi::zvec_collection_t = ptr::null_mut();
        let code = unsafe {
            ffi::zvec_collection_create_and_open(
                path_c.as_ptr(),
                schema.ptr,
                ptr::null(),
                &mut handle,
            )
        };
        // The schema pointer is consumed (cloned) by upstream; drop our wrapper
        // without destroying the C pointer.
        std::mem::forget(schema);
        check_error(code as c_int)?;

        if handle.is_null() {
            return Err(crate::error::Error::InternalError(
                "Failed to create collection: null pointer".into(),
            ));
        }

        Ok(Self {
            ptr: handle,
            path: Some(path_str),
        })
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        let path_c = CString::new(path_str.as_str())
            .map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))?;

        let mut handle: *mut ffi::zvec_collection_t = ptr::null_mut();
        let code = unsafe { ffi::zvec_collection_open(path_c.as_ptr(), ptr::null(), &mut handle) };
        check_error(code as c_int)?;

        if handle.is_null() {
            return Err(crate::error::Error::InternalError(
                "Failed to open collection: null pointer".into(),
            ));
        }

        Ok(Self {
            ptr: handle,
            path: Some(path_str),
        })
    }

    /// Get the filesystem path where this collection was opened.
    ///
    /// In zvec v0.5.0 the upstream C API does not expose a path accessor,
    /// so this returns the path the collection was opened/created with.
    pub fn path(&self) -> Result<String> {
        Ok(self.path.clone().unwrap_or_default())
    }

    /// Insert documents into the collection.
    ///
    /// Returns a [`WriteResults`] indicating the success or failure of each
    /// insert (uses the per-document `_with_results` upstream variant).
    pub fn insert(&self, docs: &[Doc]) -> Result<WriteResults> {
        let doc_ptrs: Vec<*const ffi::zvec_doc_t> = docs
            .iter()
            .map(|d| d.ptr as *const ffi::zvec_doc_t)
            .collect();
        let mut results: *mut ffi::zvec_write_result_t = ptr::null_mut();
        let mut result_count: usize = 0;
        let code = unsafe {
            ffi::zvec_collection_insert_with_results(
                self.ptr,
                doc_ptrs.as_ptr() as *mut *const ffi::zvec_doc_t,
                doc_ptrs.len(),
                &mut results,
                &mut result_count,
            )
        };
        check_error(code as c_int)?;
        Ok(WriteResults::from_raw(results, result_count))
    }

    /// Insert documents, returning only aggregate success/error counts.
    ///
    /// This is lighter-weight than [`insert`] when per-document status is not
    /// needed.
    pub fn insert_counted(&self, docs: &[Doc]) -> Result<(usize, usize)> {
        let doc_ptrs: Vec<*const ffi::zvec_doc_t> = docs
            .iter()
            .map(|d| d.ptr as *const ffi::zvec_doc_t)
            .collect();
        let mut success: usize = 0;
        let mut errors: usize = 0;
        let code = unsafe {
            ffi::zvec_collection_insert(
                self.ptr,
                doc_ptrs.as_ptr() as *mut *const ffi::zvec_doc_t,
                doc_ptrs.len(),
                &mut success,
                &mut errors,
            )
        };
        check_error(code as c_int)?;
        Ok((success, errors))
    }

    /// Upsert a single document into the collection.
    ///
    /// This is a convenience method for upserting one document.
    /// See [`upsert`] for batch operations.
    pub fn upsert_one(&self, doc: Doc) -> Result<WriteResults> {
        self.upsert(&[doc])
    }

    /// Insert a single document into the collection.
    ///
    /// This is a convenience method for inserting one document.
    /// See [`insert`] for batch operations.
    pub fn insert_one(&self, doc: Doc) -> Result<WriteResults> {
        self.insert(&[doc])
    }

    /// Upsert documents into the collection.
    ///
    /// If a document with the same primary key exists, it will be updated.
    /// Otherwise, it will be inserted.
    pub fn upsert(&self, docs: &[Doc]) -> Result<WriteResults> {
        let doc_ptrs: Vec<*const ffi::zvec_doc_t> = docs
            .iter()
            .map(|d| d.ptr as *const ffi::zvec_doc_t)
            .collect();
        let mut results: *mut ffi::zvec_write_result_t = ptr::null_mut();
        let mut result_count: usize = 0;
        let code = unsafe {
            ffi::zvec_collection_upsert_with_results(
                self.ptr,
                doc_ptrs.as_ptr() as *mut *const ffi::zvec_doc_t,
                doc_ptrs.len(),
                &mut results,
                &mut result_count,
            )
        };
        check_error(code as c_int)?;
        Ok(WriteResults::from_raw(results, result_count))
    }

    /// Update existing documents in the collection.
    ///
    /// Documents must already exist in the collection.
    pub fn update(&self, docs: &[Doc]) -> Result<WriteResults> {
        let doc_ptrs: Vec<*const ffi::zvec_doc_t> = docs
            .iter()
            .map(|d| d.ptr as *const ffi::zvec_doc_t)
            .collect();
        let mut results: *mut ffi::zvec_write_result_t = ptr::null_mut();
        let mut result_count: usize = 0;
        let code = unsafe {
            ffi::zvec_collection_update_with_results(
                self.ptr,
                doc_ptrs.as_ptr() as *mut *const ffi::zvec_doc_t,
                doc_ptrs.len(),
                &mut results,
                &mut result_count,
            )
        };
        check_error(code as c_int)?;
        Ok(WriteResults::from_raw(results, result_count))
    }

    /// Update a single document in the collection.
    ///
    /// This is a convenience method for updating one document.
    /// See [`update`] for batch operations.
    pub fn update_one(&self, doc: Doc) -> Result<WriteResults> {
        self.update(&[doc])
    }

    /// Delete documents by primary key.
    pub fn delete(&self, pks: &[&str]) -> Result<WriteResults> {
        let pk_cstrings: Vec<CString> = pks
            .iter()
            .map(|pk| CString::new(*pk).expect("primary key contains NUL byte"))
            .collect();
        let pk_ptrs: Vec<*const std::os::raw::c_char> =
            pk_cstrings.iter().map(|pk| pk.as_ptr()).collect();
        let mut results: *mut ffi::zvec_write_result_t = ptr::null_mut();
        let mut result_count: usize = 0;
        let code = unsafe {
            ffi::zvec_collection_delete_with_results(
                self.ptr,
                pk_ptrs.as_ptr(),
                pk_ptrs.len(),
                &mut results,
                &mut result_count,
            )
        };
        check_error(code as c_int)?;
        Ok(WriteResults::from_raw(results, result_count))
    }

    /// Delete a single document by primary key.
    ///
    /// This is a convenience method for deleting one document.
    /// See [`delete`] for batch operations.
    pub fn delete_one(&self, pk: &str) -> Result<WriteResults> {
        self.delete(&[pk])
    }

    /// Delete documents matching a filter expression.
    pub fn delete_by_filter(&self, filter: &str) -> Result<()> {
        let filter_c = CString::new(filter).expect("filter expression contains NUL byte");
        let code = unsafe { ffi::zvec_collection_delete_by_filter(self.ptr, filter_c.as_ptr()) };
        check_error(code as c_int)
    }

    /// Execute a vector similarity search query.
    ///
    /// Returns a [`DocList`] containing the matching documents.
    pub fn query(&self, query: VectorQuery) -> Result<DocList> {
        let mut docs: *mut *mut ffi::zvec_doc_t = ptr::null_mut();
        let mut count: usize = 0;
        let code =
            unsafe { ffi::zvec_collection_query(self.ptr, query.ptr, &mut docs, &mut count) };
        check_error(code as c_int)?;
        Ok(DocList::from_raw(docs, count))
    }

    /// Execute a grouped vector similarity search query.
    ///
    /// Groups results by a specified field value. Uses the `zvecgb_*` shim
    /// (group-by is not part of the upstream v0.5.0 C API).
    pub fn group_by_query(&self, query: GroupByVectorQuery) -> Result<GroupResults> {
        let mut out: *mut ffi::zvecgb_group_results_t = ptr::null_mut();
        let code = unsafe {
            ffi::zvecgb_collection_group_by_query(
                self.ptr as *mut std::os::raw::c_void,
                query.ptr,
                &mut out,
            )
        };
        check_error(code as c_int)?;
        Ok(GroupResults::from_ptr(out))
    }

    /// Execute a multi-query (hybrid search) combining multiple
    /// sub-queries with optional re-ranking.
    ///
    /// Returns a [`DocList`] containing the reranked top-K documents.
    pub fn multi_query(&self, query: &crate::MultiQuery) -> Result<DocList> {
        let mut docs: *mut *mut ffi::zvec_doc_t = ptr::null_mut();
        let mut count: usize = 0;
        let code = unsafe {
            ffi::zvec_collection_multi_query(self.ptr, query.as_ptr(), &mut docs, &mut count)
        };
        check_error(code as c_int)?;
        Ok(DocList::from_raw(docs, count))
    }

    /// Fetch documents by primary key.
    ///
    /// Returns a [`DocMap`] mapping primary keys to documents.
    pub fn fetch(&self, pks: &[&str]) -> Result<DocMap> {
        let pk_cstrings: Vec<CString> = pks
            .iter()
            .map(|pk| CString::new(*pk).expect("primary key contains NUL byte"))
            .collect();
        let pk_ptrs: Vec<*const std::os::raw::c_char> =
            pk_cstrings.iter().map(|pk| pk.as_ptr()).collect();
        let mut docs: *mut *mut ffi::zvec_doc_t = ptr::null_mut();
        let mut found: usize = 0;
        let code = unsafe {
            ffi::zvec_collection_fetch(
                self.ptr,
                pk_ptrs.as_ptr(),
                pk_ptrs.len(),
                ptr::null(),
                0,
                true,
                &mut docs,
                &mut found,
            )
        };
        check_error(code as c_int)?;
        Ok(DocMap::from_raw(docs, found))
    }

    /// Create an index on a vector field.
    ///
    /// # Arguments
    ///
    /// * `column_name` - Name of the vector field to index
    /// * `params` - Index parameters (HNSW, IVF, FLAT, etc.)
    pub fn create_index(&self, column_name: &str, params: IndexParams) -> Result<()> {
        let column_c = CString::new(column_name).expect("column name contains NUL byte");
        let code =
            unsafe { ffi::zvec_collection_create_index(self.ptr, column_c.as_ptr(), params.ptr) };
        check_error(code as c_int)
    }

    /// Drop an index from a column.
    pub fn drop_index(&self, column_name: &str) -> Result<()> {
        let column_c = CString::new(column_name).expect("column name contains NUL byte");
        let code = unsafe { ffi::zvec_collection_drop_index(self.ptr, column_c.as_ptr()) };
        check_error(code as c_int)
    }

    /// Optimize the collection for better search performance.
    pub fn optimize(&self) -> Result<()> {
        let code = unsafe { ffi::zvec_collection_optimize(self.ptr) };
        check_error(code as c_int)
    }

    /// Flush pending writes to disk.
    pub fn flush(&self) -> Result<()> {
        let code = unsafe { ffi::zvec_collection_flush(self.ptr) };
        check_error(code as c_int)
    }

    /// Get collection statistics.
    pub fn stats(&self) -> Result<CollectionStats> {
        let mut stats_ptr: *mut ffi::zvec_collection_stats_t = ptr::null_mut();
        let code = unsafe { ffi::zvec_collection_get_stats(self.ptr, &mut stats_ptr) };
        check_error(code as c_int)?;

        if stats_ptr.is_null() {
            return Err(crate::error::Error::InternalError(
                "Failed to get collection stats: null pointer".into(),
            ));
        }

        Ok(CollectionStats { ptr: stats_ptr })
    }

    /// Get the collection schema. Returns a non-owning schema wrapper.
    pub fn schema(&self) -> Result<CollectionSchema> {
        let mut schema_ptr: *mut ffi::zvec_collection_schema_t = ptr::null_mut();
        let code = unsafe { ffi::zvec_collection_get_schema(self.ptr, &mut schema_ptr) };
        check_error(code as c_int)?;

        if schema_ptr.is_null() {
            return Err(crate::error::Error::InternalError(
                "Failed to get collection schema: null pointer".into(),
            ));
        }

        Ok(CollectionSchema::from_ptr(schema_ptr))
    }

    /// Add a new column to the collection.
    pub fn add_column(&self, column_schema: FieldSchema, expression: Option<&str>) -> Result<()> {
        let expr_c = expression.map(|e| CString::new(e).expect("expression contains NUL byte"));
        let expr_ptr = expr_c.as_ref().map(|e| e.as_ptr()).unwrap_or(ptr::null());
        let code =
            unsafe { ffi::zvec_collection_add_column(self.ptr, column_schema.ptr, expr_ptr) };
        check_error(code as c_int)
    }

    /// Drop a column from the collection.
    pub fn drop_column(&self, column_name: &str) -> Result<()> {
        let column_c = CString::new(column_name).expect("column name contains NUL byte");
        let code = unsafe { ffi::zvec_collection_drop_column(self.ptr, column_c.as_ptr()) };
        check_error(code as c_int)
    }

    /// Alter a column in the collection.
    pub fn alter_column(
        &self,
        column_name: &str,
        rename: Option<&str>,
        new_column_schema: Option<FieldSchema>,
    ) -> Result<()> {
        let rename_c = rename.map(|r| CString::new(r).expect("rename contains NUL byte"));
        let rename_ptr = rename_c.as_ref().map(|r| r.as_ptr()).unwrap_or(ptr::null());
        let new_schema_ptr = new_column_schema
            .as_ref()
            .map(|s| s.ptr as *const ffi::zvec_field_schema_t)
            .unwrap_or(ptr::null());
        let column_c = CString::new(column_name).expect("column name contains NUL byte");
        let code = unsafe {
            ffi::zvec_collection_alter_column(
                self.ptr,
                column_c.as_ptr(),
                rename_ptr,
                new_schema_ptr,
            )
        };
        check_error(code as c_int)
    }

    /// Destroy the collection and delete all on-disk storage.
    ///
    /// Consumes self. Distinct from `Drop`, which merely closes the handle.
    pub fn destroy(self) -> Result<()> {
        let code = unsafe { ffi::zvec_collection_destroy(self.ptr) };
        // Prevent Drop from running on the consumed handle.
        std::mem::forget(self);
        check_error(code as c_int)
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Close the handle only — do NOT delete on-disk storage.
            // (`zvec_collection_destroy` would delete storage; that lives in
            // the explicit `destroy()` method above.)
            unsafe { ffi::zvec_collection_close(self.ptr) };
        }
    }
}

/// Parameters for creating an index on a vector field.
///
/// In zvec v0.5.0 index params are built via `zvec_index_params_create(type)`
/// followed by typed setters, rather than the old all-args constructors.
///
/// # Index Types
///
/// - **HNSW**: Fast approximate search using hierarchical navigable small world graphs
/// - **IVF**: Inverted file index, good for large datasets
/// - **FLAT**: Brute force search, exact results
/// - **INVERT**: Inverted index for scalar fields
///
/// # Example
///
/// ```rust,no_run
/// use zvec_bindings::{IndexParams, MetricType, QuantizeType};
///
/// // HNSW index with L2 distance
/// let params = IndexParams::hnsw(16, 200, MetricType::L2, QuantizeType::Undefined);
///
/// // Flat index with cosine similarity
/// let params = IndexParams::flat(MetricType::Cosine, QuantizeType::Undefined);
/// ```
pub struct IndexParams {
    pub(crate) ptr: *mut ffi::zvec_index_params_t,
}

impl IndexParams {
    /// Create HNSW index parameters.
    ///
    /// # Arguments
    ///
    /// * `m` - Number of connections per node (typically 8-64)
    /// * `ef_construction` - Size of dynamic candidate list during construction (typically 100-400)
    /// * `metric` - Distance metric (L2, Cosine, etc.)
    /// * `quantize` - Quantization type for compression
    pub fn hnsw(m: i32, ef_construction: i32, metric: MetricType, quantize: QuantizeType) -> Self {
        Self::build(crate::types::IndexType::Hnsw, |ptr| {
            let _ = check_error(unsafe {
                ffi::zvec_index_params_set_hnsw_params(ptr, m, ef_construction)
            } as c_int);
            let _ =
                check_error(
                    unsafe { ffi::zvec_index_params_set_metric_type(ptr, metric.into()) } as c_int,
                );
            let _ = check_error(unsafe {
                ffi::zvec_index_params_set_quantize_type(ptr, quantize.into())
            } as c_int);
        })
    }

    /// Create IVF index parameters.
    ///
    /// # Arguments
    ///
    /// * `n_list` - Number of clusters/inverted lists
    /// * `n_iters` - Number of k-means iterations
    /// * `use_soar` - Whether to use SOAR optimization
    /// * `metric` - Distance metric
    /// * `quantize` - Quantization type
    pub fn ivf(
        n_list: i32,
        n_iters: i32,
        use_soar: bool,
        metric: MetricType,
        quantize: QuantizeType,
    ) -> Self {
        Self::build(crate::types::IndexType::Ivf, |ptr| {
            let _ = check_error(unsafe {
                ffi::zvec_index_params_set_ivf_params(ptr, n_list, n_iters, use_soar)
            } as c_int);
            let _ =
                check_error(
                    unsafe { ffi::zvec_index_params_set_metric_type(ptr, metric.into()) } as c_int,
                );
            let _ = check_error(unsafe {
                ffi::zvec_index_params_set_quantize_type(ptr, quantize.into())
            } as c_int);
        })
    }

    /// Create FLAT (brute force) index parameters.
    ///
    /// # Arguments
    ///
    /// * `metric` - Distance metric
    /// * `quantize` - Quantization type
    pub fn flat(metric: MetricType, quantize: QuantizeType) -> Self {
        Self::build(crate::types::IndexType::Flat, |ptr| {
            let _ =
                check_error(
                    unsafe { ffi::zvec_index_params_set_metric_type(ptr, metric.into()) } as c_int,
                );
            let _ = check_error(unsafe {
                ffi::zvec_index_params_set_quantize_type(ptr, quantize.into())
            } as c_int);
        })
    }

    /// Create inverted index parameters for scalar fields.
    ///
    /// # Arguments
    ///
    /// * `enable_range_optimization` - Whether to optimize range queries
    pub fn invert(enable_range_optimization: bool) -> Self {
        Self::build(crate::types::IndexType::Invert, |ptr| {
            let _ = check_error(unsafe {
                ffi::zvec_index_params_set_invert_params(ptr, enable_range_optimization, false)
            } as c_int);
        })
    }

    /// Create Full-Text Search index parameters.
    ///
    /// # Arguments
    ///
    /// * `tokenizer_name` - Tokenizer pipeline name (e.g. `"standard"`
    ///   for English/whitespace, `"jieba"` for Chinese). The jieba
    ///   tokenizer requires a dict dir via
    ///   [`set_default_jieba_dict_dir`](crate::set_default_jieba_dict_dir)
    ///   or the `ZVEC_JIEBA_DICT_DIR` environment variable.
    /// * `filters` - Optional list of token filter names (e.g.
    ///   `"lowercase"`). Pass `None` or an empty slice for no filters.
    /// * `extra_params` - Optional JSON-style extra parameters forwarded
    ///   to the tokenizer. Pass `None` to omit.
    ///
    /// Returns an error if the underlying allocation or setter fails
    /// (e.g. wrong index type, invalid tokenizer name).
    pub fn fts(
        tokenizer_name: &str,
        filters: Option<&[&str]>,
        extra_params: Option<&str>,
    ) -> Result<Self> {
        let ptr = unsafe { ffi::zvec_index_params_create(crate::types::IndexType::Fts.into()) };
        if ptr.is_null() {
            return Err(crate::error::Error::InternalError(
                "zvec_index_params_create returned null".into(),
            ));
        }

        let tokenizer_c = CString::new(tokenizer_name)
            .map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))?;
        let extra_c = match extra_params {
            Some(s) => Some(
                CString::new(s).map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))?,
            ),
            None => None,
        };
        let extra_ptr = extra_c
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null());

        // Build the optional filters string array. If `filters` is None
        // or empty we pass a null pointer (upstream keeps current value).
        let filters_array: Option<Vec<CString>> = filters.map(|fs| {
            fs.iter()
                .map(|f| CString::new(*f).expect("filter name contains NUL byte"))
                .collect()
        });

        let (filters_ptr, _filters_owned) = match &filters_array {
            Some(arr) if !arr.is_empty() => {
                let sa = unsafe { ffi::zvec_string_array_create(arr.len()) };
                if sa.is_null() {
                    return Err(crate::error::Error::InternalError(
                        "zvec_string_array_create returned null".into(),
                    ));
                }
                for (i, cstr) in arr.iter().enumerate() {
                    unsafe { ffi::zvec_string_array_add(sa, i, cstr.as_ptr()) };
                }
                (sa, true)
            }
            _ => (
                std::ptr::null::<ffi::zvec_string_array_t>() as *mut ffi::zvec_string_array_t,
                false,
            ),
        };

        let code = unsafe {
            ffi::zvec_index_params_set_fts_params(
                ptr,
                tokenizer_c.as_ptr(),
                filters_ptr as *const ffi::zvec_string_array_t,
                extra_ptr,
            )
        };
        let result = check_error(code as c_int);

        // Always destroy the string array if we allocated one — upstream
        // copies the entries during set_fts_params.
        if !filters_ptr.is_null() {
            unsafe { ffi::zvec_string_array_destroy(filters_ptr as *mut ffi::zvec_string_array_t) };
        }

        result?;
        Ok(Self { ptr })
    }

    /// Low-level builder: create params of the given type then run `configure`.
    fn build<F: FnOnce(*mut ffi::zvec_index_params_t)>(
        index_type: IndexType,
        configure: F,
    ) -> Self {
        let ptr = unsafe { ffi::zvec_index_params_create(index_type.into()) };
        configure(ptr);
        Self { ptr }
    }

    /// Get the index type.
    pub fn index_type(&self) -> IndexType {
        unsafe { ffi::zvec_index_params_get_type(self.ptr).into() }
    }
}

impl Drop for IndexParams {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_index_params_destroy(self.ptr) };
        }
    }
}

pub struct CollectionOptions {
    ptr: *mut ffi::zvec_collection_options_t,
}

impl CollectionOptions {
    pub fn new() -> Self {
        let ptr = unsafe { ffi::zvec_collection_options_create() };
        Self { ptr }
    }

    pub fn read_only(self, read_only: bool) -> Self {
        let _ =
            check_error(
                unsafe { ffi::zvec_collection_options_set_read_only(self.ptr, read_only) } as c_int,
            );
        self
    }

    pub fn enable_mmap(self, enable: bool) -> Self {
        let _ =
            check_error(
                unsafe { ffi::zvec_collection_options_set_enable_mmap(self.ptr, enable) } as c_int,
            );
        self
    }

    /// Set the maximum write buffer size in bytes.
    pub fn max_buffer_size(self, size: usize) -> Self {
        let _ =
            check_error(
                unsafe { ffi::zvec_collection_options_set_max_buffer_size(self.ptr, size) }
                    as c_int,
            );
        self
    }
}

impl Default for CollectionOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CollectionOptions {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_collection_options_destroy(self.ptr) };
        }
    }
}

// SAFETY: Collection wraps a raw pointer to zvec C++ object.
// The underlying zvec library uses internal mutexes (schema_handle_mtx_, write_mtx_)
// for thread safety. Query operations are const and thread-safe.
// This impl allows Collection to be sent between threads and wrapped in Arc<RwLock>.
unsafe impl Send for Collection {}

// SAFETY: Collection is safe to share between threads because:
// 1. The underlying zvec C++ object uses internal mutexes for thread safety
// 2. Query operations (const methods) are thread-safe by design
// 3. Write operations use internal locking (write_mtx_)
unsafe impl Sync for Collection {}
