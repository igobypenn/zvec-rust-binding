use std::ffi::CString;
use std::path::Path;
use std::ptr;

use crate::doc::{Doc, DocList, DocMap, WriteResults};
use crate::error::{check_status, Result};
use crate::ffi;
use crate::query::{GroupByVectorQuery, GroupResults, VectorQuery};
use crate::schema::{CollectionSchema, FieldSchema};
use crate::types::{IndexType, MetricType, QuantizeType};

pub struct CollectionStats {
    pub doc_count: u64,
    pub memory_usage: u64,
    pub json_details: Option<String>,
}

impl CollectionStats {
    pub fn doc_count(&self) -> u64 {
        self.doc_count
    }

    pub fn memory_usage(&self) -> u64 {
        self.memory_usage
    }

    pub fn json_details(&self) -> Option<&str> {
        self.json_details.as_deref()
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
/// schema.add_field(VectorSchema::fp32("embedding", 128).into())?;
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
}

impl Collection {
    pub fn create_and_open<P: AsRef<Path>>(path: P, schema: CollectionSchema) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        let path_c = CString::new(path_str).unwrap();

        let mut status: ffi::zvec_status_t = unsafe { std::mem::zeroed() };
        let ptr = unsafe {
            ffi::zvec_collection_create_and_open(
                path_c.as_ptr(),
                schema.ptr,
                ptr::null_mut(),
                &mut status,
            )
        };

        check_status(status)?;

        if ptr.is_null() {
            return Err(crate::error::Error::InternalError(
                "Failed to create collection: null pointer".into(),
            ));
        }

        Ok(Self { ptr })
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        let path_c = CString::new(path_str).unwrap();

        let mut status: ffi::zvec_status_t = unsafe { std::mem::zeroed() };
        let ptr =
            unsafe { ffi::zvec_collection_open(path_c.as_ptr(), ptr::null_mut(), &mut status) };

        check_status(status)?;

        if ptr.is_null() {
            return Err(crate::error::Error::InternalError(
                "Failed to open collection: null pointer".into(),
            ));
        }

        Ok(Self { ptr })
    }

    /// Get the filesystem path where this collection is stored.
    pub fn path(&self) -> Result<String> {
        let mut path_ptr: *const std::os::raw::c_char = ptr::null();
        let status = unsafe { ffi::zvec_collection_path(self.ptr, &mut path_ptr) };
        check_status(status)?;

        if path_ptr.is_null() {
            return Ok(String::new());
        }

        Ok(unsafe {
            std::ffi::CStr::from_ptr(path_ptr)
                .to_string_lossy()
                .into_owned()
        })
    }

    /// Insert documents into the collection.
    ///
    /// Returns a [`WriteResults`] indicating the success or failure of each insert.
    pub fn insert(&self, docs: &[Doc]) -> Result<WriteResults> {
        let mut doc_ptrs: Vec<*mut ffi::zvec_doc_t> = docs.iter().map(|d| d.ptr).collect();
        let mut results: ffi::zvec_write_results_t = unsafe { std::mem::zeroed() };

        let status = unsafe {
            ffi::zvec_collection_insert(
                self.ptr,
                doc_ptrs.as_mut_ptr(),
                doc_ptrs.len(),
                &mut results,
            )
        };

        check_status(status)?;
        Ok(WriteResults { inner: results })
    }

    /// Upsert documents into the collection.
    ///
    /// If a document with the same primary key exists, it will be updated.
    /// Otherwise, it will be inserted.
    pub fn upsert(&self, docs: &[Doc]) -> Result<WriteResults> {
        let mut doc_ptrs: Vec<*mut ffi::zvec_doc_t> = docs.iter().map(|d| d.ptr).collect();
        let mut results: ffi::zvec_write_results_t = unsafe { std::mem::zeroed() };

        let status = unsafe {
            ffi::zvec_collection_upsert(
                self.ptr,
                doc_ptrs.as_mut_ptr(),
                doc_ptrs.len(),
                &mut results,
            )
        };

        check_status(status)?;
        Ok(WriteResults { inner: results })
    }

    /// Update existing documents in the collection.
    ///
    /// Documents must already exist in the collection.
    pub fn update(&self, docs: &[Doc]) -> Result<WriteResults> {
        let mut doc_ptrs: Vec<*mut ffi::zvec_doc_t> = docs.iter().map(|d| d.ptr).collect();
        let mut results: ffi::zvec_write_results_t = unsafe { std::mem::zeroed() };

        let status = unsafe {
            ffi::zvec_collection_update(
                self.ptr,
                doc_ptrs.as_mut_ptr(),
                doc_ptrs.len(),
                &mut results,
            )
        };

        check_status(status)?;
        Ok(WriteResults { inner: results })
    }

    /// Delete documents by primary key.
    pub fn delete(&self, pks: &[&str]) -> Result<WriteResults> {
        let pk_cstrings: Vec<CString> = pks.iter().map(|pk| CString::new(*pk).unwrap()).collect();
        let mut pk_ptrs: Vec<*const std::os::raw::c_char> =
            pk_cstrings.iter().map(|pk| pk.as_ptr()).collect();
        let mut results: ffi::zvec_write_results_t = unsafe { std::mem::zeroed() };

        let status = unsafe {
            ffi::zvec_collection_delete(self.ptr, pk_ptrs.as_mut_ptr(), pk_ptrs.len(), &mut results)
        };

        check_status(status)?;
        Ok(WriteResults { inner: results })
    }

    /// Delete documents matching a filter expression.
    pub fn delete_by_filter(&self, filter: &str) -> Result<()> {
        let filter_c = CString::new(filter).unwrap();
        let status = unsafe { ffi::zvec_collection_delete_by_filter(self.ptr, filter_c.as_ptr()) };
        check_status(status)
    }

    /// Execute a vector similarity search query.
    ///
    /// Returns a [`DocList`] containing the matching documents.
    pub fn query(&self, query: VectorQuery) -> Result<DocList> {
        let mut results: ffi::zvec_doc_list_t = unsafe { std::mem::zeroed() };
        let status = unsafe { ffi::zvec_collection_query(self.ptr, query.ptr, &mut results) };
        check_status(status)?;
        Ok(DocList { inner: results })
    }

    /// Execute a grouped vector similarity search query.
    ///
    /// Groups results by a specified field value.
    pub fn group_by_query(&self, query: GroupByVectorQuery) -> Result<GroupResults> {
        let mut results: ffi::zvec_group_results_t = unsafe { std::mem::zeroed() };
        let status =
            unsafe { ffi::zvec_collection_group_by_query(self.ptr, query.ptr, &mut results) };
        check_status(status)?;
        Ok(GroupResults { inner: results })
    }

    /// Fetch documents by primary key.
    ///
    /// Returns a [`DocMap`] mapping primary keys to documents.
    pub fn fetch(&self, pks: &[&str]) -> Result<DocMap> {
        let pk_cstrings: Vec<CString> = pks.iter().map(|pk| CString::new(*pk).unwrap()).collect();
        let mut pk_ptrs: Vec<*const std::os::raw::c_char> =
            pk_cstrings.iter().map(|pk| pk.as_ptr()).collect();
        let mut results: ffi::zvec_doc_map_t = unsafe { std::mem::zeroed() };

        let status = unsafe {
            ffi::zvec_collection_fetch(self.ptr, pk_ptrs.as_mut_ptr(), pk_ptrs.len(), &mut results)
        };

        check_status(status)?;
        Ok(DocMap { inner: results })
    }

    /// Create an index on a vector field.
    ///
    /// # Arguments
    ///
    /// * `column_name` - Name of the vector field to index
    /// * `params` - Index parameters (HNSW, IVF, FLAT, etc.)
    pub fn create_index(&self, column_name: &str, params: IndexParams) -> Result<()> {
        let column_c = CString::new(column_name).unwrap();
        let status = unsafe {
            ffi::zvec_collection_create_index(
                self.ptr,
                column_c.as_ptr(),
                params.ptr,
                ptr::null_mut(),
            )
        };
        check_status(status)
    }

    /// Drop an index from a column.
    pub fn drop_index(&self, column_name: &str) -> Result<()> {
        let column_c = CString::new(column_name).unwrap();
        let status = unsafe { ffi::zvec_collection_drop_index(self.ptr, column_c.as_ptr()) };
        check_status(status)
    }

    /// Optimize the collection for better search performance.
    pub fn optimize(&self) -> Result<()> {
        let status = unsafe { ffi::zvec_collection_optimize(self.ptr, ptr::null_mut()) };
        check_status(status)
    }

    /// Flush pending writes to disk.
    pub fn flush(&self) -> Result<()> {
        let status = unsafe { ffi::zvec_collection_flush(self.ptr) };
        check_status(status)
    }

    /// Get collection statistics.
    pub fn stats(&self) -> Result<CollectionStats> {
        let mut stats_ptr: *mut ffi::zvec_collection_stats_t = ptr::null_mut();
        let status = unsafe { ffi::zvec_collection_stats(self.ptr, &mut stats_ptr) };
        check_status(status)?;

        if stats_ptr.is_null() {
            return Err(crate::error::Error::InternalError(
                "Failed to get collection stats: null pointer".into(),
            ));
        }

        let stats = unsafe { &*stats_ptr };
        let doc_count = stats.doc_count;
        let memory_usage = stats.memory_usage;
        let json_details = if stats.json_details.is_null() {
            None
        } else {
            Some(unsafe {
                std::ffi::CStr::from_ptr(stats.json_details)
                    .to_string_lossy()
                    .into_owned()
            })
        };

        unsafe { ffi::zvec_collection_stats_free(stats_ptr) };

        Ok(CollectionStats {
            doc_count,
            memory_usage,
            json_details,
        })
    }

    /// Get the collection schema.
    pub fn schema(&self) -> Result<CollectionSchema> {
        let mut schema_ptr: *mut ffi::zvec_collection_schema_t = ptr::null_mut();
        let status = unsafe { ffi::zvec_collection_schema(self.ptr, &mut schema_ptr) };
        check_status(status)?;

        if schema_ptr.is_null() {
            return Err(crate::error::Error::InternalError(
                "Failed to get collection schema: null pointer".into(),
            ));
        }

        Ok(CollectionSchema::from_ptr(schema_ptr))
    }

    /// Add a new column to the collection.
    pub fn add_column(&self, column_schema: FieldSchema, expression: Option<&str>) -> Result<()> {
        let expr_c = expression.map(|e| CString::new(e).unwrap());
        let expr_ptr = expr_c.as_ref().map(|e| e.as_ptr()).unwrap_or(ptr::null());
        let status =
            unsafe { ffi::zvec_collection_add_column(self.ptr, column_schema.ptr, expr_ptr) };
        check_status(status)
    }

    /// Drop a column from the collection.
    pub fn drop_column(&self, column_name: &str) -> Result<()> {
        let column_c = CString::new(column_name).unwrap();
        let status = unsafe { ffi::zvec_collection_drop_column(self.ptr, column_c.as_ptr()) };
        check_status(status)
    }

    /// Alter a column in the collection.
    pub fn alter_column(
        &self,
        column_name: &str,
        rename: Option<&str>,
        new_column_schema: Option<FieldSchema>,
    ) -> Result<()> {
        let rename_c = rename.map(|r| CString::new(r).unwrap());
        let rename_ptr = rename_c.as_ref().map(|r| r.as_ptr()).unwrap_or(ptr::null());
        let new_schema_ptr = new_column_schema
            .as_ref()
            .map(|s| s.ptr)
            .unwrap_or(ptr::null_mut());
        let column_c = CString::new(column_name).unwrap();
        let status = unsafe {
            ffi::zvec_collection_alter_column(
                self.ptr,
                column_c.as_ptr(),
                rename_ptr,
                new_schema_ptr,
            )
        };
        check_status(status)
    }

    /// Destroy the collection and delete all data.
    pub fn destroy(self) -> Result<()> {
        let status = unsafe { ffi::zvec_collection_destroy_storage(self.ptr) };
        check_status(status)
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_collection_destroy(self.ptr) };
        }
    }
}

/// Parameters for creating an index on a vector field.
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
    ptr: *mut ffi::zvec_index_params_t,
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
        let ptr = unsafe {
            ffi::zvec_index_params_new_hnsw(m, ef_construction, metric.into(), quantize.into())
        };
        Self { ptr }
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
        let ptr = unsafe {
            ffi::zvec_index_params_new_ivf(
                n_list,
                n_iters,
                use_soar,
                metric.into(),
                quantize.into(),
            )
        };
        Self { ptr }
    }

    /// Create FLAT (brute force) index parameters.
    ///
    /// # Arguments
    ///
    /// * `metric` - Distance metric
    /// * `quantize` - Quantization type
    pub fn flat(metric: MetricType, quantize: QuantizeType) -> Self {
        let ptr = unsafe { ffi::zvec_index_params_new_flat(metric.into(), quantize.into()) };
        Self { ptr }
    }

    /// Create inverted index parameters for scalar fields.
    ///
    /// # Arguments
    ///
    /// * `enable_range_optimization` - Whether to optimize range queries
    pub fn invert(enable_range_optimization: bool) -> Self {
        let ptr = unsafe { ffi::zvec_index_params_new_invert(enable_range_optimization) };
        Self { ptr }
    }

    /// Get the index type.
    pub fn index_type(&self) -> IndexType {
        unsafe { ffi::zvec_index_params_type(self.ptr).into() }
    }
}

impl Drop for IndexParams {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_index_params_free(self.ptr) };
        }
    }
}

pub struct CollectionOptions {
    ptr: *mut ffi::zvec_collection_options_t,
}

impl CollectionOptions {
    pub fn new() -> Self {
        let ptr = unsafe { ffi::zvec_collection_options_new() };
        Self { ptr }
    }

    pub fn read_only(self, read_only: bool) -> Self {
        unsafe { ffi::zvec_collection_options_set_read_only(self.ptr, read_only) };
        self
    }

    pub fn enable_mmap(self, enable: bool) -> Self {
        unsafe { ffi::zvec_collection_options_set_enable_mmap(self.ptr, enable) };
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
            unsafe { ffi::zvec_collection_options_free(self.ptr) };
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
