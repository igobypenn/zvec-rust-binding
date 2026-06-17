use std::ffi::CString;
use std::os::raw::c_int;

use crate::error::{check_error, Result};
use crate::ffi;

// ===== HNSW query params =====

pub struct HnswQueryParam {
    pub(crate) ptr: *mut ffi::zvec_hnsw_query_params_t,
    ef_search: i32,
}

impl HnswQueryParam {
    pub fn new(ef_search: i32) -> Self {
        let ptr = unsafe { ffi::zvec_query_params_hnsw_create(ef_search, 0.0, false, false) };
        Self { ptr, ef_search }
    }

    pub fn ef_search(&self) -> i32 {
        self.ef_search
    }
}

impl Drop for HnswQueryParam {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_query_params_hnsw_destroy(self.ptr) };
        }
    }
}

// ===== IVF query params =====

pub struct IVFQueryParam {
    pub(crate) ptr: *mut ffi::zvec_ivf_query_params_t,
    nprobe: i32,
}

impl IVFQueryParam {
    pub fn new(nprobe: i32) -> Self {
        let ptr = unsafe { ffi::zvec_query_params_ivf_create(nprobe, false, 10.0) };
        Self { ptr, nprobe }
    }

    pub fn nprobe(&self) -> i32 {
        self.nprobe
    }
}

impl Drop for IVFQueryParam {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_query_params_ivf_destroy(self.ptr) };
        }
    }
}

// ===== Flat query params =====

/// Flat (brute-force) query parameters (zvec v0.5.0).
pub struct FlatQueryParam {
    pub(crate) ptr: *mut ffi::zvec_flat_query_params_t,
}

impl FlatQueryParam {
    pub fn new() -> Self {
        let ptr = unsafe { ffi::zvec_query_params_flat_create(false, 10.0) };
        Self { ptr }
    }
}

impl Default for FlatQueryParam {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FlatQueryParam {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_query_params_flat_destroy(self.ptr) };
        }
    }
}

// ===== FTS query params =====

/// Full-text-search query parameters (zvec v0.5.0).
pub struct FtsQueryParam {
    pub(crate) ptr: *mut ffi::zvec_fts_query_params_t,
}

impl FtsQueryParam {
    /// `default_operator` is "OR" or "AND" (case-insensitive); pass `None`
    /// to keep the library default.
    pub fn new(default_operator: Option<&str>) -> Self {
        let op_c = default_operator.map(|s| CString::new(s).expect("operator contains NUL byte"));
        let op_ptr = op_c
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null());
        let ptr = unsafe { ffi::zvec_query_params_fts_create(op_ptr) };
        Self { ptr }
    }
}

impl Drop for FtsQueryParam {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_query_params_fts_destroy(self.ptr) };
        }
    }
}

// ===== QueryParam enum =====

pub enum QueryParam {
    Hnsw(HnswQueryParam),
    IVF(IVFQueryParam),
    Flat(FlatQueryParam),
    Fts(FtsQueryParam),
}

// ===== VectorQuery =====

pub struct VectorQuery {
    pub(crate) ptr: *mut ffi::zvec_vector_query_t,
    id: Option<String>,
}

impl VectorQuery {
    pub fn new(field_name: &str) -> Self {
        let field_c = CString::new(field_name).expect("field name contains NUL byte");
        let ptr = unsafe { ffi::zvec_vector_query_create() };
        unsafe { ffi::zvec_vector_query_set_field_name(ptr, field_c.as_ptr()) };
        Self { ptr, id: None }
    }

    pub fn topk(self, topk: usize) -> Self {
        let code = unsafe { ffi::zvec_vector_query_set_topk(self.ptr, topk as c_int) };
        // topk setter does not fail for valid inputs; ignore non-fatal codes.
        let _ = check_error(code as c_int);
        self
    }

    pub fn filter(self, filter: &str) -> Self {
        let filter_c = CString::new(filter).expect("filter contains NUL byte");
        let code = unsafe { ffi::zvec_vector_query_set_filter(self.ptr, filter_c.as_ptr()) };
        let _ = check_error(code as c_int);
        self
    }

    pub fn include_vector(self, include: bool) -> Self {
        let code = unsafe { ffi::zvec_vector_query_set_include_vector(self.ptr, include) };
        let _ = check_error(code as c_int);
        self
    }

    pub fn include_doc_id(self, include: bool) -> Self {
        let code = unsafe { ffi::zvec_vector_query_set_include_doc_id(self.ptr, include) };
        let _ = check_error(code as c_int);
        self
    }

    pub fn output_fields(self, fields: &[&str]) -> Self {
        let fields_c: Vec<CString> = fields
            .iter()
            .map(|f| CString::new(*f).expect("output field name contains NUL byte"))
            .collect();
        let mut fields_ptr: Vec<*const std::os::raw::c_char> =
            fields_c.iter().map(|f| f.as_ptr()).collect();
        let code = unsafe {
            ffi::zvec_vector_query_set_output_fields(
                self.ptr,
                fields_ptr.as_mut_ptr(),
                fields_ptr.len(),
            )
        };
        let _ = check_error(code as c_int);
        self
    }

    pub fn query_params(self, params: QueryParam) -> Self {
        let code = match params {
            QueryParam::Hnsw(p) => {
                let ptr = p.ptr;
                std::mem::forget(p);
                unsafe { ffi::zvec_vector_query_set_hnsw_params(self.ptr, ptr) }
            }
            QueryParam::IVF(p) => {
                let ptr = p.ptr;
                std::mem::forget(p);
                unsafe { ffi::zvec_vector_query_set_ivf_params(self.ptr, ptr) }
            }
            QueryParam::Flat(p) => {
                let ptr = p.ptr;
                std::mem::forget(p);
                unsafe { ffi::zvec_vector_query_set_flat_params(self.ptr, ptr) }
            }
            QueryParam::Fts(p) => {
                let ptr = p.ptr;
                std::mem::forget(p);
                unsafe { ffi::zvec_vector_query_set_fts_params(self.ptr, ptr) }
            }
        };
        let _ = check_error(code as c_int);
        self
    }

    pub fn hnsw_params(self, ef_search: i32) -> Self {
        let params = HnswQueryParam::new(ef_search);
        self.query_params(QueryParam::Hnsw(params))
    }

    pub fn ivf_params(self, nprobe: i32) -> Self {
        let params = IVFQueryParam::new(nprobe);
        self.query_params(QueryParam::IVF(params))
    }

    /// Set the query vector (dense FP32). The data is copied internally.
    pub fn vector(self, vector: &[f32]) -> Result<Self> {
        let code = unsafe {
            ffi::zvec_vector_query_set_query_vector(
                self.ptr,
                vector.as_ptr() as *const std::os::raw::c_void,
                std::mem::size_of_val(vector),
            )
        };
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Attach a Full-Text Search payload to this vector query for hybrid
    /// search. The payload is copied internally by upstream, so `fts`
    /// is dropped on return regardless of success.
    ///
    /// Requires the collection to have an FTS index on the target field.
    pub fn fts(self, fts: crate::Fts) -> Result<Self> {
        let code = unsafe { ffi::zvec_vector_query_set_fts(self.ptr, fts.as_ptr()) };
        // Drop `fts` — upstream copied it; we own our wrapper either way.
        drop(fts);
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Set a sparse query vector (FP32).
    ///
    /// Upstream packs sparse data as `[nnz: u32][indices: u32*nnz][values:
    /// f32*nnz]` into the generic `set_query_vector` byte buffer.
    pub fn sparse_vector(self, indices: &[u32], values: &[f32]) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(crate::error::Error::InvalidArgument(
                "indices and values must have same length".into(),
            ));
        }
        let nnz = indices.len();
        let mut buf: Vec<u8> = Vec::with_capacity(
            std::mem::size_of::<u32>()
                + std::mem::size_of_val(indices)
                + nnz * std::mem::size_of::<f32>(),
        );
        buf.extend_from_slice(&(nnz as u32).to_ne_bytes());
        for &idx in indices {
            buf.extend_from_slice(&idx.to_ne_bytes());
        }
        for &val in values {
            buf.extend_from_slice(&val.to_ne_bytes());
        }
        let code = unsafe {
            ffi::zvec_vector_query_set_query_vector(
                self.ptr,
                buf.as_ptr() as *const std::os::raw::c_void,
                buf.len(),
            )
        };
        check_error(code as c_int)?;
        Ok(self)
    }

    pub fn id(self, id: impl Into<String>) -> Self {
        let ptr = self.ptr;
        std::mem::forget(self);
        Self {
            ptr,
            id: Some(id.into()),
        }
    }

    pub fn has_id(&self) -> bool {
        self.id.is_some()
    }

    pub fn has_vector(&self) -> bool {
        self.id.is_none()
    }

    pub fn get_id(&self) -> Option<&str> {
        self.id.as_deref()
    }
}

impl Drop for VectorQuery {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_vector_query_destroy(self.ptr) };
        }
    }
}

// ===== GroupByVectorQuery (uses the zvecgb_* shim) =====

pub struct GroupByVectorQuery {
    pub(crate) ptr: *mut ffi::zvecgb_group_by_vector_query_t,
}

impl GroupByVectorQuery {
    pub fn new(field_name: &str) -> Self {
        let field_c = CString::new(field_name).expect("field name contains NUL byte");
        let ptr = unsafe { ffi::zvecgb_group_by_vector_query_create(field_c.as_ptr()) };
        Self { ptr }
    }

    pub fn group_by(self, field_name: &str) -> Self {
        let field_c = CString::new(field_name).expect("field name contains NUL byte");
        unsafe { ffi::zvecgb_group_by_vector_query_set_group_by_field(self.ptr, field_c.as_ptr()) };
        self
    }

    pub fn group_count(self, count: u32) -> Self {
        unsafe { ffi::zvecgb_group_by_vector_query_set_group_count(self.ptr, count) };
        self
    }

    pub fn group_topk(self, topk: u32) -> Self {
        unsafe { ffi::zvecgb_group_by_vector_query_set_group_topk(self.ptr, topk) };
        self
    }

    pub fn filter(self, filter: &str) -> Self {
        let filter_c = CString::new(filter).expect("filter contains NUL byte");
        unsafe { ffi::zvecgb_group_by_vector_query_set_filter(self.ptr, filter_c.as_ptr()) };
        self
    }

    pub fn output_fields(self, fields: &[&str]) -> Self {
        let fields_c: Vec<CString> = fields
            .iter()
            .map(|f| CString::new(*f).expect("output field name contains NUL byte"))
            .collect();
        let mut fields_ptr: Vec<*const std::os::raw::c_char> =
            fields_c.iter().map(|f| f.as_ptr()).collect();
        unsafe {
            ffi::zvecgb_group_by_vector_query_set_output_fields(
                self.ptr,
                fields_ptr.as_mut_ptr(),
                fields_ptr.len(),
            )
        };
        self
    }

    pub fn vector(self, vector: &[f32]) -> Result<Self> {
        let code = unsafe {
            ffi::zvecgb_group_by_vector_query_set_vector_fp32(
                self.ptr,
                vector.as_ptr(),
                vector.len(),
            )
        };
        if code == 0 {
            Ok(self)
        } else {
            Err(crate::error::Error::InternalError(format!(
                "group_by_vector_query set_vector failed (code={})",
                code
            )))
        }
    }
}

impl Drop for GroupByVectorQuery {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvecgb_group_by_vector_query_destroy(self.ptr) };
        }
    }
}

// ===== GroupResults =====

/// Owning wrapper around `zvecgb_group_results_t`.
pub struct GroupResults {
    pub(crate) ptr: *mut ffi::zvecgb_group_results_t,
}

impl GroupResults {
    /// Construct from a raw pointer. Takes ownership; the pointer will be
    /// destroyed on drop.
    pub(crate) fn from_ptr(ptr: *mut ffi::zvecgb_group_results_t) -> Self {
        Self { ptr }
    }

    pub fn len(&self) -> usize {
        unsafe { ffi::zvecgb_group_results_count(self.ptr) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<GroupResultRef<'_>> {
        let count = self.len();
        if index < count {
            Some(GroupResultRef {
                results: self.ptr,
                index,
                _marker: std::marker::PhantomData,
            })
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = GroupResultRef<'_>> + '_ {
        (0..self.len()).filter_map(|i| self.get(i))
    }
}

impl Drop for GroupResults {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvecgb_group_results_destroy(self.ptr) };
        }
    }
}

/// A borrowed view into one group inside [`GroupResults`].
pub struct GroupResultRef<'a> {
    results: *const ffi::zvecgb_group_results_t,
    index: usize,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> GroupResultRef<'a> {
    pub fn group_by_value(&self) -> &str {
        unsafe {
            let ptr = ffi::zvecgb_group_results_group_by_value(self.results, self.index);
            if ptr.is_null() {
                ""
            } else {
                std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
            }
        }
    }

    /// Returns the documents belonging to this group as a non-owning
    /// [`DocList`] borrow. The underlying storage is owned by the parent
    /// [`GroupResults`] and must outlive the returned `DocList`.
    pub fn docs(&self) -> crate::doc::DocList {
        unsafe {
            let ptr = ffi::zvecgb_group_results_docs_ptr(self.results, self.index)
                as *mut *mut ffi::zvec_doc_t;
            let count = ffi::zvecgb_group_results_docs_count(self.results, self.index);
            crate::doc::DocList::borrow_raw(ptr, count)
        }
    }
}

// SAFETY: GroupResults owns its FFI data and can be safely sent between threads.
unsafe impl Send for GroupResults {}
