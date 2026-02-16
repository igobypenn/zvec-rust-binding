use std::ffi::CString;

use crate::error::{check_status, Result};
use crate::ffi;

pub struct HnswQueryParam {
    pub(crate) ptr: *mut ffi::zvec_query_params_t,
    ef_search: i32,
}

impl HnswQueryParam {
    pub fn new(ef_search: i32) -> Self {
        let ptr = unsafe { ffi::zvec_query_params_new_hnsw(ef_search) };
        Self { ptr, ef_search }
    }

    pub fn ef_search(&self) -> i32 {
        self.ef_search
    }
}

impl Drop for HnswQueryParam {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_query_params_free(self.ptr) };
        }
    }
}

pub struct IVFQueryParam {
    pub(crate) ptr: *mut ffi::zvec_query_params_t,
    nprobe: i32,
}

impl IVFQueryParam {
    pub fn new(nprobe: i32) -> Self {
        let ptr = unsafe { ffi::zvec_query_params_new_ivf(nprobe) };
        Self { ptr, nprobe }
    }

    pub fn nprobe(&self) -> i32 {
        self.nprobe
    }
}

impl Drop for IVFQueryParam {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_query_params_free(self.ptr) };
        }
    }
}

pub enum QueryParam {
    Hnsw(HnswQueryParam),
    IVF(IVFQueryParam),
}

pub struct VectorQuery {
    pub(crate) ptr: *mut ffi::zvec_vector_query_t,
    id: Option<String>,
}

impl VectorQuery {
    pub fn new(field_name: &str) -> Self {
        let field_c = CString::new(field_name).unwrap();
        let ptr = unsafe { ffi::zvec_vector_query_new(field_c.as_ptr()) };
        Self { ptr, id: None }
    }

    pub fn topk(self, topk: usize) -> Self {
        unsafe { ffi::zvec_vector_query_set_topk(self.ptr, topk as std::os::raw::c_int) };
        self
    }

    pub fn filter(self, filter: &str) -> Self {
        let filter_c = CString::new(filter).unwrap();
        unsafe { ffi::zvec_vector_query_set_filter(self.ptr, filter_c.as_ptr()) };
        self
    }

    pub fn include_vector(self, include: bool) -> Self {
        unsafe { ffi::zvec_vector_query_set_include_vector(self.ptr, include) };
        self
    }

    pub fn include_doc_id(self, include: bool) -> Self {
        unsafe { ffi::zvec_vector_query_set_include_doc_id(self.ptr, include) };
        self
    }

    pub fn output_fields(self, fields: &[&str]) -> Self {
        let fields_c: Vec<CString> = fields.iter().map(|f| CString::new(*f).unwrap()).collect();
        let mut fields_ptr: Vec<*const std::os::raw::c_char> =
            fields_c.iter().map(|f| f.as_ptr()).collect();
        unsafe {
            ffi::zvec_vector_query_set_output_fields(
                self.ptr,
                fields_ptr.as_mut_ptr(),
                fields_ptr.len(),
            )
        };
        self
    }

    pub fn query_params(self, params: QueryParam) -> Self {
        let ptr = match &params {
            QueryParam::Hnsw(p) => p.ptr,
            QueryParam::IVF(p) => p.ptr,
        };
        unsafe { ffi::zvec_vector_query_set_query_params(self.ptr, ptr) };
        std::mem::forget(params);
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

    pub fn vector(self, vector: &[f32]) -> Result<Self> {
        let status = unsafe {
            ffi::zvec_vector_query_set_vector_fp32(self.ptr, vector.as_ptr(), vector.len())
        };
        check_status(status)?;
        Ok(self)
    }

    pub fn sparse_vector(self, indices: &[u32], values: &[f32]) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(crate::error::Error::InvalidArgument(
                "indices and values must have same length".into(),
            ));
        }
        let status = unsafe {
            ffi::zvec_vector_query_set_sparse_vector_fp32(
                self.ptr,
                indices.as_ptr(),
                indices.len(),
                values.as_ptr(),
                values.len(),
            )
        };
        check_status(status)?;
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
            unsafe { ffi::zvec_vector_query_free(self.ptr) };
        }
    }
}

pub struct GroupByVectorQuery {
    pub(crate) ptr: *mut ffi::zvec_group_by_vector_query_t,
}

impl GroupByVectorQuery {
    pub fn new(field_name: &str) -> Self {
        let field_c = CString::new(field_name).unwrap();
        let ptr = unsafe { ffi::zvec_group_by_vector_query_new(field_c.as_ptr()) };
        Self { ptr }
    }

    pub fn group_by(self, field_name: &str) -> Self {
        let field_c = CString::new(field_name).unwrap();
        unsafe { ffi::zvec_group_by_vector_query_set_group_by_field(self.ptr, field_c.as_ptr()) };
        self
    }

    pub fn group_count(self, count: u32) -> Self {
        unsafe { ffi::zvec_group_by_vector_query_set_group_count(self.ptr, count) };
        self
    }

    pub fn group_topk(self, topk: u32) -> Self {
        unsafe { ffi::zvec_group_by_vector_query_set_group_topk(self.ptr, topk) };
        self
    }

    pub fn filter(self, filter: &str) -> Self {
        let filter_c = CString::new(filter).unwrap();
        unsafe { ffi::zvec_group_by_vector_query_set_filter(self.ptr, filter_c.as_ptr()) };
        self
    }

    pub fn output_fields(self, fields: &[&str]) -> Self {
        let fields_c: Vec<CString> = fields.iter().map(|f| CString::new(*f).unwrap()).collect();
        let mut fields_ptr: Vec<*const std::os::raw::c_char> =
            fields_c.iter().map(|f| f.as_ptr()).collect();
        unsafe {
            ffi::zvec_group_by_vector_query_set_output_fields(
                self.ptr,
                fields_ptr.as_mut_ptr(),
                fields_ptr.len(),
            )
        };
        self
    }

    pub fn vector(self, vector: &[f32]) -> Result<Self> {
        let status = unsafe {
            ffi::zvec_group_by_vector_query_set_vector_fp32(self.ptr, vector.as_ptr(), vector.len())
        };
        check_status(status)?;
        Ok(self)
    }
}

impl Drop for GroupByVectorQuery {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_group_by_vector_query_free(self.ptr) };
        }
    }
}

pub struct GroupResults {
    pub(crate) inner: ffi::zvec_group_results_t,
}

impl GroupResults {
    pub fn len(&self) -> usize {
        self.inner.count
    }

    pub fn is_empty(&self) -> bool {
        self.inner.count == 0
    }

    pub fn get(&self, index: usize) -> Option<GroupResultRef<'_>> {
        if index < self.inner.count {
            Some(GroupResultRef {
                inner: unsafe { &*self.inner.groups.add(index) },
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
        unsafe { ffi::zvec_group_results_free(&mut self.inner) };
    }
}

pub struct GroupResultRef<'a> {
    inner: &'a ffi::zvec_group_result_t,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> GroupResultRef<'a> {
    pub fn group_by_value(&self) -> &str {
        unsafe {
            if self.inner.group_by_value.is_null() {
                ""
            } else {
                std::ffi::CStr::from_ptr(self.inner.group_by_value)
                    .to_str()
                    .unwrap_or("")
            }
        }
    }

    pub fn docs(&self) -> &crate::doc::DocList {
        unsafe { std::mem::transmute(&self.inner.docs) }
    }
}

// SAFETY: GroupResults owns its FFI data and can be safely sent between threads.
unsafe impl Send for GroupResults {}
