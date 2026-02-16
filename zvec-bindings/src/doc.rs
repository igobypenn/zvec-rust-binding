use std::ffi::CString;
use std::ptr;

use crate::error::{check_status, Result};
use crate::ffi;

/// A document in a collection.
///
/// Documents contain a primary key and zero or more fields (scalar values,
/// dense vectors, or sparse vectors).
///
/// # Example
///
/// ```rust,no_run
/// use zvec_bindings::Doc;
///
/// let mut doc = Doc::id("doc_1");
/// doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
/// doc.set_int64("count", 42)?;
/// doc.set_string("name", "example")?;
/// # Ok::<(), zvec_bindings::Error>(())
/// ```
pub struct Doc {
    pub(crate) ptr: *mut ffi::zvec_doc_t,
}

impl Doc {
    /// Create a new empty document.
    pub fn new() -> Self {
        let ptr = unsafe { ffi::zvec_doc_new() };
        Self { ptr }
    }

    /// Create a new document with the given primary key.
    pub fn with_pk(pk: impl Into<String>) -> Self {
        let mut doc = Self::new();
        doc.set_pk(pk);
        doc
    }

    /// Create a new document with the given ID (alias for `with_pk`).
    pub fn id(id: impl Into<String>) -> Self {
        Self::with_pk(id)
    }

    /// Set the primary key and return self for chaining.
    pub fn with_pk_mut(mut self, pk: impl Into<String>) -> Self {
        self.set_pk(pk);
        self
    }

    /// Set a vector field and return self for chaining.
    pub fn with_vector(mut self, field: &str, vector: &[f32]) -> Result<Self> {
        self.set_vector(field, vector)?;
        Ok(self)
    }

    /// Set a string field and return self for chaining.
    pub fn with_string(mut self, field: &str, value: &str) -> Result<Self> {
        self.set_string(field, value)?;
        Ok(self)
    }

    /// Set a float field and return self for chaining.
    pub fn with_float(mut self, field: &str, value: f32) -> Result<Self> {
        self.set_float(field, value)?;
        Ok(self)
    }

    /// Set an int64 field and return self for chaining.
    pub fn with_int64(mut self, field: &str, value: i64) -> Result<Self> {
        self.set_int64(field, value)?;
        Ok(self)
    }

    /// Set the primary key.
    pub fn set_pk(&mut self, pk: impl Into<String>) {
        let pk_c = CString::new(pk.into()).unwrap();
        unsafe { ffi::zvec_doc_set_pk(self.ptr, pk_c.as_ptr()) };
    }

    /// Get the primary key.
    pub fn pk(&self) -> &str {
        unsafe {
            let ptr = ffi::zvec_doc_pk(self.ptr);
            if ptr.is_null() {
                ""
            } else {
                std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
            }
        }
    }

    pub fn score(&self) -> f32 {
        unsafe { ffi::zvec_doc_score(self.ptr) }
    }

    pub fn doc_id(&self) -> u64 {
        unsafe { ffi::zvec_doc_doc_id(self.ptr) }
    }

    pub fn set_bool(&mut self, field: &str, value: bool) -> Result<()> {
        let field_c = CString::new(field).unwrap();
        let status = unsafe { ffi::zvec_doc_set_bool(self.ptr, field_c.as_ptr(), value) };
        check_status(status)
    }

    pub fn set_int32(&mut self, field: &str, value: i32) -> Result<()> {
        let field_c = CString::new(field).unwrap();
        let status = unsafe { ffi::zvec_doc_set_int32(self.ptr, field_c.as_ptr(), value) };
        check_status(status)
    }

    pub fn set_int64(&mut self, field: &str, value: i64) -> Result<()> {
        let field_c = CString::new(field).unwrap();
        let status = unsafe { ffi::zvec_doc_set_int64(self.ptr, field_c.as_ptr(), value) };
        check_status(status)
    }

    pub fn set_float(&mut self, field: &str, value: f32) -> Result<()> {
        let field_c = CString::new(field).unwrap();
        let status = unsafe { ffi::zvec_doc_set_float(self.ptr, field_c.as_ptr(), value) };
        check_status(status)
    }

    pub fn set_double(&mut self, field: &str, value: f64) -> Result<()> {
        let field_c = CString::new(field).unwrap();
        let status = unsafe { ffi::zvec_doc_set_double(self.ptr, field_c.as_ptr(), value) };
        check_status(status)
    }

    pub fn set_string(&mut self, field: &str, value: &str) -> Result<()> {
        let field_c = CString::new(field).unwrap();
        let value_c = CString::new(value).unwrap();
        let status =
            unsafe { ffi::zvec_doc_set_string(self.ptr, field_c.as_ptr(), value_c.as_ptr()) };
        check_status(status)
    }

    pub fn set_vector(&mut self, field: &str, vector: &[f32]) -> Result<()> {
        let field_c = CString::new(field).unwrap();
        let status = unsafe {
            ffi::zvec_doc_set_vector_fp32(self.ptr, field_c.as_ptr(), vector.as_ptr(), vector.len())
        };
        check_status(status)
    }

    pub fn set_sparse_vector(
        &mut self,
        field: &str,
        indices: &[u32],
        values: &[f32],
    ) -> Result<()> {
        if indices.len() != values.len() {
            return Err(crate::error::Error::InvalidArgument(
                "indices and values must have same length".into(),
            ));
        }
        let field_c = CString::new(field).unwrap();
        let status = unsafe {
            ffi::zvec_doc_set_sparse_vector_fp32(
                self.ptr,
                field_c.as_ptr(),
                indices.as_ptr(),
                indices.len(),
                values.as_ptr(),
                values.len(),
            )
        };
        check_status(status)
    }

    pub fn get_bool(&self, field: &str) -> Option<bool> {
        let field_c = CString::new(field).unwrap();
        let mut value: bool = false;
        let found = unsafe { ffi::zvec_doc_get_bool(self.ptr, field_c.as_ptr(), &mut value) };
        if found {
            Some(value)
        } else {
            None
        }
    }

    pub fn get_int32(&self, field: &str) -> Option<i32> {
        let field_c = CString::new(field).unwrap();
        let mut value: i32 = 0;
        let found = unsafe { ffi::zvec_doc_get_int32(self.ptr, field_c.as_ptr(), &mut value) };
        if found {
            Some(value)
        } else {
            None
        }
    }

    pub fn get_int64(&self, field: &str) -> Option<i64> {
        let field_c = CString::new(field).unwrap();
        let mut value: i64 = 0;
        let found = unsafe { ffi::zvec_doc_get_int64(self.ptr, field_c.as_ptr(), &mut value) };
        if found {
            Some(value)
        } else {
            None
        }
    }

    pub fn get_float(&self, field: &str) -> Option<f32> {
        let field_c = CString::new(field).unwrap();
        let mut value: f32 = 0.0;
        let found = unsafe { ffi::zvec_doc_get_float(self.ptr, field_c.as_ptr(), &mut value) };
        if found {
            Some(value)
        } else {
            None
        }
    }

    pub fn get_string(&self, field: &str) -> Option<&str> {
        let field_c = CString::new(field).unwrap();
        let mut value: *const std::os::raw::c_char = ptr::null();
        let found = unsafe { ffi::zvec_doc_get_string(self.ptr, field_c.as_ptr(), &mut value) };
        if found && !value.is_null() {
            unsafe { std::ffi::CStr::from_ptr(value).to_str().ok() }
        } else {
            None
        }
    }

    pub fn get_vector(&self, field: &str) -> Option<Vec<f32>> {
        let field_c = CString::new(field).unwrap();
        let mut buf = vec![0.0f32; 4096];
        let actual_len = unsafe {
            ffi::zvec_doc_get_vector_fp32(self.ptr, field_c.as_ptr(), buf.as_mut_ptr(), buf.len())
        };
        if actual_len == 0 {
            return None;
        }
        if actual_len > buf.len() {
            buf.resize(actual_len, 0.0);
            unsafe {
                ffi::zvec_doc_get_vector_fp32(
                    self.ptr,
                    field_c.as_ptr(),
                    buf.as_mut_ptr(),
                    buf.len(),
                )
            };
        } else {
            buf.truncate(actual_len);
        }
        Some(buf)
    }

    pub fn has(&self, field: &str) -> bool {
        let field_c = CString::new(field).unwrap();
        unsafe { ffi::zvec_doc_has(self.ptr, field_c.as_ptr()) }
    }

    pub fn has_value(&self, field: &str) -> bool {
        let field_c = CString::new(field).unwrap();
        unsafe { ffi::zvec_doc_has_value(self.ptr, field_c.as_ptr()) }
    }

    pub fn is_null(&self, field: &str) -> bool {
        let field_c = CString::new(field).unwrap();
        unsafe { ffi::zvec_doc_is_null(self.ptr, field_c.as_ptr()) }
    }
}

impl Default for Doc {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Doc {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_doc_free(self.ptr) };
        }
    }
}

pub struct DocList {
    pub(crate) inner: ffi::zvec_doc_list_t,
}

impl DocList {
    pub fn iter(&self) -> DocListIter<'_> {
        DocListIter {
            docs: self,
            index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.count
    }

    pub fn is_empty(&self) -> bool {
        self.inner.count == 0
    }

    pub fn get(&self, index: usize) -> Option<DocRef<'_>> {
        if index < self.inner.count {
            Some(DocRef {
                ptr: unsafe { *self.inner.docs.add(index) },
                _marker: std::marker::PhantomData,
            })
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a DocList {
    type Item = DocRef<'a>;
    type IntoIter = DocListIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Drop for DocList {
    fn drop(&mut self) {
        unsafe { ffi::zvec_doc_list_free(&mut self.inner) };
    }
}

pub struct DocListIter<'a> {
    docs: &'a DocList,
    index: usize,
}

impl<'a> Iterator for DocListIter<'a> {
    type Item = DocRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.docs.len() {
            let doc = self.docs.get(self.index).unwrap();
            self.index += 1;
            Some(doc)
        } else {
            None
        }
    }
}

pub struct DocRef<'a> {
    ptr: *mut ffi::zvec_doc_t,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> DocRef<'a> {
    pub fn pk(&self) -> &str {
        unsafe {
            let ptr = ffi::zvec_doc_pk(self.ptr);
            if ptr.is_null() {
                ""
            } else {
                std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
            }
        }
    }

    pub fn score(&self) -> f32 {
        unsafe { ffi::zvec_doc_score(self.ptr) }
    }

    pub fn doc_id(&self) -> u64 {
        unsafe { ffi::zvec_doc_doc_id(self.ptr) }
    }

    pub fn get_string(&self, field: &str) -> Option<&str> {
        let field_c = CString::new(field).unwrap();
        let mut value: *const std::os::raw::c_char = ptr::null();
        let found = unsafe { ffi::zvec_doc_get_string(self.ptr, field_c.as_ptr(), &mut value) };
        if found && !value.is_null() {
            unsafe { std::ffi::CStr::from_ptr(value).to_str().ok() }
        } else {
            None
        }
    }

    pub fn get_float(&self, field: &str) -> Option<f32> {
        let field_c = CString::new(field).unwrap();
        let mut value: f32 = 0.0;
        let found = unsafe { ffi::zvec_doc_get_float(self.ptr, field_c.as_ptr(), &mut value) };
        if found {
            Some(value)
        } else {
            None
        }
    }

    pub fn get_int64(&self, field: &str) -> Option<i64> {
        let field_c = CString::new(field).unwrap();
        let mut value: i64 = 0;
        let found = unsafe { ffi::zvec_doc_get_int64(self.ptr, field_c.as_ptr(), &mut value) };
        if found {
            Some(value)
        } else {
            None
        }
    }

    pub fn get_bool(&self, field: &str) -> Option<bool> {
        let field_c = CString::new(field).unwrap();
        let mut value: bool = false;
        let found = unsafe { ffi::zvec_doc_get_bool(self.ptr, field_c.as_ptr(), &mut value) };
        if found {
            Some(value)
        } else {
            None
        }
    }

    pub fn get_int32(&self, field: &str) -> Option<i32> {
        let field_c = CString::new(field).unwrap();
        let mut value: i32 = 0;
        let found = unsafe { ffi::zvec_doc_get_int32(self.ptr, field_c.as_ptr(), &mut value) };
        if found {
            Some(value)
        } else {
            None
        }
    }

    pub fn get_double(&self, field: &str) -> Option<f64> {
        let field_c = CString::new(field).unwrap();
        let mut value: f64 = 0.0;
        let found = unsafe { ffi::zvec_doc_get_double(self.ptr, field_c.as_ptr(), &mut value) };
        if found {
            Some(value)
        } else {
            None
        }
    }

    pub fn get_vector(&self, field: &str) -> Option<Vec<f32>> {
        let field_c = CString::new(field).unwrap();
        let mut buf = vec![0.0f32; 4096];
        let actual_len = unsafe {
            ffi::zvec_doc_get_vector_fp32(self.ptr, field_c.as_ptr(), buf.as_mut_ptr(), buf.len())
        };
        if actual_len == 0 {
            return None;
        }
        if actual_len > buf.len() {
            buf.resize(actual_len, 0.0);
            unsafe {
                ffi::zvec_doc_get_vector_fp32(
                    self.ptr,
                    field_c.as_ptr(),
                    buf.as_mut_ptr(),
                    buf.len(),
                )
            };
        } else {
            buf.truncate(actual_len);
        }
        Some(buf)
    }
}

pub struct WriteResults {
    pub(crate) inner: ffi::zvec_write_results_t,
}

impl WriteResults {
    pub fn len(&self) -> usize {
        self.inner.count
    }

    pub fn is_empty(&self) -> bool {
        self.inner.count == 0
    }

    pub fn get(&self, index: usize) -> Option<crate::error::Result<()>> {
        if index < self.inner.count {
            let status = unsafe { *self.inner.statuses.add(index) };
            Some(check_status(status))
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = crate::error::Result<()>> + '_ {
        (0..self.inner.count).filter_map(move |i| self.get(i))
    }
}

impl Drop for WriteResults {
    fn drop(&mut self) {
        unsafe { ffi::zvec_write_results_free(&mut self.inner) };
    }
}

pub struct DocMap {
    pub(crate) inner: ffi::zvec_doc_map_t,
}

impl DocMap {
    pub fn get(&self, pk: &str) -> Option<DocRef<'_>> {
        for i in 0..self.inner.count {
            unsafe {
                let key = std::ffi::CStr::from_ptr(*self.inner.keys.add(i))
                    .to_str()
                    .unwrap_or("");
                if key == pk {
                    return Some(DocRef {
                        ptr: *self.inner.docs.add(i),
                        _marker: std::marker::PhantomData,
                    });
                }
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        self.inner.count
    }

    pub fn is_empty(&self) -> bool {
        self.inner.count == 0
    }

    pub fn keys(&self) -> Vec<&str> {
        let mut keys = Vec::with_capacity(self.inner.count);
        for i in 0..self.inner.count {
            unsafe {
                let key = std::ffi::CStr::from_ptr(*self.inner.keys.add(i))
                    .to_str()
                    .unwrap_or("");
                keys.push(key);
            }
        }
        keys
    }
}

impl Drop for DocMap {
    fn drop(&mut self) {
        unsafe { ffi::zvec_doc_map_free(&mut self.inner) };
    }
}

// SAFETY: These types own their FFI data and don't share mutable state.
// They can be safely sent between threads.
unsafe impl Send for DocList {}
unsafe impl Send for DocMap {}
unsafe impl Send for WriteResults {}
