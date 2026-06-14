use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;

use crate::error::{check_error, Result};
use crate::ffi;
use crate::types::DataType;

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
        let ptr = unsafe { ffi::zvec_doc_create() };
        Self { ptr }
    }

    /// Create a new document with the given primary key.
    ///
    /// # Panics
    ///
    /// Panics if the primary key contains interior NUL bytes.
    /// For a fallible version, use [`with_pk_mut`].
    pub fn with_pk(pk: impl Into<String>) -> Self {
        let mut doc = Self::new();
        doc.set_pk(pk).expect("primary key contains NUL byte");
        doc
    }

    /// Create a new document with the given ID (alias for `with_pk`).
    ///
    /// This method is provided for convenience but `with_pk` is preferred
    /// for consistency with the `pk()` getter and `set_pk()` setter.
    #[inline]
    pub fn id(id: impl Into<String>) -> Self {
        Self::with_pk(id)
    }

    /// Set the primary key and return self for chaining.
    pub fn with_pk_mut(mut self, pk: impl Into<String>) -> Result<Self> {
        self.set_pk(pk)?;
        Ok(self)
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
    ///
    /// Returns an error if the primary key contains interior NUL bytes.
    pub fn set_pk(&mut self, pk: impl Into<String>) -> Result<()> {
        let pk_c = CString::new(pk.into())
            .map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))?;
        unsafe { ffi::zvec_doc_set_pk(self.ptr, pk_c.as_ptr()) };
        Ok(())
    }

    /// Get the primary key.
    pub fn pk(&self) -> &str {
        unsafe { pk_from_ptr(ffi::zvec_doc_get_pk_pointer(self.ptr)) }
    }

    pub fn score(&self) -> f32 {
        unsafe { ffi::zvec_doc_get_score(self.ptr) }
    }

    pub fn doc_id(&self) -> u64 {
        unsafe { ffi::zvec_doc_get_doc_id(self.ptr) }
    }

    pub fn set_bool(&mut self, field: &str, value: bool) -> Result<()> {
        set_field_by_value(self.ptr, field, DataType::Bool, &value)
    }

    pub fn set_int32(&mut self, field: &str, value: i32) -> Result<()> {
        set_field_by_value(self.ptr, field, DataType::Int32, &value)
    }

    pub fn set_int64(&mut self, field: &str, value: i64) -> Result<()> {
        set_field_by_value(self.ptr, field, DataType::Int64, &value)
    }

    pub fn set_uint32(&mut self, field: &str, value: u32) -> Result<()> {
        set_field_by_value(self.ptr, field, DataType::UInt32, &value)
    }

    pub fn set_uint64(&mut self, field: &str, value: u64) -> Result<()> {
        set_field_by_value(self.ptr, field, DataType::UInt64, &value)
    }

    pub fn set_float(&mut self, field: &str, value: f32) -> Result<()> {
        set_field_by_value(self.ptr, field, DataType::Float, &value)
    }

    pub fn set_double(&mut self, field: &str, value: f64) -> Result<()> {
        set_field_by_value(self.ptr, field, DataType::Double, &value)
    }

    pub fn set_string(&mut self, field: &str, value: &str) -> Result<()> {
        let value_c = CString::new(value)
            .map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))?;
        set_field_by_value_raw(
            self.ptr,
            field,
            DataType::String,
            value_c.as_ptr() as *const std::os::raw::c_void,
            value.len(),
        )
    }

    pub fn set_vector(&mut self, field: &str, vector: &[f32]) -> Result<()> {
        set_field_by_value_raw(
            self.ptr,
            field,
            DataType::VectorFp32,
            vector.as_ptr() as *const std::os::raw::c_void,
            vector.len() * std::mem::size_of::<f32>(),
        )
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
        // Upstream `zvec_doc_add_field_by_value` expects sparse vectors packed
        // as: [nnz: u32][indices: u32 * nnz][values: f32 * nnz].
        let nnz = indices.len();
        let mut buf: Vec<u8> = Vec::with_capacity(
            std::mem::size_of::<u32>() + nnz * std::mem::size_of::<u32>() + nnz * std::mem::size_of::<f32>(),
        );
        buf.extend_from_slice(&(nnz as u32).to_ne_bytes());
        for &idx in indices {
            buf.extend_from_slice(&idx.to_ne_bytes());
        }
        for &val in values {
            buf.extend_from_slice(&val.to_ne_bytes());
        }
        set_field_by_value_raw(
            self.ptr,
            field,
            DataType::SparseVectorFp32,
            buf.as_ptr() as *const std::os::raw::c_void,
            buf.len(),
        )
    }

    pub fn get_bool(&self, field: &str) -> Option<bool> {
        get_basic_value(self.ptr, field, DataType::Bool)
    }

    pub fn get_int32(&self, field: &str) -> Option<i32> {
        get_basic_value(self.ptr, field, DataType::Int32)
    }

    pub fn get_int64(&self, field: &str) -> Option<i64> {
        get_basic_value(self.ptr, field, DataType::Int64)
    }

    pub fn get_uint32(&self, field: &str) -> Option<u32> {
        get_basic_value(self.ptr, field, DataType::UInt32)
    }

    pub fn get_uint64(&self, field: &str) -> Option<u64> {
        get_basic_value(self.ptr, field, DataType::UInt64)
    }

    pub fn get_float(&self, field: &str) -> Option<f32> {
        get_basic_value(self.ptr, field, DataType::Float)
    }

    pub fn get_double(&self, field: &str) -> Option<f64> {
        get_basic_value(self.ptr, field, DataType::Double)
    }

    pub fn get_string(&self, field: &str) -> Option<&str> {
        get_string_value(self.ptr, field)
    }

    pub fn get_vector(&self, field: &str) -> Option<Vec<f32>> {
        get_vector_value(self.ptr, field)
    }

    pub fn has(&self, field: &str) -> bool {
        let field_c = CString::new(field).expect("field name contains NUL byte");
        unsafe { ffi::zvec_doc_has_field(self.ptr, field_c.as_ptr()) }
    }

    pub fn has_value(&self, field: &str) -> bool {
        let field_c = CString::new(field).expect("field name contains NUL byte");
        unsafe { ffi::zvec_doc_has_field_value(self.ptr, field_c.as_ptr()) }
    }

    pub fn is_null(&self, field: &str) -> bool {
        let field_c = CString::new(field).expect("field name contains NUL byte");
        unsafe { ffi::zvec_doc_is_field_null(self.ptr, field_c.as_ptr()) }
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
            unsafe { ffi::zvec_doc_destroy(self.ptr) };
        }
    }
}

// ===== Free helper functions =====

fn pk_from_ptr(ptr: *const std::os::raw::c_char) -> &'static str {
    unsafe {
        if ptr.is_null() {
            ""
        } else {
            std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
        }
    }
}

/// Add a field by value (typed convenience for fixed-size scalars).
fn set_field_by_value<T>(
    doc: *mut ffi::zvec_doc_t,
    field: &str,
    data_type: DataType,
    value: &T,
) -> Result<()> {
    set_field_by_value_raw(
        doc,
        field,
        data_type,
        value as *const T as *const std::os::raw::c_void,
        std::mem::size_of::<T>(),
    )
}

/// Add a field by value with an explicit byte size.
fn set_field_by_value_raw(
    doc: *mut ffi::zvec_doc_t,
    field: &str,
    data_type: DataType,
    value: *const std::os::raw::c_void,
    value_size: usize,
) -> Result<()> {
    let field_c = CString::new(field).map_err(|e| {
        crate::error::Error::InvalidArgument(format!("field name: {}", e))
    })?;
    let code = unsafe {
        ffi::zvec_doc_add_field_by_value(
            doc,
            field_c.as_ptr(),
            data_type.into(),
            value,
            value_size,
        )
    };
    check_error(code as c_int)
}

/// Read a basic (fixed-size numeric) field via `zvec_doc_get_field_value_basic`.
/// Returns `None` if the field is missing or the type does not match.
fn get_basic_value<T: Default + Copy>(
    doc: *const ffi::zvec_doc_t,
    field: &str,
    data_type: DataType,
) -> Option<T> {
    let field_c = CString::new(field).expect("field name contains NUL byte");
    let mut value: T = T::default();
    let code = unsafe {
        ffi::zvec_doc_get_field_value_basic(
            doc,
            field_c.as_ptr(),
            data_type.into(),
            &mut value as *mut T as *mut std::os::raw::c_void,
            std::mem::size_of::<T>(),
        )
    };
    if code == ffi::zvec_error_code_t_ZVEC_OK {
        Some(value)
    } else {
        None
    }
}

/// Read a string field via `zvec_doc_get_field_value_pointer` (zero-copy view
/// into document memory). Returns `None` if the field is missing.
///
/// The returned `&str` lifetime is detached from `field`; the caller is
/// responsible for ensuring the document outlives the borrow.
fn get_string_value<'a>(doc: *const ffi::zvec_doc_t, field: &str) -> Option<&'a str> {
    let field_c = CString::new(field).expect("field name contains NUL byte");
    let mut value: *const std::os::raw::c_void = ptr::null();
    let mut size: usize = 0;
    let code = unsafe {
        ffi::zvec_doc_get_field_value_pointer(
            doc,
            field_c.as_ptr(),
            DataType::String.into(),
            &mut value,
            &mut size,
        )
    };
    if code == ffi::zvec_error_code_t_ZVEC_OK && !value.is_null() {
        unsafe {
            std::ffi::CStr::from_ptr(value as *const std::os::raw::c_char)
                .to_str()
                .ok()
        }
    } else {
        None
    }
}

/// Read a dense FP32 vector field via `zvec_doc_get_field_value_pointer`
/// (zero-copy view into document memory, copied into an owned `Vec<f32>`).
fn get_vector_value(doc: *const ffi::zvec_doc_t, field: &str) -> Option<Vec<f32>> {
    let field_c = CString::new(field).expect("field name contains NUL byte");
    let mut value: *const std::os::raw::c_void = ptr::null();
    let mut size: usize = 0;
    let code = unsafe {
        ffi::zvec_doc_get_field_value_pointer(
            doc,
            field_c.as_ptr(),
            DataType::VectorFp32.into(),
            &mut value,
            &mut size,
        )
    };
    if code == ffi::zvec_error_code_t_ZVEC_OK && !value.is_null() && size > 0 {
        let len = size / std::mem::size_of::<f32>();
        let slice = unsafe { std::slice::from_raw_parts(value as *const f32, len) };
        Some(slice.to_vec())
    } else {
        None
    }
}

// ===== DocList =====

/// Owning wrapper around an array of document pointers returned by query/fetch.
///
/// Holds `*mut *mut zvec_doc_t` + count. When `owned` is true (the default,
/// produced by `from_raw`), the array and all its documents are freed on drop
/// via `zvec_docs_free`. When `owned` is false (produced by `borrow_raw`),
/// drop is a no-op — the caller guarantees the underlying storage outlives
/// this `DocList`.
pub struct DocList {
    pub(crate) docs: *mut *mut ffi::zvec_doc_t,
    pub(crate) count: usize,
    pub(crate) owned: bool,
}

impl DocList {
    /// Construct a `DocList` from raw FFI outputs. Takes ownership of the
    /// array; it will be freed on drop.
    pub(crate) fn from_raw(docs: *mut *mut ffi::zvec_doc_t, count: usize) -> Self {
        Self {
            docs,
            count,
            owned: true,
        }
    }

    /// Construct a non-owning `DocList` that borrows an array of document
    /// pointers. Drop is a no-op; the caller must keep the source alive.
    pub(crate) fn borrow_raw(docs: *mut *mut ffi::zvec_doc_t, count: usize) -> Self {
        Self {
            docs,
            count,
            owned: false,
        }
    }

    pub fn iter(&self) -> DocListIter<'_> {
        DocListIter {
            docs: self,
            index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get(&self, index: usize) -> Option<DocRef<'_>> {
        if index < self.count {
            Some(DocRef {
                ptr: unsafe { *self.docs.add(index) },
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
        if self.owned && !self.docs.is_null() {
            unsafe { ffi::zvec_docs_free(self.docs, self.count) };
        }
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

/// A borrowed view into a document owned by a [`DocList`] or [`DocMap`].
pub struct DocRef<'a> {
    pub(crate) ptr: *mut ffi::zvec_doc_t,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> DocRef<'a> {
    pub fn pk(&self) -> &str {
        unsafe { pk_from_ptr(ffi::zvec_doc_get_pk_pointer(self.ptr)) }
    }

    pub fn score(&self) -> f32 {
        unsafe { ffi::zvec_doc_get_score(self.ptr) }
    }

    pub fn doc_id(&self) -> u64 {
        unsafe { ffi::zvec_doc_get_doc_id(self.ptr) }
    }

    pub fn get_bool(&self, field: &str) -> Option<bool> {
        get_basic_value(self.ptr, field, DataType::Bool)
    }

    pub fn get_int32(&self, field: &str) -> Option<i32> {
        get_basic_value(self.ptr, field, DataType::Int32)
    }

    pub fn get_int64(&self, field: &str) -> Option<i64> {
        get_basic_value(self.ptr, field, DataType::Int64)
    }

    pub fn get_uint32(&self, field: &str) -> Option<u32> {
        get_basic_value(self.ptr, field, DataType::UInt32)
    }

    pub fn get_uint64(&self, field: &str) -> Option<u64> {
        get_basic_value(self.ptr, field, DataType::UInt64)
    }

    pub fn get_float(&self, field: &str) -> Option<f32> {
        get_basic_value(self.ptr, field, DataType::Float)
    }

    pub fn get_double(&self, field: &str) -> Option<f64> {
        get_basic_value(self.ptr, field, DataType::Double)
    }

    pub fn get_string(&self, field: &str) -> Option<&str> {
        get_string_value(self.ptr, field)
    }

    pub fn get_vector(&self, field: &str) -> Option<Vec<f32>> {
        get_vector_value(self.ptr, field)
    }

    pub fn has(&self, field: &str) -> bool {
        let field_c = CString::new(field).expect("field name contains NUL byte");
        unsafe { ffi::zvec_doc_has_field(self.ptr, field_c.as_ptr()) }
    }

    pub fn has_value(&self, field: &str) -> bool {
        let field_c = CString::new(field).expect("field name contains NUL byte");
        unsafe { ffi::zvec_doc_has_field_value(self.ptr, field_c.as_ptr()) }
    }

    pub fn is_null(&self, field: &str) -> bool {
        let field_c = CString::new(field).expect("field name contains NUL byte");
        unsafe { ffi::zvec_doc_is_field_null(self.ptr, field_c.as_ptr()) }
    }
}

// ===== WriteResults =====

/// Per-document results returned by insert/upsert/update/delete operations.
///
/// Wraps a `zvec_write_result_t` array freed via `zvec_write_results_free`.
pub struct WriteResults {
    pub(crate) results: *mut ffi::zvec_write_result_t,
    pub(crate) count: usize,
}

impl WriteResults {
    /// Construct from raw FFI outputs. Takes ownership of the array.
    pub(crate) fn from_raw(results: *mut ffi::zvec_write_result_t, count: usize) -> Self {
        Self { results, count }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get(&self, index: usize) -> Option<crate::error::Result<()>> {
        if index < self.count {
            let entry = unsafe { &*self.results.add(index) };
            if entry.code == ffi::zvec_error_code_t_ZVEC_OK {
                Some(Ok(()))
            } else {
                let message = if entry.message.is_null() {
                    String::new()
                } else {
                    unsafe {
                        std::ffi::CStr::from_ptr(entry.message)
                            .to_string_lossy()
                            .into_owned()
                    }
                };
                Some(Err(error_from_code(entry.code as c_int, message)))
            }
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = crate::error::Result<()>> + '_ {
        (0..self.count).filter_map(move |i| self.get(i))
    }
}

impl Drop for WriteResults {
    fn drop(&mut self) {
        if !self.results.is_null() {
            unsafe { ffi::zvec_write_results_free(self.results, self.count) };
        }
    }
}

// ===== DocMap =====

/// Mapping from primary key to document, built client-side from a flat
/// fetch result array.
///
/// Holds the underlying `*mut *mut zvec_doc_t` array (freed on drop via
/// `zvec_docs_free`) plus an index mapping PK strings to array offsets.
pub struct DocMap {
    docs: *mut *mut ffi::zvec_doc_t,
    count: usize,
    index: std::collections::HashMap<String, usize>,
}

impl DocMap {
    /// Construct a `DocMap` from raw FFI outputs, reading each document's PK
    /// to build the lookup index. Takes ownership of the array.
    pub(crate) fn from_raw(docs: *mut *mut ffi::zvec_doc_t, count: usize) -> Self {
        let mut index = std::collections::HashMap::with_capacity(count);
        for i in 0..count {
            unsafe {
                let doc = *docs.add(i);
                if !doc.is_null() {
                    let pk_ptr = ffi::zvec_doc_get_pk_pointer(doc);
                    let pk = if pk_ptr.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(pk_ptr)
                            .to_string_lossy()
                            .into_owned()
                    };
                    index.insert(pk, i);
                }
            }
        }
        Self { docs, count, index }
    }

    pub fn get(&self, pk: &str) -> Option<DocRef<'_>> {
        let i = *self.index.get(pk)?;
        Some(DocRef {
            ptr: unsafe { *self.docs.add(i) },
            _marker: std::marker::PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn keys(&self) -> Vec<&str> {
        self.index.keys().map(|s| s.as_str()).collect()
    }
}

impl Drop for DocMap {
    fn drop(&mut self) {
        if !self.docs.is_null() {
            unsafe { ffi::zvec_docs_free(self.docs, self.count) };
        }
    }
}

// Map a `zvec_error_code_t` + message to an `Error` variant, mirroring
// `error::check_error` but using an explicit message instead of the
// thread-local last-error buffer (per-doc results carry their own message).
fn error_from_code(code: c_int, message: String) -> crate::error::Error {
    use std::os::raw::c_int as C;
    let ok = ffi::zvec_error_code_t_ZVEC_OK as C;
    if code == ok {
        return crate::error::Error::Unknown(String::from("expected error, got OK"));
    }
    match code {
        c if c == ffi::zvec_error_code_t_ZVEC_ERROR_NOT_FOUND as C => {
            crate::error::Error::NotFound(message)
        }
        c if c == ffi::zvec_error_code_t_ZVEC_ERROR_ALREADY_EXISTS as C => {
            crate::error::Error::AlreadyExists(message)
        }
        c if c == ffi::zvec_error_code_t_ZVEC_ERROR_INVALID_ARGUMENT as C => {
            crate::error::Error::InvalidArgument(message)
        }
        c if c == ffi::zvec_error_code_t_ZVEC_ERROR_PERMISSION_DENIED as C => {
            crate::error::Error::PermissionDenied(message)
        }
        c if c == ffi::zvec_error_code_t_ZVEC_ERROR_FAILED_PRECONDITION as C => {
            crate::error::Error::FailedPrecondition(message)
        }
        c if c == ffi::zvec_error_code_t_ZVEC_ERROR_RESOURCE_EXHAUSTED as C => {
            crate::error::Error::ResourceExhausted(message)
        }
        c if c == ffi::zvec_error_code_t_ZVEC_ERROR_UNAVAILABLE as C => {
            crate::error::Error::Unavailable(message)
        }
        c if c == ffi::zvec_error_code_t_ZVEC_ERROR_INTERNAL_ERROR as C => {
            crate::error::Error::InternalError(message)
        }
        c if c == ffi::zvec_error_code_t_ZVEC_ERROR_NOT_SUPPORTED as C => {
            crate::error::Error::NotSupported(message)
        }
        _ => crate::error::Error::Unknown(message),
    }
}

// SAFETY: These types own their FFI data and don't share mutable state.
// They can be safely sent between threads.
unsafe impl Send for DocList {}
unsafe impl Send for DocMap {}
unsafe impl Send for WriteResults {}
