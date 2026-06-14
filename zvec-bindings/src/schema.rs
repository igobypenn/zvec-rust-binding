use std::ffi::CString;
use std::os::raw::c_int;

use crate::error::{check_error, Result};
use crate::ffi;
use crate::types::DataType;

pub struct FieldSchema {
    pub(crate) ptr: *mut ffi::zvec_field_schema_t,
    owned: bool,
}

impl FieldSchema {
    pub fn new(name: &str, data_type: DataType) -> Self {
        let name_c = CString::new(name).expect("name contains NUL byte");
        let ptr =
            unsafe { ffi::zvec_field_schema_create(name_c.as_ptr(), data_type.into(), false, 0) };
        Self { ptr, owned: true }
    }

    pub fn new_vector(name: &str, data_type: DataType, dimension: u32) -> Self {
        let name_c = CString::new(name).expect("name contains NUL byte");
        let ptr = unsafe {
            ffi::zvec_field_schema_create(name_c.as_ptr(), data_type.into(), false, dimension)
        };
        Self { ptr, owned: true }
    }

    pub fn bool_(name: &str) -> Self {
        Self::new(name, DataType::Bool)
    }

    pub fn int32(name: &str) -> Self {
        Self::new(name, DataType::Int32)
    }

    pub fn int64(name: &str) -> Self {
        Self::new(name, DataType::Int64)
    }

    pub fn float(name: &str) -> Self {
        Self::new(name, DataType::Float)
    }

    pub fn double(name: &str) -> Self {
        Self::new(name, DataType::Double)
    }

    pub fn string(name: &str) -> Self {
        Self::new(name, DataType::String)
    }

    pub fn set_nullable(&mut self, nullable: bool) -> Result<()> {
        let code = unsafe { ffi::zvec_field_schema_set_nullable(self.ptr, nullable) };
        check_error(code as c_int)
    }

    /// Attach index parameters (e.g. FTS) to this field schema.
    ///
    /// Required for FTS indexes: the index params must be set on the
    /// field schema BEFORE the collection is created, so that documents
    /// are indexed as they are inserted. Upstream deep-copies the params,
    /// so the caller retains ownership of `params`.
    pub fn set_index_params(&mut self, params: &crate::IndexParams) -> Result<()> {
        let code = unsafe { ffi::zvec_field_schema_set_index_params(self.ptr, params.ptr) };
        check_error(code as c_int)
    }

    pub fn name(&self) -> &str {
        unsafe {
            let ptr = ffi::zvec_field_schema_get_name(self.ptr);
            if ptr.is_null() {
                ""
            } else {
                std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
            }
        }
    }

    pub fn data_type(&self) -> DataType {
        unsafe { ffi::zvec_field_schema_get_data_type(self.ptr).into() }
    }

    /// Returns the index type declared on this field (e.g. FTS for a
    /// string field with FTS params attached). Returns
    /// [`IndexType::Undefined`](crate::IndexType::Undefined) if no index
    /// params have been set.
    pub fn index_type(&self) -> crate::IndexType {
        unsafe { ffi::zvec_field_schema_get_index_type(self.ptr).into() }
    }

    /// Returns true if this field has any index declared.
    pub fn has_index(&self) -> bool {
        unsafe { ffi::zvec_field_schema_has_index(self.ptr) }
    }

    pub fn nullable(&self) -> bool {
        unsafe { ffi::zvec_field_schema_is_nullable(self.ptr) }
    }

    pub fn dimension(&self) -> u32 {
        unsafe { ffi::zvec_field_schema_get_dimension(self.ptr) }
    }
}

impl Drop for FieldSchema {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe { ffi::zvec_field_schema_destroy(self.ptr) };
        }
    }
}

pub struct CollectionSchema {
    pub(crate) ptr: *mut ffi::zvec_collection_schema_t,
    owned: bool,
}

impl CollectionSchema {
    pub fn new(name: &str) -> Self {
        let name_c = CString::new(name).expect("name contains NUL byte");
        let ptr = unsafe { ffi::zvec_collection_schema_create(name_c.as_ptr()) };
        Self { ptr, owned: true }
    }

    /// Wrap a borrowed schema pointer (e.g. returned by `collection.schema()`).
    /// The wrapper does not own the pointer and will not destroy it on drop.
    pub(crate) fn from_ptr(ptr: *mut ffi::zvec_collection_schema_t) -> Self {
        Self { ptr, owned: false }
    }

    /// Add a field to the collection schema.
    ///
    /// The schema clones the field internally; the caller retains ownership
    /// of the passed-in `FieldSchema`.
    ///
    /// Accepts either `FieldSchema` or `VectorSchema` (conversion is automatic).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use zvec_bindings::{CollectionSchema, VectorSchema};
    ///
    /// let mut schema = CollectionSchema::new("my_collection");
    /// // No .into() needed - VectorSchema converts automatically
    /// schema.add_field(VectorSchema::fp32("embedding", 128))?;
    /// # Ok::<(), zvec_bindings::Error>(())
    /// ```
    pub fn add_field(&mut self, field: impl Into<FieldSchema>) -> Result<()> {
        let field = field.into();
        let code = unsafe { ffi::zvec_collection_schema_add_field(self.ptr, field.ptr) };
        check_error(code as c_int)
    }

    pub fn name(&self) -> &str {
        unsafe {
            let ptr = ffi::zvec_collection_schema_get_name(self.ptr);
            if ptr.is_null() {
                ""
            } else {
                std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
            }
        }
    }

    pub fn field_names(&self) -> Vec<String> {
        unsafe {
            let mut names_ptr: *mut *const std::os::raw::c_char = std::ptr::null_mut();
            let mut count: usize = 0;
            let code = ffi::zvec_collection_schema_get_all_field_names(
                self.ptr,
                &mut names_ptr,
                &mut count,
            );
            if code != ffi::zvec_error_code_t_ZVEC_OK || names_ptr.is_null() {
                return Vec::new();
            }
            let mut names = Vec::with_capacity(count);
            for i in 0..count {
                let s = *names_ptr.add(i);
                if !s.is_null() {
                    if let Ok(name) = std::ffi::CStr::from_ptr(s).to_str() {
                        names.push(name.to_string());
                    }
                }
            }
            ffi::zvec_free(names_ptr as *mut _);
            names
        }
    }

    pub fn vector_field_names(&self) -> Vec<String> {
        unsafe {
            let mut fields_ptr: *mut *mut ffi::zvec_field_schema_t = std::ptr::null_mut();
            let mut count: usize = 0;
            let code = ffi::zvec_collection_schema_get_vector_fields(
                self.ptr,
                &mut fields_ptr,
                &mut count,
            );
            if code != ffi::zvec_error_code_t_ZVEC_OK || fields_ptr.is_null() {
                return Vec::new();
            }
            let mut names = Vec::with_capacity(count);
            for i in 0..count {
                let field = *fields_ptr.add(i);
                if !field.is_null() {
                    let name_ptr = ffi::zvec_field_schema_get_name(field);
                    if !name_ptr.is_null() {
                        if let Ok(name) = std::ffi::CStr::from_ptr(name_ptr).to_str() {
                            names.push(name.to_string());
                        }
                    }
                }
            }
            ffi::zvec_free(fields_ptr as *mut _);
            names
        }
    }
}

impl Drop for CollectionSchema {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe { ffi::zvec_collection_schema_destroy(self.ptr) };
        }
    }
}

pub struct VectorSchema {
    name: String,
    data_type: DataType,
    dimension: u32,
}

impl VectorSchema {
    pub fn new(name: impl Into<String>, data_type: DataType, dimension: u32) -> Self {
        Self {
            name: name.into(),
            data_type,
            dimension,
        }
    }

    pub fn fp32(name: impl Into<String>, dimension: u32) -> Self {
        Self::new(name, DataType::VectorFp32, dimension)
    }

    pub fn fp16(name: impl Into<String>, dimension: u32) -> Self {
        Self::new(name, DataType::VectorFp16, dimension)
    }

    pub fn sparse_fp32(name: impl Into<String>) -> Self {
        Self::new(name, DataType::SparseVectorFp32, 0)
    }

    pub fn sparse_fp32_with_dim(name: impl Into<String>, dimension: u32) -> Self {
        Self::new(name, DataType::SparseVectorFp32, dimension)
    }

    pub fn sparse_fp16(name: impl Into<String>) -> Self {
        Self::new(name, DataType::SparseVectorFp16, 0)
    }

    pub fn sparse_fp16_with_dim(name: impl Into<String>, dimension: u32) -> Self {
        Self::new(name, DataType::SparseVectorFp16, dimension)
    }

    pub fn into_field_schema(self) -> FieldSchema {
        FieldSchema::new_vector(&self.name, self.data_type, self.dimension)
    }
}

impl From<VectorSchema> for FieldSchema {
    fn from(schema: VectorSchema) -> Self {
        schema.into_field_schema()
    }
}

// SAFETY: These types own their FFI pointers and don't share state.
// CollectionSchema is typically consumed during collection creation.
unsafe impl Send for CollectionSchema {}
unsafe impl Send for FieldSchema {}
