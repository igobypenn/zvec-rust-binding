use std::ffi::CString;

use crate::error::{check_status, Result};
use crate::ffi;
use crate::types::DataType;

pub struct FieldSchema {
    pub(crate) ptr: *mut ffi::zvec_field_schema_t,
    owned: bool,
}

impl FieldSchema {
    pub fn new(name: &str, data_type: DataType) -> Self {
        let name_c = CString::new(name).unwrap();
        let ptr = unsafe { ffi::zvec_field_schema_new(name_c.as_ptr(), data_type.into()) };
        Self { ptr, owned: true }
    }

    pub fn new_vector(name: &str, data_type: DataType, dimension: u32) -> Self {
        let name_c = CString::new(name).unwrap();
        let ptr = unsafe {
            ffi::zvec_field_schema_new_with_dimension(name_c.as_ptr(), data_type.into(), dimension)
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

    pub fn set_nullable(&mut self, nullable: bool) {
        unsafe { ffi::zvec_field_schema_set_nullable(self.ptr, nullable) };
    }

    pub fn name(&self) -> &str {
        unsafe {
            let ptr = ffi::zvec_field_schema_name(self.ptr);
            if ptr.is_null() {
                ""
            } else {
                std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
            }
        }
    }

    pub fn data_type(&self) -> DataType {
        unsafe { ffi::zvec_field_schema_data_type(self.ptr).into() }
    }

    pub fn nullable(&self) -> bool {
        unsafe { ffi::zvec_field_schema_nullable(self.ptr) }
    }

    pub fn dimension(&self) -> u32 {
        unsafe { ffi::zvec_field_schema_dimension(self.ptr) }
    }
}

impl Drop for FieldSchema {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe { ffi::zvec_field_schema_free(self.ptr) };
        }
    }
}

pub struct CollectionSchema {
    pub(crate) ptr: *mut ffi::zvec_collection_schema_t,
    owned: bool,
}

impl CollectionSchema {
    pub fn new(name: &str) -> Self {
        let name_c = CString::new(name).unwrap();
        let ptr = unsafe { ffi::zvec_collection_schema_new(name_c.as_ptr()) };
        Self { ptr, owned: true }
    }

    pub(crate) fn from_ptr(ptr: *mut ffi::zvec_collection_schema_t) -> Self {
        Self { ptr, owned: false }
    }

    pub fn add_field(&mut self, field: FieldSchema) -> Result<()> {
        let status = unsafe { ffi::zvec_collection_schema_add_field(self.ptr, field.ptr) };
        check_status(status)
    }

    pub fn name(&self) -> &str {
        unsafe {
            let ptr = ffi::zvec_collection_schema_name(self.ptr);
            if ptr.is_null() {
                ""
            } else {
                std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
            }
        }
    }

    pub fn field_names(&self) -> Vec<String> {
        let arr = unsafe { ffi::zvec_collection_schema_field_names(self.ptr) };
        let mut names = Vec::with_capacity(arr.count);
        for i in 0..arr.count {
            unsafe {
                if !arr.strings.add(i).is_null() {
                    let s = std::ffi::CStr::from_ptr(*arr.strings.add(i));
                    if let Ok(name) = s.to_str() {
                        names.push(name.to_string());
                    }
                }
            }
        }
        let mut arr_mut = arr;
        unsafe { ffi::zvec_string_array_free(&mut arr_mut) };
        names
    }

    pub fn vector_field_names(&self) -> Vec<String> {
        let arr = unsafe { ffi::zvec_collection_schema_vector_field_names(self.ptr) };
        let mut names = Vec::with_capacity(arr.count);
        for i in 0..arr.count {
            unsafe {
                if !arr.strings.add(i).is_null() {
                    let s = std::ffi::CStr::from_ptr(*arr.strings.add(i));
                    if let Ok(name) = s.to_str() {
                        names.push(name.to_string());
                    }
                }
            }
        }
        let mut arr_mut = arr;
        unsafe { ffi::zvec_string_array_free(&mut arr_mut) };
        names
    }
}

impl Drop for CollectionSchema {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe { ffi::zvec_collection_schema_free(self.ptr) };
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
