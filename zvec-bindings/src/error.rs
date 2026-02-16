#![allow(non_upper_case_globals)]

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum StatusCode {
    Ok = 0,
    NotFound = 1,
    AlreadyExists = 2,
    InvalidArgument = 3,
    NotSupported = 4,
    InternalError = 5,
    PermissionDenied = 6,
    FailedPrecondition = 7,
    Unknown = 8,
}

impl From<u32> for StatusCode {
    fn from(code: u32) -> Self {
        match code {
            0 => StatusCode::Ok,
            1 => StatusCode::NotFound,
            2 => StatusCode::AlreadyExists,
            3 => StatusCode::InvalidArgument,
            4 => StatusCode::NotSupported,
            5 => StatusCode::InternalError,
            6 => StatusCode::PermissionDenied,
            7 => StatusCode::FailedPrecondition,
            _ => StatusCode::Unknown,
        }
    }
}

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Not supported: {0}")]
    NotSupported(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Failed precondition: {0}")]
    FailedPrecondition(String),

    #[error("Unknown error: {0}")]
    Unknown(String),

    #[error("Null pointer")]
    NullPointer,

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Field not found: {0}")]
    FieldNotFound(String),

    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("UTF-8 error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::ffi::NulError),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn check_status(status: crate::ffi::zvec_status_t) -> Result<()> {
    use crate::ffi::*;

    if status.code == zvec_status_code_ZVEC_STATUS_OK {
        return Ok(());
    }

    let message = if status.message.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(status.message) }
            .to_string_lossy()
            .into_owned()
    };

    let error = match status.code {
        zvec_status_code_ZVEC_STATUS_NOT_FOUND => Error::NotFound(message),
        zvec_status_code_ZVEC_STATUS_ALREADY_EXISTS => Error::AlreadyExists(message),
        zvec_status_code_ZVEC_STATUS_INVALID_ARGUMENT => Error::InvalidArgument(message),
        zvec_status_code_ZVEC_STATUS_NOT_SUPPORTED => Error::NotSupported(message),
        zvec_status_code_ZVEC_STATUS_INTERNAL_ERROR => Error::InternalError(message),
        zvec_status_code_ZVEC_STATUS_PERMISSION_DENIED => Error::PermissionDenied(message),
        zvec_status_code_ZVEC_STATUS_FAILED_PRECONDITION => Error::FailedPrecondition(message),
        _ => Error::Unknown(message),
    };

    Err(error)
}
