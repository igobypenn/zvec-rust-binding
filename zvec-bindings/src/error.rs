#![allow(non_upper_case_globals)]

use std::os::raw::c_int;
use thiserror::Error;

/// Status codes returned by zvec C API operations.
///
/// These match the upstream `zvec_error_code_t` values from `c_api.h`
/// (zvec v0.5.0): 0=OK, 1=NotFound, 2=AlreadyExists, 3=InvalidArgument,
/// 4=PermissionDenied, 5=FailedPrecondition, 6=ResourceExhausted,
/// 7=Unavailable, 8=InternalError, 9=NotSupported, 10=Unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum StatusCode {
    Ok = 0,
    NotFound = 1,
    AlreadyExists = 2,
    InvalidArgument = 3,
    PermissionDenied = 4,
    FailedPrecondition = 5,
    ResourceExhausted = 6,
    Unavailable = 7,
    InternalError = 8,
    NotSupported = 9,
    Unknown = 10,
}

impl From<i32> for StatusCode {
    fn from(code: i32) -> Self {
        match code {
            0 => StatusCode::Ok,
            1 => StatusCode::NotFound,
            2 => StatusCode::AlreadyExists,
            3 => StatusCode::InvalidArgument,
            4 => StatusCode::PermissionDenied,
            5 => StatusCode::FailedPrecondition,
            6 => StatusCode::ResourceExhausted,
            7 => StatusCode::Unavailable,
            8 => StatusCode::InternalError,
            9 => StatusCode::NotSupported,
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

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Failed precondition: {0}")]
    FailedPrecondition(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Unavailable: {0}")]
    Unavailable(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Not supported: {0}")]
    NotSupported(String),

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

/// Check a `zvec_error_code_t` returned by an upstream C API call and turn
/// any non-zero code into an `Error` (with the thread-local error message
/// attached via `zvec_get_last_error`).
///
/// Replaces the old `check_status(zvec_status_t)` helper. Unlike the old
/// helper, this version also calls `zvec_free` on the message buffer
/// returned by `zvec_get_last_error`, fixing a small message leak that
/// existed in the v0.2.1 custom-wrapper flow.
pub fn check_error(code: c_int) -> Result<()> {
    if code == crate::ffi::zvec_error_code_t_ZVEC_OK as c_int {
        return Ok(());
    }

    let message = last_error_message();

    let error = match code {
        c if c == crate::ffi::zvec_error_code_t_ZVEC_ERROR_NOT_FOUND as c_int => {
            Error::NotFound(message)
        }
        c if c == crate::ffi::zvec_error_code_t_ZVEC_ERROR_ALREADY_EXISTS as c_int => {
            Error::AlreadyExists(message)
        }
        c if c == crate::ffi::zvec_error_code_t_ZVEC_ERROR_INVALID_ARGUMENT as c_int => {
            Error::InvalidArgument(message)
        }
        c if c == crate::ffi::zvec_error_code_t_ZVEC_ERROR_PERMISSION_DENIED as c_int => {
            Error::PermissionDenied(message)
        }
        c if c == crate::ffi::zvec_error_code_t_ZVEC_ERROR_FAILED_PRECONDITION as c_int => {
            Error::FailedPrecondition(message)
        }
        c if c == crate::ffi::zvec_error_code_t_ZVEC_ERROR_RESOURCE_EXHAUSTED as c_int => {
            Error::ResourceExhausted(message)
        }
        c if c == crate::ffi::zvec_error_code_t_ZVEC_ERROR_UNAVAILABLE as c_int => {
            Error::Unavailable(message)
        }
        c if c == crate::ffi::zvec_error_code_t_ZVEC_ERROR_INTERNAL_ERROR as c_int => {
            Error::InternalError(message)
        }
        c if c == crate::ffi::zvec_error_code_t_ZVEC_ERROR_NOT_SUPPORTED as c_int => {
            Error::NotSupported(message)
        }
        _ => Error::Unknown(message),
    };

    Err(error)
}

/// Retrieve the thread-local last error message and free the C-allocated
/// buffer. Returns an empty string if there is no message.
pub(crate) fn last_error_message() -> String {
    unsafe {
        let mut buf: *mut std::os::raw::c_char = std::ptr::null_mut();
        crate::ffi::zvec_get_last_error(&mut buf);
        if buf.is_null() {
            return String::new();
        }
        let s = std::ffi::CStr::from_ptr(buf).to_string_lossy().into_owned();
        crate::ffi::zvec_free(buf as *mut _);
        s
    }
}
