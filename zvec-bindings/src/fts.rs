use std::ffi::CString;
use std::os::raw::c_int;

use crate::error::{check_error, Result};
use crate::ffi;

/// Full-Text Search payload attached to a vector query for hybrid search.
///
/// Construct with [`Fts::new`], set either a `query_string` (advanced
/// boolean expression) or a `match_string` (natural-language terms), then
/// attach to a [`VectorQuery`](crate::VectorQuery) via
/// [`VectorQuery::fts`](crate::VectorQuery::fts).
///
/// The collection must have an FTS index on the target field for the
/// payload to take effect.
///
/// # Example
///
/// ```rust,no_run
/// use zvec_bindings::Fts;
/// use zvec_bindings::VectorQuery;
///
/// # fn main() -> zvec_bindings::Result<()> {
/// let mut fts = Fts::new()?;
/// fts.set_match_string("hello world")?;
/// let _q = VectorQuery::new("content")
///     .topk(5)
///     .fts(fts)?;
/// # Ok(())
/// # }
/// ```
pub struct Fts {
    ptr: *mut ffi::zvec_fts_t,
}

impl Fts {
    /// Create a new empty FTS payload.
    ///
    /// Returns an error if the underlying allocation fails.
    pub fn new() -> Result<Self> {
        let ptr = unsafe { ffi::zvec_fts_create() };
        if ptr.is_null() {
            return Err(crate::error::Error::InternalError(
                "zvec_fts_create returned null".into(),
            ));
        }
        Ok(Self { ptr })
    }

    /// Set the FTS advanced query expression (boolean / field-aware syntax).
    ///
    /// Mutually exclusive with [`set_match_string`](Self::set_match_string);
    /// pick one.
    pub fn set_query_string(&mut self, s: &str) -> Result<&mut Self> {
        let cstr =
            CString::new(s).map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))?;
        let code = unsafe { ffi::zvec_fts_set_query_string(self.ptr, cstr.as_ptr()) };
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Set the FTS natural-language match string (tokenized and OR/AND-ed
    /// per the index default operator).
    ///
    /// Mutually exclusive with [`set_query_string`](Self::set_query_string).
    pub fn set_match_string(&mut self, s: &str) -> Result<&mut Self> {
        let cstr =
            CString::new(s).map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))?;
        let code = unsafe { ffi::zvec_fts_set_match_string(self.ptr, cstr.as_ptr()) };
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Raw underlying FFI pointer. Used by
    /// [`VectorQuery::fts`](crate::VectorQuery::fts) to attach the payload.
    pub(crate) fn as_ptr(&self) -> *mut ffi::zvec_fts_t {
        self.ptr
    }
}

impl Default for Fts {
    fn default() -> Self {
        Self::new().expect("zvec_fts_create failed in Default")
    }
}

impl Drop for Fts {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_fts_destroy(self.ptr) };
        }
    }
}
