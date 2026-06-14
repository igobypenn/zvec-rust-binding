use std::ffi::CString;
use std::os::raw::c_int;

use crate::error::{check_error, Result};
use crate::ffi;
use crate::query::{FlatQueryParam, HnswQueryParam, IVFQueryParam};

/// A sub-query within a multi-query. Each sub-query targets a single
/// field (vector or FTS) with its own vector and parameters.
///
/// Built using the consumer-builder pattern (consumes self on each
/// setter, like [`crate::VectorQuery`]).
pub struct SubQuery {
    pub(crate) ptr: *mut ffi::zvec_sub_query_t,
}

impl SubQuery {
    /// Create a new empty sub-query.
    pub fn new() -> Result<Self> {
        let ptr = unsafe { ffi::zvec_sub_query_create() };
        if ptr.is_null() {
            return Err(crate::error::Error::InternalError(
                "zvec_sub_query_create returned null".into(),
            ));
        }
        Ok(Self { ptr })
    }

    /// Set the target field name (vector field or FTS-indexed string
    /// field).
    pub fn field_name(self, name: &str) -> Result<Self> {
        let cstr = CString::new(name)
            .map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))?;
        let code = unsafe { ffi::zvec_sub_query_set_field_name(self.ptr, cstr.as_ptr()) };
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Set the candidate pool size retrieved from this sub-query before
    /// reranking. Larger values improve recall at the cost of latency.
    pub fn num_candidates(self, n: u32) -> Self {
        let code =
            unsafe { ffi::zvec_sub_query_set_num_candidates(self.ptr, n as c_int) };
        let _ = check_error(code as c_int);
        self
    }

    /// Set a dense FP32 query vector. The data is copied into the
    /// sub-query.
    pub fn vector(self, data: &[f32]) -> Result<Self> {
        let code = unsafe {
            ffi::zvec_sub_query_set_query_vector(
                self.ptr,
                data.as_ptr() as *const std::os::raw::c_void,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Set a sparse FP32 query vector (two parallel arrays of indices
    /// and values). The data is copied into the sub-query via two
    /// upstream calls (`zvec_sub_query_set_sparse_indices` then
    /// `zvec_sub_query_set_sparse_values`).
    pub fn sparse_vector(self, indices: &[u32], values: &[f32]) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(crate::error::Error::InvalidArgument(
                "indices and values must have same length".into(),
            ));
        }
        let n = indices.len();
        let code = unsafe {
            ffi::zvec_sub_query_set_sparse_indices(self.ptr, indices.as_ptr(), n)
        };
        check_error(code as c_int)?;
        let code = unsafe {
            ffi::zvec_sub_query_set_sparse_values(self.ptr, values.as_ptr(), n)
        };
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Attach HNSW query parameters. Ownership of `params` is
    /// transferred to the sub-query (its inner pointer is consumed;
    /// the wrapper is forgotten).
    pub fn hnsw_params(self, params: HnswQueryParam) -> Self {
        // Upstream `set_*_params` takes ownership. Move the pointer out
        // without running Drop on the wrapper.
        let ptr = params.ptr;
        std::mem::forget(params);
        let code = unsafe { ffi::zvec_sub_query_set_hnsw_params(self.ptr, ptr) };
        let _ = check_error(code as c_int);
        self
    }

    /// Attach IVF query parameters. Ownership of `params` is
    /// transferred to the sub-query.
    pub fn ivf_params(self, params: IVFQueryParam) -> Self {
        let ptr = params.ptr;
        std::mem::forget(params);
        let code = unsafe { ffi::zvec_sub_query_set_ivf_params(self.ptr, ptr) };
        let _ = check_error(code as c_int);
        self
    }

    /// Attach Flat (brute-force) query parameters. Ownership of `params`
    /// is transferred to the sub-query.
    pub fn flat_params(self, params: FlatQueryParam) -> Self {
        let ptr = params.ptr;
        std::mem::forget(params);
        let code = unsafe { ffi::zvec_sub_query_set_flat_params(self.ptr, ptr) };
        let _ = check_error(code as c_int);
        self
    }
}

impl Default for SubQuery {
    fn default() -> Self {
        Self::new().expect("zvec_sub_query_create failed in Default")
    }
}

impl Drop for SubQuery {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_sub_query_destroy(self.ptr) };
        }
    }
}

// SAFETY: SubQuery owns its FFI data and doesn't share state.
unsafe impl Send for SubQuery {}

/// A multi-query combining multiple sub-queries (vector + FTS + sparse
/// etc.) with optional re-ranking of results.
///
/// Build with [`MultiQuery::new`], add sub-queries via
/// [`add_sub_query`](Self::add_sub_query), tune topk / filter / output
/// fields, optionally attach a rerank strategy
/// ([`rerank_rrf`](Self::rerank_rrf) or
/// [`rerank_weighted`](Self::rerank_weighted)), then execute via
/// [`Collection::multi_query`](crate::Collection::multi_query).
pub struct MultiQuery {
    pub(crate) ptr: *mut ffi::zvec_multi_query_t,
}

impl MultiQuery {
    /// Create a new empty multi-query.
    pub fn new() -> Result<Self> {
        let ptr = unsafe { ffi::zvec_multi_query_create() };
        if ptr.is_null() {
            return Err(crate::error::Error::InternalError(
                "zvec_multi_query_create returned null".into(),
            ));
        }
        Ok(Self { ptr })
    }

    /// Add a sub-query. Upstream copies the sub-query internally; the
    /// passed-in `sub` is dropped normally after the copy.
    pub fn add_sub_query(&mut self, sub: SubQuery) -> Result<&mut Self> {
        let code = unsafe { ffi::zvec_multi_query_add_sub_query(self.ptr, sub.ptr) };
        // Per upstream docs ("copied, caller retains ownership"), let
        // `sub` run its Drop impl normally — no mem::forget needed.
        drop(sub);
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Set the final top-K result count.
    pub fn topk(self, k: i32) -> Self {
        let code = unsafe { ffi::zvec_multi_query_set_topk(self.ptr, k) };
        let _ = check_error(code as c_int);
        self
    }

    /// Set a filter expression applied to all sub-queries.
    pub fn filter(self, filter: &str) -> Result<Self> {
        let cstr = CString::new(filter)
            .map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))?;
        let code = unsafe { ffi::zvec_multi_query_set_filter(self.ptr, cstr.as_ptr()) };
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Set the output fields returned in each result document.
    pub fn output_fields(self, fields: &[&str]) -> Result<Self> {
        let fields_c: Vec<CString> = fields
            .iter()
            .map(|f| {
                CString::new(*f)
                    .map_err(|e| crate::error::Error::InvalidArgument(e.to_string()))
            })
            .collect::<Result<_>>()?;
        let mut fields_ptr: Vec<*const std::os::raw::c_char> =
            fields_c.iter().map(|f| f.as_ptr()).collect();
        let code = unsafe {
            ffi::zvec_multi_query_set_output_fields(
                self.ptr,
                fields_ptr.as_mut_ptr(),
                fields_ptr.len(),
            )
        };
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Apply Reciprocal Rank Fusion reranking with the given rank
    /// constant (typical default: 60).
    pub fn rerank_rrf(self, rank_constant: i32) -> Self {
        let code = unsafe { ffi::zvec_multi_query_set_rerank_rrf(self.ptr, rank_constant) };
        let _ = check_error(code as c_int);
        self
    }

    /// Apply weighted-score reranking. `weights` is a flat list of
    /// per-sub-query weights (f64); the i-th entry corresponds to the
    /// i-th sub-query added via [`add_sub_query`](Self::add_sub_query).
    pub fn rerank_weighted(self, weights: &[f64]) -> Result<Self> {
        let code = unsafe {
            ffi::zvec_multi_query_set_rerank_weighted(
                self.ptr,
                weights.as_ptr(),
                weights.len(),
            )
        };
        check_error(code as c_int)?;
        Ok(self)
    }

    /// Raw underlying FFI pointer. Used by
    /// [`Collection::multi_query`](crate::Collection::multi_query).
    pub(crate) fn as_ptr(&self) -> *mut ffi::zvec_multi_query_t {
        self.ptr
    }
}

impl Default for MultiQuery {
    fn default() -> Self {
        Self::new().expect("zvec_multi_query_create failed in Default")
    }
}

impl Drop for MultiQuery {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::zvec_multi_query_destroy(self.ptr) };
        }
    }
}

// SAFETY: MultiQuery owns its FFI data and doesn't share state.
unsafe impl Send for MultiQuery {}
