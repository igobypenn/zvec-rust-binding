//! # zvec-bindings - Rust bindings for zvec vector database
//!
//! zvec is an open-source in-process vector database built on Alibaba's Proxima engine.
//! This crate provides idiomatic Rust bindings for the zvec C++ library.
//!
//! ## Features
//!
//! - Full API coverage for vector similarity search
//! - Safe Rust API with proper error handling
//! - Support for dense and sparse vectors
//! - HNSW, IVF, and FLAT index types
//! - Static linking for easy deployment
//! - Optional thread-safe [`SharedCollection`] via `sync` feature
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use zvec_bindings::{create_and_open, CollectionSchema, Doc, VectorQuery, VectorSchema};
//!
//! # fn main() -> zvec_bindings::Result<()> {
//! // Create schema with a vector field
//! let mut schema = CollectionSchema::new("my_collection");
//! schema.add_field(VectorSchema::fp32("embedding", 128))?;
//!
//! // Create and open collection
//! let collection = create_and_open("./my_db", schema)?;
//!
//! // Insert a document
//! let mut doc = Doc::id("doc_1");
//! doc.set_vector("embedding", &[0.1, 0.2, 0.3, /* ... */])?;
//! collection.insert(&[doc])?;
//!
//! // Search for similar vectors
//! let query = VectorQuery::new("embedding")
//!     .topk(10)
//!     .vector(&[0.4, 0.3, 0.2, /* ... */])?;
//!
//! let results = collection.query(query)?;
//! for doc in results.iter() {
//!     println!("score={:.4}", doc.score());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Index Types
//!
//! zvec supports multiple index types for different use cases:
//!
//! - **HNSW**: Hierarchical Navigable Small World - fast approximate search
//! - **IVF**: Inverted File - good for large datasets
//! - **FLAT**: Brute force - exact search, good for small datasets
//!
//! ```rust,no_run
//! use zvec_bindings::{IndexParams, MetricType, QuantizeType};
//!
//! # fn main() -> zvec_bindings::Result<()> {
//! # use zvec_bindings::{create_and_open, CollectionSchema, VectorSchema};
//! # let mut schema = CollectionSchema::new("test");
//! # schema.add_field(VectorSchema::fp32("embedding", 128))?;
//! # let collection = create_and_open("./my_db", schema)?;
//! // Create HNSW index
//! let params = IndexParams::hnsw(16, 200, MetricType::L2, QuantizeType::Undefined);
//! collection.create_index("embedding", params)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Thread Safety (feature: `sync`)
//!
//! Enable the `sync` feature for thread-safe collection access:
//!
//! ```toml
//! [dependencies]
//! zvec-bindings = { version = "0.1", features = ["sync"] }
//! ```
//!
//! ```rust,no_run
//! # #[cfg(feature = "sync")]
//! # fn main() -> zvec_bindings::Result<()> {
//! use zvec_bindings::{create_and_open_shared, SharedCollection, VectorQuery, VectorSchema, CollectionSchema, Doc};
//!
//! let mut schema = CollectionSchema::new("my_collection");
//! schema.add_field(VectorSchema::fp32("embedding", 128))?;
//!
//! let collection = create_and_open_shared("./my_db", schema)?;
//!
//! // Clone for sharing between threads (cheap - just Arc clone)
//! let c1 = collection.clone();
//! let c2 = collection.clone();
//!
//! // Thread 1: concurrent reads
//! std::thread::spawn(move || {
//!     let query = VectorQuery::new("embedding").topk(10).vector(&[0.1, 0.2, 0.3, 0.4]).unwrap();
//!     let results = c1.query(query).unwrap();
//! });
//!
//! // Thread 2: writes are exclusive
//! std::thread::spawn(move || {
//!     let mut doc = Doc::id("doc_1");
//!     doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4]).unwrap();
//!     c2.insert(&[doc]).unwrap();
//! });
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "sync"))]
//! # fn main() -> zvec_bindings::Result<()> { Ok(()) }
//! ```

pub use zvec_sys as ffi;

pub mod collection;
pub mod doc;
pub mod error;
pub mod query;
pub mod rerank;
pub mod schema;
pub mod types;

#[cfg(feature = "sync")]
pub mod sync;

pub use collection::Collection;
pub use collection::CollectionStats;
pub use collection::IndexParams;
pub use doc::Doc;
pub use error::{Error, Result, StatusCode};
pub use query::{GroupByVectorQuery, HnswQueryParam, IVFQueryParam, VectorQuery};
pub use rerank::{RrfReRanker, WeightedReRanker};
pub use schema::{CollectionSchema, FieldSchema, VectorSchema};
pub use types::{DataType, IndexType, LogLevel, LogType, MetricType, QuantizeType};

#[cfg(feature = "sync")]
pub use sync::{create_and_open_shared, open_shared, SharedCollection};

pub fn init() -> Result<()> {
    let success = unsafe { ffi::zvec_init() };
    if success {
        Ok(())
    } else {
        Err(Error::InternalError("Failed to initialize zvec".into()))
    }
}

pub fn list_registered_metrics() -> Vec<String> {
    let mut metrics_ptr: *mut *const std::os::raw::c_char = std::ptr::null_mut();
    let count = unsafe { ffi::zvec_list_registered_metrics(&mut metrics_ptr) };

    if metrics_ptr.is_null() || count <= 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        unsafe {
            let ptr = *metrics_ptr.add(i);
            if !ptr.is_null() {
                let cstr = std::ffi::CStr::from_ptr(ptr);
                if let Ok(s) = cstr.to_str() {
                    result.push(s.to_string());
                }
            }
        }
    }
    result
}

pub fn create_and_open<P: AsRef<std::path::Path>>(
    path: P,
    schema: CollectionSchema,
) -> Result<Collection> {
    Collection::create_and_open(path, schema)
}

pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Collection> {
    Collection::open(path)
}

/// Macro for creating documents with terse syntax.
///
/// This macro provides a convenient way to construct documents with
/// primary keys and typed fields using a builder pattern.
///
/// **All forms return `Result<Doc>`.**
///
/// # Basic Usage
///
/// ```rust,no_run
/// use zvec_bindings::{doc, field};
///
/// # fn main() -> zvec_bindings::Result<()> {
/// // Create a document with primary key only
/// let doc1 = doc!("doc_1")?;
///
/// // Create a document with primary key and vector field
/// let doc2 = doc!("doc_2",
///     field::vector("embedding", &[0.1, 0.2, 0.3, 0.4])
/// )?;
///
/// // Create a document with multiple typed fields
/// let doc3 = doc!("doc_3",
///     field::vector("embedding", &[0.1, 0.2, 0.3, 0.4]),
///     field::string("name", "example"),
///     field::int64("count", 42)
/// )?;
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! doc {
    // Just primary key (returns Result<Doc>)
    ($pk:expr) => {{
        let mut doc = $crate::Doc::new();
        doc.set_pk($pk)?;
        ::std::result::Result::Ok::<_, $crate::Error>(doc)
    }};

    // Primary key with fields (returns Result<Doc>)
    ($pk:expr, $($setter:expr),* $(,)?) => {{
        let mut doc = $crate::Doc::new();
        doc.set_pk($pk)?;
        $($setter(&mut doc)?;)*
        ::std::result::Result::Ok::<_, $crate::Error>(doc)
    }};
}

/// Field setter helpers for use with the `doc!` macro.
///
/// These return closures that can be applied to a document.
pub mod field {
    use crate::doc::Doc;
    use crate::error::Result;

    /// Returns a closure that sets a dense vector field.
    pub fn vector<'a>(
        field: &'a str,
        values: &'a [f32],
    ) -> impl FnOnce(&mut Doc) -> Result<()> + 'a {
        move |doc| doc.set_vector(field, values)
    }

    /// Returns a closure that sets a string field.
    pub fn string<'a>(field: &'a str, value: &'a str) -> impl FnOnce(&mut Doc) -> Result<()> + 'a {
        move |doc| doc.set_string(field, value)
    }

    /// Returns a closure that sets an int64 field.
    pub fn int64(field: &str, value: i64) -> impl FnOnce(&mut Doc) -> Result<()> + '_ {
        move |doc| doc.set_int64(field, value)
    }

    /// Returns a closure that sets a float field.
    pub fn float(field: &str, value: f32) -> impl FnOnce(&mut Doc) -> Result<()> + '_ {
        move |doc| doc.set_float(field, value)
    }

    /// Returns a closure that sets a boolean field.
    pub fn boolean(field: &str, value: bool) -> impl FnOnce(&mut Doc) -> Result<()> + '_ {
        move |doc| doc.set_bool(field, value)
    }

    /// Returns a closure that sets a double field.
    pub fn double(field: &str, value: f64) -> impl FnOnce(&mut Doc) -> Result<()> + '_ {
        move |doc| doc.set_double(field, value)
    }

    /// Returns a closure that sets an int32 field.
    pub fn int32(field: &str, value: i32) -> impl FnOnce(&mut Doc) -> Result<()> + '_ {
        move |doc| doc.set_int32(field, value)
    }

    /// Returns a closure that sets a sparse vector field.
    pub fn sparse_vector<'a>(
        field: &'a str,
        indices: &'a [u32],
        values: &'a [f32],
    ) -> impl FnOnce(&mut Doc) -> Result<()> + 'a {
        move |doc| doc.set_sparse_vector(field, indices, values)
    }
}
