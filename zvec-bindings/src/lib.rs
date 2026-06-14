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
//! - HNSW, IVF, FLAT, and FTS index types
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

pub mod fts;
pub mod multi_query;

pub use collection::Collection;
pub use collection::CollectionStats;
pub use collection::IndexParams;
pub use doc::Doc;
pub use error::{Error, Result, StatusCode};
pub use fts::Fts;
pub use multi_query::{MultiQuery, SubQuery};
pub use query::{
    FlatQueryParam, FtsQueryParam, GroupByVectorQuery, HnswQueryParam, IVFQueryParam,
    QueryParam, VectorQuery,
};
pub use rerank::{RrfReRanker, WeightedReRanker};
pub use schema::{CollectionSchema, FieldSchema, VectorSchema};
pub use types::{DataType, IndexType, LogLevel, LogType, MetricType, QuantizeType};

#[cfg(feature = "sync")]
pub use sync::{create_and_open_shared, open_shared, SharedCollection};

/// Initialize the zvec library with default configuration.
///
/// Calls `zvec_initialize(NULL)` (zvec v0.5.0 C API). Optional — the
/// library auto-initializes on first use. Safe to call multiple times;
/// subsequent calls after the first are no-ops at the C layer.
pub fn init() -> Result<()> {
    let code = unsafe { ffi::zvec_initialize(std::ptr::null()) };
    crate::error::check_error(code as std::os::raw::c_int)
}

/// Returns true if the zvec runtime has been initialized.
pub fn is_initialized() -> bool {
    unsafe { ffi::zvec_is_initialized() }
}

/// Shut down the zvec library and release all global resources.
///
/// After this call, the library can be re-initialized with [`init`] or
/// [`LogConfig::apply`]. Existing [`Collection`] handles should not be
/// used after shutdown.
pub fn shutdown() -> Result<()> {
    let code = unsafe { ffi::zvec_shutdown() };
    crate::error::check_error(code as std::os::raw::c_int)
}

/// Returns the zvec runtime version string (e.g. `"0.5.0-g<git> (built ...)"`).
///
/// The string is owned by the library and valid for the lifetime of the
/// process; do not free it.
pub fn version() -> String {
    unsafe {
        let ptr = ffi::zvec_get_version();
        if ptr.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(ptr)
            .to_string_lossy()
            .into_owned()
    }
}

/// Returns the runtime zvec version as `(major, minor, patch)`.
pub fn version_tuple() -> (u32, u32, u32) {
    unsafe {
        (
            ffi::zvec_get_version_major() as u32,
            ffi::zvec_get_version_minor() as u32,
            ffi::zvec_get_version_patch() as u32,
        )
    }
}

/// Returns true if the runtime zvec version is at least the given version.
pub fn check_version(major: u32, minor: u32, patch: u32) -> bool {
    unsafe {
        ffi::zvec_check_version(
            major as std::os::raw::c_int,
            minor as std::os::raw::c_int,
            patch as std::os::raw::c_int,
        )
    }
}

/// Set the process-wide default jieba dict directory (used by the FTS
/// jieba tokenizer when no per-field dict dir is configured).
///
/// Pass an empty string or `None` to clear. This is decoupled from
/// [`init`]; last writer wins.
pub fn set_default_jieba_dict_dir(path: Option<&str>) {
    let cstr = match path {
        Some(s) if !s.is_empty() => {
            Some(std::ffi::CString::new(s).expect("jieba dict dir contains NUL byte"))
        }
        _ => None,
    };
    let ptr = cstr.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null());
    unsafe { ffi::zvec_set_default_jieba_dict_dir(ptr) };
}

// ===== LogConfig =====

/// Builder for the zvec runtime log configuration.
///
/// Construct with [`LogConfig::console`] or [`LogConfig::file`], optionally
/// tune file rotation via [`with_max_file_size`](Self::with_max_file_size)
/// and [`with_overdue_days`](Self::with_overdue_days), then call
/// [`apply`](Self::apply) to initialize the zvec runtime with this log
/// configuration.
///
/// `apply` consumes self and calls `zvec_initialize(config)`; the library
/// must not already be initialized (call [`shutdown`] first to re-configure
/// logging on an already-running runtime).
pub struct LogConfig {
    level: LogLevel,
    is_file: bool,
    dir: Option<String>,
    basename: Option<String>,
    max_file_size_mb: u32,
    overdue_days: u32,
}

impl LogConfig {
    /// Create a console (stderr) log configuration at the given level.
    pub fn console(level: LogLevel) -> Self {
        Self {
            level,
            is_file: false,
            dir: None,
            basename: None,
            max_file_size_mb: 0,
            overdue_days: 0,
        }
    }

    /// Create a file-based log configuration.
    ///
    /// `dir` is the log directory; `basename` is the log file base name.
    /// Default rotation is 100 MB file size with 7-day retention; override
    /// with [`with_max_file_size`](Self::with_max_file_size) and
    /// [`with_overdue_days`](Self::with_overdue_days).
    pub fn file(level: LogLevel, dir: &str, basename: &str) -> Self {
        Self {
            level,
            is_file: true,
            dir: Some(dir.to_string()),
            basename: Some(basename.to_string()),
            max_file_size_mb: 100,
            overdue_days: 7,
        }
    }

    /// Override the per-file maximum size (bytes). Only meaningful for
    /// file loggers.
    pub fn with_max_file_size(mut self, size: u64) -> Self {
        // FFI expects MB; convert from bytes (round up to at least 1 MB).
        self.max_file_size_mb = ((size + (1 << 20) - 1) >> 20) as u32;
        self
    }

    /// Override the log retention window in days. Only meaningful for
    /// file loggers.
    pub fn with_overdue_days(mut self, days: u32) -> Self {
        self.overdue_days = days;
        self
    }

    /// Apply this configuration to the zvec runtime.
    ///
    /// Builds a `zvec_config_data_t` with the log config attached and
    /// calls `zvec_initialize`. Returns an error if initialization fails
    /// (e.g. the library is already initialized — call [`shutdown`] first).
    pub fn apply(self) -> Result<()> {
        use std::os::raw::c_int;

        // 1. Build the log_config.
        let log_config = if self.is_file {
            let dir = self
                .dir
                .as_ref()
                .expect("file logger requires dir; this is a LogConfig bug");
            let basename = self
                .basename
                .as_ref()
                .expect("file logger requires basename; this is a LogConfig bug");
            let dir_c = std::ffi::CString::new(dir.as_str())
                .map_err(|e| Error::InvalidArgument(e.to_string()))?;
            let basename_c = std::ffi::CString::new(basename.as_str())
                .map_err(|e| Error::InvalidArgument(e.to_string()))?;
            unsafe {
                ffi::zvec_config_log_create_file(
                    self.level.into(),
                    dir_c.as_ptr(),
                    basename_c.as_ptr(),
                    self.max_file_size_mb,
                    self.overdue_days,
                )
            }
        } else {
            unsafe { ffi::zvec_config_log_create_console(self.level.into()) }
        };

        if log_config.is_null() {
            return Err(Error::InternalError(
                "zvec_config_log_create_* returned null".into(),
            ));
        }

        // 2. Build the config_data and attach the log config (ownership
        //    of log_config is transferred to config_data on success).
        let config_data = unsafe { ffi::zvec_config_data_create() };
        if config_data.is_null() {
            unsafe { ffi::zvec_config_log_destroy(log_config) };
            return Err(Error::InternalError(
                "zvec_config_data_create returned null".into(),
            ));
        }

        let mut overall: Result<()> = Ok(());
        let set_code =
            unsafe { ffi::zvec_config_data_set_log_config(config_data, log_config) };
        if let Err(e) = crate::error::check_error(set_code as c_int) {
            overall = Err(e);
        } else {
            // 3. Initialize the library with the assembled config.
            let init_code = unsafe { ffi::zvec_initialize(config_data) };
            if let Err(e) = crate::error::check_error(init_code as c_int) {
                overall = Err(e);
            }
        }

        unsafe { ffi::zvec_config_data_destroy(config_data) };
        overall
    }
}

/// List registered metric type names.
///
/// **Deprecated:** the upstream zvec v0.5.0 C API does not expose a metric
/// listing function. This always returns an empty `Vec`.
#[deprecated(note = "upstream zvec v0.5.0 does not expose a metric listing API")]
pub fn list_registered_metrics() -> Vec<String> {
    Vec::new()
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
