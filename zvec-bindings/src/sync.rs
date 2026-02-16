use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::collection::Collection;
use crate::doc::{Doc, DocList, DocMap, WriteResults};
use crate::error::Result;
use crate::query::{GroupByVectorQuery, GroupResults, VectorQuery};
use crate::schema::CollectionSchema;
use crate::IndexParams;

/// A thread-safe wrapper around [`Collection`] for concurrent access.
///
/// `SharedCollection` uses `Arc<RwLock<Collection>>` internally to provide:
/// - Concurrent reads (multiple threads can query/fetch simultaneously)
/// - Exclusive writes (insert/update/delete are serialized)
///
/// # Example
///
/// ```rust,no_run
/// use zvec_bindings::{create_and_open_shared, SharedCollection, VectorQuery, VectorSchema, CollectionSchema, Doc};
///
/// # fn main() -> zvec_bindings::Result<()> {
/// let mut schema = CollectionSchema::new("my_collection");
/// schema.add_field(VectorSchema::fp32("embedding", 128).into())?;
///
/// let collection = create_and_open_shared("./my_db", schema)?;
///
/// // Clone for sharing between threads (cheap - just Arc clone)
/// let c1 = collection.clone();
/// let c2 = collection.clone();
///
/// // Thread 1: concurrent reads
/// std::thread::spawn(move || {
///     let query = VectorQuery::new("embedding").topk(10).vector(&[0.1, 0.2, 0.3, 0.4]).unwrap();
///     let results = c1.query(query).unwrap();
/// });
///
/// // Thread 2: writes are exclusive
/// std::thread::spawn(move || {
///     let mut doc = Doc::id("doc_1");
///     doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4]).unwrap();
///     c2.insert(&[doc]).unwrap();
/// });
/// # Ok(())
/// # }
/// ```
pub struct SharedCollection {
    inner: Arc<RwLock<Collection>>,
}

impl SharedCollection {
    /// Create a new `SharedCollection` from an existing [`Collection`].
    pub fn new(collection: Collection) -> Self {
        Self {
            inner: Arc::new(RwLock::new(collection)),
        }
    }

    // ===== READ OPERATIONS (take read lock) =====

    /// Execute a vector similarity search query.
    ///
    /// Takes a read lock, allowing concurrent queries.
    pub fn query(&self, query: VectorQuery) -> Result<DocList> {
        let guard = self.inner.read().expect("collection lock poisoned");
        guard.query(query)
    }

    /// Execute a grouped vector similarity search query.
    ///
    /// Takes a read lock, allowing concurrent queries.
    pub fn group_by_query(&self, query: GroupByVectorQuery) -> Result<GroupResults> {
        let guard = self.inner.read().expect("collection lock poisoned");
        guard.group_by_query(query)
    }

    /// Fetch documents by primary key.
    ///
    /// Takes a read lock, allowing concurrent fetches.
    pub fn fetch(&self, pks: &[&str]) -> Result<DocMap> {
        let guard = self.inner.read().expect("collection lock poisoned");
        guard.fetch(pks)
    }

    /// Get the filesystem path where this collection is stored.
    pub fn path(&self) -> Result<String> {
        let guard = self.inner.read().expect("collection lock poisoned");
        guard.path()
    }

    // ===== WRITE OPERATIONS (take write lock) =====

    /// Insert documents into the collection.
    ///
    /// Takes a write lock, exclusive access.
    pub fn insert(&self, docs: &[Doc]) -> Result<WriteResults> {
        let guard = self.inner.write().expect("collection lock poisoned");
        guard.insert(docs)
    }

    /// Upsert documents into the collection.
    ///
    /// Takes a write lock, exclusive access.
    pub fn upsert(&self, docs: &[Doc]) -> Result<WriteResults> {
        let guard = self.inner.write().expect("collection lock poisoned");
        guard.upsert(docs)
    }

    /// Update existing documents in the collection.
    ///
    /// Takes a write lock, exclusive access.
    pub fn update(&self, docs: &[Doc]) -> Result<WriteResults> {
        let guard = self.inner.write().expect("collection lock poisoned");
        guard.update(docs)
    }

    /// Delete documents by primary key.
    ///
    /// Takes a write lock, exclusive access.
    pub fn delete(&self, pks: &[&str]) -> Result<WriteResults> {
        let guard = self.inner.write().expect("collection lock poisoned");
        guard.delete(pks)
    }

    /// Delete documents matching a filter expression.
    ///
    /// Takes a write lock, exclusive access.
    pub fn delete_by_filter(&self, filter: &str) -> Result<()> {
        let guard = self.inner.write().expect("collection lock poisoned");
        guard.delete_by_filter(filter)
    }

    /// Create an index on a vector field.
    ///
    /// Takes a write lock, exclusive access.
    pub fn create_index(&self, column_name: &str, params: IndexParams) -> Result<()> {
        let guard = self.inner.write().expect("collection lock poisoned");
        guard.create_index(column_name, params)
    }

    /// Drop an index from a column.
    ///
    /// Takes a write lock, exclusive access.
    pub fn drop_index(&self, column_name: &str) -> Result<()> {
        let guard = self.inner.write().expect("collection lock poisoned");
        guard.drop_index(column_name)
    }

    /// Optimize the collection for better search performance.
    ///
    /// Takes a write lock, exclusive access.
    pub fn optimize(&self) -> Result<()> {
        let guard = self.inner.write().expect("collection lock poisoned");
        guard.optimize()
    }

    /// Flush pending writes to disk.
    ///
    /// Takes a write lock, exclusive access.
    pub fn flush(&self) -> Result<()> {
        let guard = self.inner.write().expect("collection lock poisoned");
        guard.flush()
    }

    /// Destroy the collection and delete all data.
    ///
    /// Consumes self. This method should only be called when no other
    /// clones of this `SharedCollection` exist.
    pub fn destroy(self) -> Result<()> {
        match Arc::try_unwrap(self.inner) {
            Ok(lock) => {
                let collection = lock.into_inner().expect("collection lock poisoned");
                collection.destroy()
            }
            Err(_) => Err(crate::error::Error::InvalidArgument(
                "cannot destroy SharedCollection: other clones exist".into(),
            )),
        }
    }
}

impl Clone for SharedCollection {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Create and open a new collection wrapped in a [`SharedCollection`].
pub fn create_and_open_shared<P: AsRef<Path>>(
    path: P,
    schema: CollectionSchema,
) -> Result<SharedCollection> {
    let collection = Collection::create_and_open(path, schema)?;
    Ok(SharedCollection::new(collection))
}

/// Open an existing collection wrapped in a [`SharedCollection`].
pub fn open_shared<P: AsRef<Path>>(path: P) -> Result<SharedCollection> {
    let collection = Collection::open(path)?;
    Ok(SharedCollection::new(collection))
}
