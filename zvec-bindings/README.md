# zvec-bindings

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)

Idiomatic Rust bindings for [zvec](https://github.com/alibaba/zvec), an open-source in-process vector database built on Alibaba's Proxima engine.

## Features

- Full API coverage matching the Python bindings
- Safe Rust API with proper error handling (`Result<T, Error>`)
- Support for dense and sparse vectors
- HNSW, IVF, and FLAT index types
- Static linking for easy deployment
- Optional thread-safe API via `sync` feature

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
zvec-bindings = "0.2.1"
```

### Prerequisites

- Linux x86_64 or macOS ARM64
- CMake 3.13+
- C++17 compiler (GCC 9+ or Clang 10+)
- Rust 1.70+

## Usage

### Basic Example

```rust
use zvec_bindings::{create_and_open, CollectionSchema, Doc, VectorQuery, VectorSchema};

fn main() -> zvec_bindings::Result<()> {
    // Create schema with a vector field
    let mut schema = CollectionSchema::new("example");
    schema.add_field(VectorSchema::fp32("embedding", 128).into())?;
    
    // Create and open collection
    let collection = create_and_open("./my_vectors", schema)?;
    
    // Insert documents
    let mut doc = Doc::id("doc_1");
    doc.set_vector("embedding", &[0.1, 0.2, 0.3, /* ... */])?;
    collection.insert(&[doc])?;
    
    // Search
    let query = VectorQuery::new("embedding")
        .topk(10)
        .vector(&[0.4, 0.3, 0.2, /* ... */])?;
    
    let results = collection.query(query)?;
    for doc in results.iter() {
        println!("score={:.4}", doc.score());
    }
    
    Ok(())
}
```

### Thread-Safe Usage (Feature: `sync`)

For multi-threaded applications, enable the `sync` feature:

```toml
[dependencies]
zvec-bindings = { version = "0.1", features = ["sync"] }
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `sync` | Enables `SharedCollection` for thread-safe multi-threaded access |
| `static` | Statically links the zvec C++ library |

## API Coverage

- Collection operations: `create_and_open`, `open`, `flush`, `destroy`
- DML: `insert`, `upsert`, `update`, `delete`, `delete_by_filter`
- DQL: `query`, `group_by_query`, `fetch`
- DDL: `create_index`, `drop_index`, `optimize`
- Index types: HNSW, IVF, FLAT, INVERT
- Data types: scalar types, dense vectors, sparse vectors

## License

Apache-2.0
