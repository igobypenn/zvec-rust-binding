# zvec-rust-bindings

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Rust bindings for [zvec](https://github.com/alibaba/zvec), an open-source in-process vector database built on Alibaba's Proxima engine.

## Features

- Full API coverage matching the Python bindings
- Safe Rust API with proper error handling (`Result<T, Error>`)
- Support for dense and sparse vectors
- HNSW, IVF, and FLAT index types
- Static linking for easy deployment
- Optional thread-safe API via `sync` feature

## Installation

### Prerequisites

**System Requirements:**
- Linux x86_64 or macOS ARM64
- CMake 3.13+
- C++17 compiler (GCC 9+ or Clang 10+)
- Rust 1.70+

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config liblz4-dev
```

**macOS (Homebrew):**
```bash
brew install cmake git lz4
```

### Build

```bash
# Clone the repository
git clone https://github.com/your-org/zvec-rust-bindings.git
cd zvec-rust-bindings

# Build (downloads and compiles zvec automatically)
cargo build --release
```

The first build:
1. Downloads zvec source from GitHub (~500MB with submodules)
2. Compiles zvec C++ library (~5-15 minutes)
3. Compiles the Rust bindings

Subsequent builds are fast - source is cached in `vendor/zvec/`.

> 📖 **Need help?** See the [Build Guide](docs/BUILD.md) for detailed instructions and troubleshooting.

### CPU Architecture Optimizations

zvec can be compiled with CPU-specific optimizations for better vector search performance. By default, zvec auto-detects your CPU architecture.

To manually specify a target architecture, set the `ZVEC_CPU_ARCH` environment variable:

```bash
# ARM architectures
ZVEC_CPU_ARCH=ARMV8A cargo build --release

# Intel with AVX-512
ZVEC_CPU_ARCH=SKYLAKE_AVX512 cargo build --release

# AMD Zen 3
ZVEC_CPU_ARCH=ZEN3 cargo build --release
```

**Available options:**

| Architecture | Options |
|-------------|---------|
| **Intel** | `NEHALEM`, `SANDYBRIDGE`, `HASWELL`, `BROADWELL`, `SKYLAKE`, `SKYLAKE_AVX512`, `SAPPHIRERAPIDS`, `EMERALDRAPIDS`, `GRANITERAPIDS` |
| **AMD** | `ZEN1`, `ZEN2`, `ZEN3` |
| **ARM** | `ARMV8A`, `ARMV8.1A`, `ARMV8.2A`, `ARMV8.3A`, `ARMV8.4A`, `ARMV8.5A`, `ARMV8.6A` |

> 📖 **Full list:** See [zvec's cmake/option.cmake](https://github.com/alibaba/zvec/blob/main/cmake/option.cmake) for all available options and [GCC x86 Options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html) for architecture details.

> ⚠️ **Note:** Changing `ZVEC_CPU_ARCH` requires a clean rebuild:
> ```bash
> rm -rf vendor/zvec/build
> ZVEC_CPU_ARCH=SKYLAKE cargo build --release
> ```

### Build Configuration

Additional environment variables for customizing the build:

| Variable | Default | Description |
|----------|---------|-------------|
| `ZVEC_GIT_REF` | `v0.2.0` | zvec version to download (tag or branch) |
| `ZVEC_BUILD_TYPE` | `Release` | CMake build type |
| `ZVEC_BUILD_PARALLEL` | CPU count | Parallel make jobs |
| `ZVEC_CPU_ARCH` | auto | CPU architecture optimization |
| `ZVEC_OPENMP` | off | Set to `ON` or `1` to enable OpenMP |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
zvec-bindings = "0.1"
```

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

```rust
use zvec_bindings::{create_and_open_shared, SharedCollection, VectorQuery, VectorSchema, CollectionSchema, Doc};

fn main() -> zvec_bindings::Result<()> {
    let mut schema = CollectionSchema::new("example");
    schema.add_field(VectorSchema::fp32("embedding", 128).into())?;
    
    let collection = create_and_open_shared("./my_vectors", schema)?;
    
    // Clone for sharing between threads (cheap - just Arc clone)
    let c1 = collection.clone();
    let c2 = collection.clone();
    
    // Thread 1: concurrent reads
    std::thread::spawn(move || {
        let query = VectorQuery::new("embedding").topk(10).vector(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        let results = c1.query(query).unwrap();
    });
    
    // Thread 2: writes are exclusive
    std::thread::spawn(move || {
        let mut doc = Doc::id("doc_1");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4]).unwrap();
        c2.insert(&[doc]).unwrap();
    });
    
    Ok(())
}
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `sync` | Enables `SharedCollection` for thread-safe multi-threaded access |
| `static` | Statically links the zvec C++ library |

## API Coverage

### Collection Operations
- ✅ `create_and_open` - Create and open a new collection
- ✅ `open` - Open an existing collection
- ✅ `flush` - Flush data to disk
- ✅ `destroy` - Delete collection storage
- ✅ `stats` - Get collection statistics
- ✅ `schema` - Get collection schema

### DML Operations
- ✅ `insert` - Insert documents
- ✅ `upsert` - Insert or update documents
- ✅ `update` - Update existing documents
- ✅ `delete` - Delete documents by primary key
- ✅ `delete_by_filter` - Delete documents matching a filter

### DQL Operations
- ✅ `query` - Vector similarity search
- ✅ `group_by_query` - Grouped vector search
- ✅ `fetch` - Fetch documents by primary key

### DDL Operations
- ✅ `create_index` - Create an index on a column
- ✅ `drop_index` - Drop an index
- ✅ `optimize` - Optimize the collection
- ✅ `add_column` - Add a new column
- ✅ `drop_column` - Drop a column
- ✅ `alter_column` - Alter a column

### Index Types
- ✅ HNSW (Hierarchical Navigable Small World)
- ✅ IVF (Inverted File)
- ✅ FLAT (Brute Force)
- ✅ INVERT (Inverted Index)

### Query Parameters
- ✅ `HnswQueryParam` - HNSW query parameters (ef_search)
- ✅ `IVFQueryParam` - IVF query parameters (nprobe)

### Re-ranking
- ✅ `RrfReRanker` - Reciprocal Rank Fusion re-ranker
- ✅ `WeightedReRanker` - Weighted score re-ranker

### Data Types
- ✅ Scalar types (bool, int32, int64, float, double, string)
- ✅ Dense vectors (fp16, fp32, fp64, int4, int8, int16)
- ✅ Sparse vectors (fp16, fp32)

### Enums
- ✅ `LogLevel` - Logging severity levels
- ✅ `LogType` - Log output destination
- ✅ `StatusCode` - Operation status codes
- ✅ `MetricType` - Distance metrics (L2, IP, Cosine)
- ✅ `QuantizeType` - Quantization types

### Global Functions
- ✅ `init()` - Initialize zvec library
- ✅ `list_registered_metrics()` - List available metrics

## Project Structure

```
zvec-rust-bindings/
├── vendor/zvec/         # Downloaded at build time (zvec C++ library)
├── zvec-c-wrapper/      # C API wrapper for zvec C++ library
│   ├── include/         # C header files
│   └── src/             # C++ implementation
├── zvec-sys/            # Raw FFI bindings (auto-generated by bindgen)
├── zvec-bindings/       # Idiomatic Rust API
│   ├── src/             # Library source
│   ├── examples/        # Usage examples
│   └── tests/           # Integration tests
├── docs/                # Documentation
└── scripts/             # Build and utility scripts
```

## Documentation

- [Build Guide](docs/BUILD.md) - Detailed build instructions and troubleshooting
- [API Documentation](https://docs.rs/zvec-bindings) - Rust API docs

## Development

```bash
# Run tests
cargo test

# Run tests with sync feature
cargo test --features sync

# Run examples
cargo run --example basic
cargo run --example crud
cargo run --example search

# Check code
cargo clippy --all-features
cargo fmt --check

# Build docs
cargo doc --open

# Run coverage
./scripts/coverage.sh
```

## License

Apache-2.0

## Acknowledgments

- [zvec](https://github.com/alibaba/zvec) - The underlying vector database
- [Alibaba Proxima](https://github.com/alibaba/proxima) - The core vector search engine
