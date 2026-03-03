use std::fs;

use zvec_bindings::{
    create_and_open, CollectionSchema, Doc, IndexParams, MetricType, QuantizeType, VectorQuery,
    VectorSchema,
};

fn main() -> zvec_bindings::Result<()> {
    let path = "./zvec_indexes_db";
    let _ = fs::remove_dir_all(path);

    let mut schema = CollectionSchema::new("indexes_example");
    schema.add_field(VectorSchema::fp32("embedding", 4))?;

    let collection = create_and_open(path, schema)?;

    let docs: Vec<Doc> = (0..1000)
        .map(|i| {
            let mut doc = Doc::id(format!("doc_{}", i));
            let x = ((i % 10) as f32) / 10.0;
            let y = ((i / 10 % 10) as f32) / 10.0;
            let z = ((i / 100) as f32) / 10.0;
            doc.set_vector("embedding", &[x, y, z, 0.0]).unwrap();
            doc
        })
        .collect();

    collection.insert(&docs)?;
    println!("Inserted {} documents", docs.len());

    println!("\n=== Creating FLAT index ===");
    let params = IndexParams::flat(MetricType::L2, QuantizeType::Undefined);
    collection.create_index("embedding", params)?;
    println!("FLAT index created");

    let query = VectorQuery::new("embedding")
        .topk(5)
        .vector(&[0.5, 0.5, 0.5, 0.0])?;
    let results = collection.query(query)?;
    println!("Search with FLAT index:");
    for doc in results.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }

    collection.drop_index("embedding")?;
    println!("\nDropped FLAT index");

    println!("\n=== Creating HNSW index ===");
    let params = IndexParams::hnsw(16, 200, MetricType::L2, QuantizeType::Undefined);
    collection.create_index("embedding", params)?;
    println!("HNSW index created");

    let query = VectorQuery::new("embedding")
        .topk(5)
        .vector(&[0.5, 0.5, 0.5, 0.0])?;
    let results = collection.query(query)?;
    println!("Search with HNSW index:");
    for doc in results.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }

    println!("\n=== Creating index with cosine similarity ===");
    collection.drop_index("embedding")?;
    let params = IndexParams::hnsw(16, 200, MetricType::Cosine, QuantizeType::Undefined);
    collection.create_index("embedding", params)?;
    println!("HNSW index with cosine similarity created");

    let query = VectorQuery::new("embedding")
        .topk(5)
        .vector(&[0.5, 0.5, 0.5, 0.0])?;
    let results = collection.query(query)?;
    println!("Search with cosine similarity:");
    for doc in results.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }

    collection.destroy()?;
    println!("\nCollection destroyed");

    Ok(())
}
