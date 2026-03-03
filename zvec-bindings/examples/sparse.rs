use std::fs;

use zvec_bindings::{create_and_open, CollectionSchema, Doc, VectorQuery, VectorSchema};

fn main() -> zvec_bindings::Result<()> {
    let path = "./zvec_sparse_db";
    let _ = fs::remove_dir_all(path);

    let mut schema = CollectionSchema::new("sparse_example");
    schema.add_field(VectorSchema::sparse_fp32_with_dim(
        "sparse_embedding",
        10000,
    ))?;

    let collection = create_and_open(path, schema)?;

    println!("=== Inserting sparse vectors ===");

    let mut doc1 = Doc::id("doc_1");
    doc1.set_sparse_vector(
        "sparse_embedding",
        &[1, 100, 500, 9000],
        &[0.8, 0.2, 0.5, 0.1],
    )?;

    let mut doc2 = Doc::id("doc_2");
    doc2.set_sparse_vector(
        "sparse_embedding",
        &[1, 200, 500, 8000],
        &[0.9, 0.3, 0.4, 0.2],
    )?;

    let mut doc3 = Doc::id("doc_3");
    doc3.set_sparse_vector(
        "sparse_embedding",
        &[100, 200, 300, 400],
        &[0.1, 0.2, 0.3, 0.4],
    )?;

    collection.insert(&[doc1, doc2, doc3])?;
    println!("Inserted 3 sparse vector documents");

    println!("\n=== Searching with sparse query ===");

    let query = VectorQuery::new("sparse_embedding")
        .topk(10)
        .sparse_vector(&[1, 100, 500], &[0.9, 0.8, 0.7])?;

    let results = collection.query(query)?;
    println!("Search results:");
    for doc in results.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }

    println!("\n=== Different sparse query ===");

    let query = VectorQuery::new("sparse_embedding")
        .topk(10)
        .sparse_vector(&[200, 300, 400], &[0.5, 0.5, 0.5])?;

    let results = collection.query(query)?;
    println!("Search results:");
    for doc in results.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }

    collection.destroy()?;
    println!("\nCollection destroyed");

    Ok(())
}
