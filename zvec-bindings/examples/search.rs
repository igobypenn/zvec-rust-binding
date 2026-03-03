use std::fs;

use zvec_bindings::{create_and_open, CollectionSchema, Doc, VectorQuery, VectorSchema};

fn main() -> zvec_bindings::Result<()> {
    let path = "./zvec_search_db";
    let _ = fs::remove_dir_all(path);

    let mut schema = CollectionSchema::new("search_example");
    schema.add_field(VectorSchema::fp32("embedding", 4))?;

    let collection = create_and_open(path, schema)?;

    let docs: Vec<Doc> = (0..100)
        .map(|i| {
            let mut doc = Doc::id(format!("doc_{}", i));
            let angle = (i as f32) * 0.0628;
            doc.set_vector("embedding", &[angle.cos(), angle.sin(), 0.0, 0.0])
                .unwrap();
            doc
        })
        .collect();

    collection.insert(&docs)?;
    println!("Inserted {} documents", docs.len());

    println!("\n=== Basic search (top 5) ===");
    let query = VectorQuery::new("embedding")
        .topk(5)
        .vector(&[1.0, 0.0, 0.0, 0.0])?;
    let results = collection.query(query)?;
    println!("Results:");
    for doc in results.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }

    println!("\n=== Search with more results (top 10) ===");
    let query = VectorQuery::new("embedding")
        .topk(10)
        .vector(&[0.0, 1.0, 0.0, 0.0])?;
    let results = collection.query(query)?;
    println!("Results:");
    for doc in results.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }

    println!("\n=== Search with vector included ===");
    let query = VectorQuery::new("embedding")
        .topk(3)
        .include_vector(true)
        .vector(&[1.0, 0.0, 0.0, 0.0])?;
    let results = collection.query(query)?;
    println!("Results with vectors:");
    for doc in results.iter() {
        let vec = doc.get_vector("embedding").unwrap();
        println!("  {} score={:.4} vec={:.3?}", doc.pk(), doc.score(), vec);
    }

    collection.destroy()?;
    println!("\nCollection destroyed");

    Ok(())
}
