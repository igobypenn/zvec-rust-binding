//! Hybrid search example: MultiQuery combining two vector sub-queries
//! with Reciprocal Rank Fusion (RRF) reranking.
//!
//! Run with: `cargo run --features sync --example hybrid`

use std::fs;

use zvec_bindings::{create_and_open, CollectionSchema, Doc, MultiQuery, SubQuery, VectorSchema};

fn main() -> zvec_bindings::Result<()> {
    let path = "./zvec_hybrid_example_db";
    let _ = fs::remove_dir_all(path);

    // Two vector fields so we can build two sub-queries (upstream requires
    // MultiQuery to have >= 2 sub-queries).
    let mut schema = CollectionSchema::new("hybrid_example");
    schema.add_field(VectorSchema::fp32("title_vec", 4))?;
    schema.add_field(VectorSchema::fp32("body_vec", 4))?;
    let collection = create_and_open(path, schema)?;

    // Insert docs where both vector fields agree on the axis label.
    let docs: Vec<Doc> = [
        ("axis_x", [1.0, 0.0, 0.0, 0.0]),
        ("axis_y", [0.0, 1.0, 0.0, 0.0]),
        ("axis_z", [0.0, 0.0, 1.0, 0.0]),
        ("diag_xy", [0.7, 0.7, 0.0, 0.0]),
        ("diag_xz", [0.7, 0.0, 0.7, 0.0]),
    ]
    .into_iter()
    .map(|(pk, v)| {
        let mut d = Doc::id(pk);
        d.set_vector("title_vec", &v).unwrap();
        d.set_vector("body_vec", &v).unwrap();
        d
    })
    .collect();
    collection.insert(&docs)?;
    println!("Inserted {} documents", docs.len());

    println!("\n=== MultiQuery with RRF rerank (target = [1,0,0,0]) ===");
    let sub1 = SubQuery::new()?
        .field_name("title_vec")?
        .num_candidates(10)
        .vector(&[1.0, 0.0, 0.0, 0.0])?;
    let sub2 = SubQuery::new()?
        .field_name("body_vec")?
        .num_candidates(10)
        .vector(&[1.0, 0.0, 0.0, 0.0])?;

    let mut mq = MultiQuery::new()?.topk(3).rerank_rrf(60);
    mq.add_sub_query(sub1)?;
    mq.add_sub_query(sub2)?;

    let results = collection.multi_query(&mq)?;
    for doc in results.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }
    println!("({} hits)", results.len());

    println!("\n=== MultiQuery with weighted rerank ===");
    let sub1 = SubQuery::new()?
        .field_name("title_vec")?
        .num_candidates(10)
        .vector(&[1.0, 0.0, 0.0, 0.0])?;
    let sub2 = SubQuery::new()?
        .field_name("body_vec")?
        .num_candidates(10)
        .vector(&[1.0, 0.0, 0.0, 0.0])?;

    let mut mq2 = MultiQuery::new()?.topk(3);
    // Weight the title field 2x the body field (f64 weights, one per sub-query).
    mq2 = mq2.rerank_weighted(&[0.7, 0.3])?;
    mq2.add_sub_query(sub1)?;
    mq2.add_sub_query(sub2)?;

    let results2 = collection.multi_query(&mq2)?;
    for doc in results2.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }
    println!("({} hits)", results2.len());

    collection.destroy()?;
    println!("\nCollection destroyed");
    Ok(())
}
