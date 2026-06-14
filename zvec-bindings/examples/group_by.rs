//! Demonstrates that group_by_query still works after the v0.5.0 migration.
//!
//! Group-by is not exposed by the upstream zvec C API in v0.5.0; we provide it
//! via a tiny static shim (zvec-sys/groupby-shim/) that wraps
//! zvec::Collection::GroupByQuery directly.

use zvec_bindings::{create_and_open, CollectionSchema, Doc, GroupByVectorQuery, VectorSchema};

fn main() -> zvec_bindings::Result<()> {
    let dir = std::path::Path::new("./zvec_example_db_groupby");
    if dir.exists() {
        std::fs::remove_dir_all(dir).ok();
    }

    let mut schema = CollectionSchema::new("groupby_example");
    schema.add_field(VectorSchema::fp32("embedding", 4))?;
    schema.add_field(zvec_bindings::FieldSchema::string("category"))?;

    let collection = create_and_open(dir, schema)?;

    let mut doc1 = Doc::id("doc_1");
    doc1.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;
    doc1.set_string("category", "fruit")?;

    let mut doc2 = Doc::id("doc_2");
    doc2.set_vector("embedding", &[0.95, 0.05, 0.0, 0.0])?;
    doc2.set_string("category", "fruit")?;

    let mut doc3 = Doc::id("doc_3");
    doc3.set_vector("embedding", &[0.0, 1.0, 0.0, 0.0])?;
    doc3.set_string("category", "vegetable")?;

    let mut doc4 = Doc::id("doc_4");
    doc4.set_vector("embedding", &[0.0, 0.9, 0.1, 0.0])?;
    doc4.set_string("category", "vegetable")?;

    collection.insert(&[doc1, doc2, doc3, doc4])?;

    let query = GroupByVectorQuery::new("embedding")
        .group_by("category")
        .group_count(10)
        .group_topk(5)
        .vector(&[1.0, 0.0, 0.0, 0.0])?;

    let results = collection.group_by_query(query)?;
    println!("Group-by returned {} group(s)", results.len());
    for (i, group) in results.iter().enumerate() {
        let value = group.group_by_value();
        let n = group.docs().len();
        println!("  group {}: '{}' ({} docs)", i, value, n);
    }

    let _ = collection.destroy();
    Ok(())
}
