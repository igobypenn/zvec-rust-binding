use std::fs;

use zvec_bindings::{create_and_open, open, CollectionSchema, Doc, FieldSchema, VectorSchema};

fn main() -> zvec_bindings::Result<()> {
    let path = "./zvec_crud_db";

    let _ = fs::remove_dir_all(path);

    let mut schema = CollectionSchema::new("crud_example");
    schema.add_field(VectorSchema::fp32("embedding", 4))?;
    schema.add_field(FieldSchema::string("name"))?;
    schema.add_field(FieldSchema::int64("count"))?;

    let collection = create_and_open(path, schema)?;

    println!("=== CREATE: Insert documents ===");
    let mut doc1 = Doc::id("doc_1");
    doc1.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;
    doc1.set_string("name", "first")?;
    doc1.set_int64("count", 1)?;

    let mut doc2 = Doc::id("doc_2");
    doc2.set_vector("embedding", &[0.0, 1.0, 0.0, 0.0])?;
    doc2.set_string("name", "second")?;
    doc2.set_int64("count", 2)?;

    let results = collection.insert(&[doc1, doc2])?;
    println!("Inserted {} documents", results.len());

    println!("\n=== READ: Fetch documents ===");
    let fetched = collection.fetch(&["doc_1", "doc_2"])?;
    println!("Fetched {} documents", fetched.len());
    for pk in fetched.keys() {
        println!("  - {}", pk);
    }

    println!("\n=== UPDATE: Modify a document ===");
    let mut updated = Doc::id("doc_1");
    updated.set_vector("embedding", &[0.5, 0.5, 0.0, 0.0])?;
    updated.set_string("name", "updated_first")?;
    updated.set_int64("count", 100)?;
    collection.update(&[updated])?;
    println!("Updated doc_1");

    println!("\n=== UPSERT: Insert or update ===");
    let mut doc3 = Doc::id("doc_3");
    doc3.set_vector("embedding", &[0.0, 0.0, 1.0, 0.0])?;
    doc3.set_string("name", "third")?;
    doc3.set_int64("count", 3)?;
    collection.upsert(&[doc3])?;
    println!("Upserted doc_3");

    println!("\n=== DELETE: Remove a document ===");
    let results = collection.delete(&["doc_2"])?;
    println!("Deleted {} documents", results.len());

    println!("\n=== Verify deletion ===");
    let fetched = collection.fetch(&["doc_2"])?;
    println!("doc_2 exists: {}", fetched.get("doc_2").is_some());

    println!("\n=== Flush and close ===");
    collection.flush()?;
    drop(collection);

    println!("\n=== Reopen collection ===");
    let collection = open("./zvec_crud_db")?;
    let fetched = collection.fetch(&["doc_1", "doc_3"])?;
    println!("Reopened, fetched {} documents", fetched.len());

    collection.destroy()?;
    println!("Collection destroyed");

    Ok(())
}
