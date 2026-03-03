use zvec_bindings::{create_and_open, CollectionSchema, Doc, VectorQuery, VectorSchema};

fn main() -> zvec_bindings::Result<()> {
    let path = "./zvec_example_db";

    let mut schema = CollectionSchema::new("example");
    schema.add_field(VectorSchema::fp32("embedding", 4))?;

    let collection = create_and_open(path, schema)?;

    let mut doc1 = Doc::id("doc_1");
    doc1.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;

    let mut doc2 = Doc::id("doc_2");
    doc2.set_vector("embedding", &[0.2, 0.3, 0.4, 0.1])?;

    let mut doc3 = Doc::id("doc_3");
    doc3.set_vector("embedding", &[0.3, 0.4, 0.1, 0.2])?;

    let docs = vec![doc1, doc2, doc3];

    let results = collection.insert(&docs)?;
    println!("Inserted {} documents", results.len());

    let query = VectorQuery::new("embedding")
        .topk(10)
        .vector(&[0.4, 0.3, 0.3, 0.1])?;

    let search_results = collection.query(query)?;

    println!("Search results:");
    for doc in search_results.iter() {
        println!("  id={}, score={:.4}", doc.pk(), doc.score());
    }

    collection.flush()?;

    Ok(())
}
