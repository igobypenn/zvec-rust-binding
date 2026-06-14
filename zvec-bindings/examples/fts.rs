//! Full-Text Search (FTS) example: build an FTS-indexed collection,
//! insert documents, and run an FTS match query.
//!
//! Run with: `cargo run --features sync --example fts`

use std::fs;

use zvec_bindings::{
    create_and_open, CollectionSchema, Doc, FieldSchema, IndexParams, VectorQuery,
};

fn main() -> zvec_bindings::Result<()> {
    let path = "./zvec_fts_example_db";
    let _ = fs::remove_dir_all(path);

    // Declare the FTS index on the FieldSchema BEFORE the collection is
    // created. The "standard" tokenizer works for English; pass "jieba"
    // for Chinese (requires a jieba dict dir, see `set_default_jieba_dict_dir`).
    let mut content_field = FieldSchema::string("content");
    let fts_params = IndexParams::fts("standard", Some(&["lowercase"]), None)?;
    content_field.set_index_params(&fts_params)?;

    let mut schema = CollectionSchema::new("fts_example");
    schema.add_field(content_field)?;
    let collection = create_and_open(path, schema)?;

    // Insert a small corpus. Four docs contain "hello"; one is an outlier.
    let docs: Vec<Doc> = [
        ("pk_0", "hello world"),
        ("pk_1", "hello foo bar"),
        ("pk_2", "hello baz"),
        ("pk_3", "hello hello"),
        ("pk_4", "nothing relevant here"),
    ]
    .into_iter()
    .map(|(pk, content)| {
        let mut d = Doc::id(pk);
        d.set_string("content", content).unwrap();
        d
    })
    .collect();
    collection.insert(&docs)?;
    println!("Inserted {} documents", docs.len());

    println!("\n=== FTS match for 'hello' (top 10) ===");
    let mut fts = zvec_bindings::Fts::new()?;
    fts.set_match_string("hello")?;
    let query = VectorQuery::new("content").topk(10).fts(fts)?;
    let results = collection.query(query)?;
    for doc in results.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }
    println!("({} hits)", results.len());

    println!("\n=== FTS match for 'nothing' (top 10) ===");
    let mut fts2 = zvec_bindings::Fts::new()?;
    fts2.set_match_string("nothing")?;
    let query2 = VectorQuery::new("content").topk(10).fts(fts2)?;
    let results2 = collection.query(query2)?;
    for doc in results2.iter() {
        println!("  {} score={:.4}", doc.pk(), doc.score());
    }
    println!("({} hits)", results2.len());

    collection.destroy()?;
    println!("\nCollection destroyed");
    Ok(())
}
