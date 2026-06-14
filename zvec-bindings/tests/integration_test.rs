use zvec_bindings::{
    create_and_open, Collection, CollectionSchema, DataType, Doc, FieldSchema, GroupByVectorQuery,
    IndexParams, MetricType, QuantizeType, VectorQuery, VectorSchema,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_is_vector() {
        use DataType::*;

        assert!(VectorFp32.is_vector());
        assert!(VectorFp16.is_vector());
        assert!(SparseVectorFp32.is_vector());
        assert!(!String.is_vector());
        assert!(!Int64.is_vector());
    }

    #[test]
    fn test_data_type_is_sparse() {
        use DataType::*;

        assert!(SparseVectorFp32.is_sparse_vector());
        assert!(SparseVectorFp16.is_sparse_vector());
        assert!(!VectorFp32.is_sparse_vector());
        assert!(!String.is_sparse_vector());
    }

    #[test]
    fn test_version_helpers() {
        // Version string should be non-empty and contain at least one digit.
        let v = zvec_bindings::version();
        assert!(!v.is_empty(), "version string should not be empty");
        assert!(
            v.chars().any(|c| c.is_ascii_digit()),
            "version string should contain a digit: got {v:?}"
        );

        // version_tuple should be (>= 0, >= 0, >= 0) — at minimum 0.0.0.
        let (major, minor, patch) = zvec_bindings::version_tuple();
        assert!(
            (major, minor, patch) >= (0, 0, 0),
            "version_tuple should be non-negative: got ({major}, {minor}, {patch})"
        );
    }

    #[test]
    fn test_check_version_self_consistency() {
        // The runtime reports some (major, minor, patch); check_version
        // for that exact version must succeed.
        let (major, minor, patch) = zvec_bindings::version_tuple();
        assert!(
            zvec_bindings::check_version(major, minor, patch),
            "check_version should succeed for the runtime's own version"
        );
    }

    #[test]
    fn test_is_initialized_auto_init() {
        // Library auto-initializes on first use (or via init()).
        // After any of the above tests have run, is_initialized should be
        // true. We can't easily force a clean state, so just assert that
        // the helper returns a bool without panicking.
        let _ = zvec_bindings::is_initialized();
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use tempfile::TempDir;

    fn tempdir() -> zvec_bindings::Result<TempDir> {
        tempfile::tempdir().map_err(|e| zvec_bindings::Error::InternalError(e.to_string()))
    }

    fn create_test_collection(path: &std::path::Path) -> zvec_bindings::Result<Collection> {
        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4))?;
        create_and_open(path, schema)
    }

    #[test]
    fn test_collection_create_and_insert() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let collection = create_test_collection(&path)?;

        let mut doc = Doc::id("test_1");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;

        let results = collection.insert(&[doc])?;
        assert_eq!(results.len(), 1);
        assert!(results.get(0).unwrap().is_ok());

        Ok(())
    }

    #[test]
    fn test_collection_fetch() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_test_collection(&path)?;

        let mut doc = Doc::id("fetch_test");
        doc.set_vector("embedding", &[0.5, 0.5, 0.5, 0.5])?;
        collection.insert(&[doc])?;

        let fetched = collection.fetch(&["fetch_test"])?;
        assert_eq!(fetched.len(), 1);
        assert!(fetched.get("fetch_test").is_some());

        Ok(())
    }

    #[test]
    fn test_collection_query() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let collection = create_test_collection(&path)?;

        let mut doc1 = Doc::id("doc_1");
        doc1.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;

        let mut doc2 = Doc::id("doc_2");
        doc2.set_vector("embedding", &[0.0, 1.0, 0.0, 0.0])?;

        collection.insert(&[doc1, doc2])?;

        let query = VectorQuery::new("embedding")
            .topk(10)
            .include_vector(true)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;

        let results = collection.query(query)?;

        assert!(!results.is_empty(), "Query should return results");

        let first = results.get(0).expect("Should have at least one result");
        let score = first.score();
        assert!(
            score > 0.0,
            "Score should be positive for matching document"
        );

        Ok(())
    }

    #[test]
    fn test_collection_delete() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let collection = create_test_collection(&path)?;

        let mut doc = Doc::id("delete_test");
        doc.set_vector("embedding", &[0.5, 0.5, 0.5, 0.5])?;

        collection.insert(&[doc])?;

        let fetched = collection.fetch(&["delete_test"])?;
        assert!(
            fetched.get("delete_test").is_some(),
            "Document should exist before delete"
        );

        let results = collection.delete(&["delete_test"])?;
        assert_eq!(results.len(), 1);
        assert!(
            results.get(0).unwrap().is_ok(),
            "Delete operation should succeed"
        );

        Ok(())
    }

    #[test]
    fn test_collection_upsert() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_test_collection(&path)?;

        let mut doc = Doc::id("upsert_test");
        doc.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;
        collection.insert(&[doc])?;

        let mut updated_doc = Doc::id("upsert_test");
        updated_doc.set_vector("embedding", &[0.0, 1.0, 0.0, 0.0])?;
        let results = collection.upsert(&[updated_doc])?;
        assert_eq!(results.len(), 1);
        assert!(results.get(0).unwrap().is_ok());

        Ok(())
    }

    #[test]
    fn test_collection_update() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_test_collection(&path)?;

        let mut doc = Doc::id("update_test");
        doc.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;
        collection.insert(&[doc])?;

        let mut updated_doc = Doc::id("update_test");
        updated_doc.set_vector("embedding", &[0.5, 0.5, 0.5, 0.5])?;
        let results = collection.update(&[updated_doc])?;
        assert_eq!(results.len(), 1);

        Ok(())
    }

    #[test]
    fn test_collection_scalar_fields() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4))?;
        schema.add_field(FieldSchema::int64("count"))?;
        schema.add_field(FieldSchema::float("score"))?;
        let collection = create_and_open(&path, schema)?;

        let mut doc = Doc::id("scalar_test");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        doc.set_int64("count", 42)?;
        doc.set_float("score", 1.5)?;
        collection.insert(&[doc])?;

        let fetched = collection.fetch(&["scalar_test"])?;
        let doc = fetched.get("scalar_test").expect("Document should exist");
        assert_eq!(doc.get_int64("count"), Some(42));
        assert!(doc.get_float("score").is_some());

        Ok(())
    }

    #[test]
    fn test_collection_multiple_vectors() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding_a", 4))?;
        schema.add_field(VectorSchema::fp32("embedding_b", 4))?;
        let collection = create_and_open(&path, schema)?;

        let mut doc = Doc::id("multi_vec");
        doc.set_vector("embedding_a", &[1.0, 0.0, 0.0, 0.0])?;
        doc.set_vector("embedding_b", &[0.0, 1.0, 0.0, 0.0])?;
        collection.insert(&[doc])?;

        let query = VectorQuery::new("embedding_a")
            .topk(10)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;
        let results = collection.query(query)?;
        assert!(!results.is_empty());

        Ok(())
    }

    #[test]
    fn test_collection_create_hnsw_index() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_test_collection(&path)?;

        let params = IndexParams::hnsw(16, 200, MetricType::L2, QuantizeType::Undefined);
        collection.create_index("embedding", params)?;

        Ok(())
    }

    #[test]
    fn test_collection_create_flat_index() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_test_collection(&path)?;

        let params = IndexParams::flat(MetricType::L2, QuantizeType::Undefined);
        collection.create_index("embedding", params)?;

        Ok(())
    }

    #[test]
    fn test_sparse_vector() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::sparse_fp32_with_dim("sparse_embedding", 1000))?;
        let collection = create_and_open(&path, schema)?;

        let mut doc = Doc::id("sparse_test");
        doc.set_sparse_vector("sparse_embedding", &[1, 5, 10], &[0.5, 0.3, 0.2])?;
        collection.insert(&[doc])?;

        let query = VectorQuery::new("sparse_embedding")
            .topk(10)
            .sparse_vector(&[1, 5, 10], &[0.5, 0.3, 0.2])?;
        let results = collection.query(query)?;
        assert!(!results.is_empty());

        Ok(())
    }

    #[test]
    fn test_collection_group_by_query() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4))?;
        schema.add_field(FieldSchema::string("category"))?;
        let collection = create_and_open(&path, schema)?;

        let mut doc1 = Doc::id("doc_1");
        doc1.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;
        doc1.set_string("category", "cat_a")?;

        let mut doc2 = Doc::id("doc_2");
        doc2.set_vector("embedding", &[0.9, 0.1, 0.0, 0.0])?;
        doc2.set_string("category", "cat_a")?;

        let mut doc3 = Doc::id("doc_3");
        doc3.set_vector("embedding", &[0.0, 1.0, 0.0, 0.0])?;
        doc3.set_string("category", "cat_b")?;

        collection.insert(&[doc1, doc2, doc3])?;

        let query = GroupByVectorQuery::new("embedding")
            .group_by("category")
            .group_count(10)
            .group_topk(5)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;

        let results = collection.group_by_query(query)?;
        assert!(!results.is_empty());

        Ok(())
    }

    #[test]
    fn test_collection_delete_by_filter() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4))?;
        schema.add_field(FieldSchema::string("status"))?;
        let collection = create_and_open(&path, schema)?;

        let mut doc1 = Doc::id("doc_1");
        doc1.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        doc1.set_string("status", "active")?;

        let mut doc2 = Doc::id("doc_2");
        doc2.set_vector("embedding", &[0.5, 0.6, 0.7, 0.8])?;
        doc2.set_string("status", "inactive")?;

        collection.insert(&[doc1, doc2])?;

        collection.delete_by_filter("status = 'inactive'")?;

        Ok(())
    }

    // ==================== FTS tests ====================

    #[test]
    fn test_fts_create_index_and_query() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("fts_db");

        // Collection with a string field (content) for FTS and a vector
        // field so we can also test hybrid queries later.
        //
        // FTS indexes MUST be declared on the FieldSchema before the
        // collection is created — `create_index()` after the fact is a
        // no-op for FTS in upstream zvec v0.5.0.
        let mut content_field = FieldSchema::string("content");
        let fts_params = IndexParams::fts("standard", Some(&["lowercase"]), None)?;
        content_field.set_index_params(&fts_params)?;

        let mut schema = CollectionSchema::new("fts_test");
        schema.add_field(content_field)?;
        schema.add_field(VectorSchema::fp32("embedding", 4))?;
        let collection = create_and_open(&path, schema)?;

        // Insert 5 docs; 4 contain "hello", 1 is an outlier.
        let make_doc = |pk: &str, content: &str, vec: &[f32]| -> zvec_bindings::Result<Doc> {
            let mut d = Doc::id(pk);
            d.set_string("content", content)?;
            d.set_vector("embedding", vec)?;
            Ok(d)
        };

        let docs = [
            make_doc("pk_0", "hello world", &[1.0, 0.0, 0.0, 0.0])?,
            make_doc("pk_1", "hello foo bar", &[0.9, 0.1, 0.0, 0.0])?,
            make_doc("pk_2", "hello baz", &[0.8, 0.2, 0.0, 0.0])?,
            make_doc("pk_3", "hello hello", &[0.7, 0.3, 0.0, 0.0])?,
            make_doc("pk_4", "nothing relevant", &[0.0, 1.0, 0.0, 0.0])?,
        ];
        let results = collection.insert(&docs)?;
        assert_eq!(results.len(), 5);
        for r in results.iter() {
            assert!(r.is_ok(), "insert should succeed for each doc");
        }

        // Run an FTS-only query: match "hello" against the content field.
        // We use VectorQuery (targeting the FTS-indexed field) with an
        // attached Fts payload and no query vector.
        let mut fts = zvec_bindings::Fts::new()?;
        fts.set_match_string("hello")?;
        let q = VectorQuery::new("content").topk(10).fts(fts)?;
        let hits = collection.query(q)?;
        assert_eq!(hits.len(), 4, "should match the 4 docs containing 'hello'");
        let pks: std::collections::HashSet<String> =
            hits.iter().map(|d| d.pk().to_string()).collect();
        for expected in ["pk_0", "pk_1", "pk_2", "pk_3"] {
            assert!(pks.contains(expected), "missing expected hit {expected}");
        }
        assert!(!pks.contains("pk_4"), "outlier doc should not match");

        // A term that nothing contains returns no results.
        let mut fts2 = zvec_bindings::Fts::new()?;
        fts2.set_match_string("nonexistent_term_xyz")?;
        let q2 = VectorQuery::new("content").topk(10).fts(fts2)?;
        let misses = collection.query(q2)?;
        assert!(misses.is_empty(), "no docs should match a missing term");

        Ok(())
    }

    // ==================== MultiQuery tests ====================

    #[test]
    fn test_multi_query_single_subquery() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("mq_db");

        // Two vector fields so we can build 2 sub-queries (upstream
        // requires MultiQuery to have >= 2 sub-queries).
        let mut schema = CollectionSchema::new("mq_test");
        schema.add_field(VectorSchema::fp32("embedding", 4))?;
        schema.add_field(VectorSchema::fp32("embedding2", 4))?;
        let collection = create_and_open(&path, schema)?;

        // Insert a handful of unit-axis vectors in both fields.
        let mut doc1 = Doc::id("axis_x");
        doc1.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;
        doc1.set_vector("embedding2", &[1.0, 0.0, 0.0, 0.0])?;
        let mut doc2 = Doc::id("axis_y");
        doc2.set_vector("embedding", &[0.0, 1.0, 0.0, 0.0])?;
        doc2.set_vector("embedding2", &[0.0, 1.0, 0.0, 0.0])?;
        let mut doc3 = Doc::id("axis_z");
        doc3.set_vector("embedding", &[0.0, 0.0, 1.0, 0.0])?;
        doc3.set_vector("embedding2", &[0.0, 0.0, 1.0, 0.0])?;
        collection.insert(&[doc1, doc2, doc3])?;

        // Build a MultiQuery with two sub-queries (one per field), each
        // targeting [1,0,0,0]. Both should agree that "axis_x" is the
        // top match. RRF rerank fuses the two rankings.
        let sub1 = zvec_bindings::SubQuery::new()?
            .field_name("embedding")?
            .num_candidates(10)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;
        let sub2 = zvec_bindings::SubQuery::new()?
            .field_name("embedding2")?
            .num_candidates(10)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;

        let mut mq = zvec_bindings::MultiQuery::new()?.topk(3).rerank_rrf(60);
        mq.add_sub_query(sub1)?;
        mq.add_sub_query(sub2)?;

        let results = collection.multi_query(&mq)?;
        assert!(!results.is_empty(), "multi_query should return results");
        // The closest doc to [1,0,0,0] in both fields is "axis_x"; verify
        // it's the top result.
        let top = results.get(0).expect("should have a top result");
        assert_eq!(top.pk(), "axis_x", "top result should be axis_x");

        Ok(())
    }

    #[test]
    fn test_multi_query_with_topk_limit() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("mq_topk_db");

        let mut schema = CollectionSchema::new("mq_topk_test");
        schema.add_field(VectorSchema::fp32("embedding", 4))?;
        schema.add_field(VectorSchema::fp32("embedding2", 4))?;
        let collection = create_and_open(&path, schema)?;

        // Insert 5 docs around a unit circle in the x-y plane.
        let mut docs = Vec::new();
        for i in 0..5 {
            let mut d = Doc::id(format!("d{i}"));
            let angle = (i as f32) * std::f32::consts::TAU / 5.0;
            let v = [angle.cos(), angle.sin(), 0.0, 0.0];
            d.set_vector("embedding", &v)?;
            d.set_vector("embedding2", &v)?;
            docs.push(d);
        }
        collection.insert(&docs)?;

        // topk=2 should bound the result count at 2.
        let sub1 = zvec_bindings::SubQuery::new()?
            .field_name("embedding")?
            .vector(&[1.0, 0.0, 0.0, 0.0])?;
        let sub2 = zvec_bindings::SubQuery::new()?
            .field_name("embedding2")?
            .vector(&[1.0, 0.0, 0.0, 0.0])?;

        let mut mq = zvec_bindings::MultiQuery::new()?.topk(2).rerank_rrf(60);
        mq.add_sub_query(sub1)?;
        mq.add_sub_query(sub2)?;

        let results = collection.multi_query(&mq)?;
        assert!(
            results.len() <= 2,
            "multi_query should respect topk=2; got {}",
            results.len()
        );
        assert!(
            !results.is_empty(),
            "multi_query should return at least one result"
        );

        Ok(())
    }
}
