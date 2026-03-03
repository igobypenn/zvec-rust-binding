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
}
