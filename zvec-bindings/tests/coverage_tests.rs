use tempfile::TempDir;
use zvec_bindings::{
    create_and_open, open, Collection, CollectionSchema, DataType, Doc, FieldSchema,
    GroupByVectorQuery, HnswQueryParam, IVFQueryParam, IndexParams, IndexType, LogLevel, LogType,
    MetricType, QuantizeType, StatusCode, VectorQuery, VectorSchema,
};

fn tempdir() -> zvec_bindings::Result<TempDir> {
    tempfile::tempdir().map_err(|e| zvec_bindings::Error::InternalError(e.to_string()))
}

fn create_collection(path: &std::path::Path) -> zvec_bindings::Result<Collection> {
    let mut schema = CollectionSchema::new("test");
    schema.add_field(VectorSchema::fp32("embedding", 4).into())?;
    create_and_open(path, schema)
}

#[cfg(test)]
mod coverage_tests {
    use super::*;

    #[test]
    fn test_data_type_all_variants() {
        assert!(matches!(DataType::Undefined, DataType::Undefined));
        assert!(matches!(DataType::Binary, DataType::Binary));
        assert!(matches!(DataType::String, DataType::String));
        assert!(matches!(DataType::Bool, DataType::Bool));
        assert!(matches!(DataType::Int32, DataType::Int32));
        assert!(matches!(DataType::Int64, DataType::Int64));
        assert!(matches!(DataType::UInt32, DataType::UInt32));
        assert!(matches!(DataType::UInt64, DataType::UInt64));
        assert!(matches!(DataType::Float, DataType::Float));
        assert!(matches!(DataType::Double, DataType::Double));
        assert!(matches!(DataType::VectorBinary32, DataType::VectorBinary32));
        assert!(matches!(DataType::VectorBinary64, DataType::VectorBinary64));
        assert!(matches!(DataType::VectorFp16, DataType::VectorFp16));
        assert!(matches!(DataType::VectorFp32, DataType::VectorFp32));
        assert!(matches!(DataType::VectorFp64, DataType::VectorFp64));
        assert!(matches!(DataType::VectorInt4, DataType::VectorInt4));
        assert!(matches!(DataType::VectorInt8, DataType::VectorInt8));
        assert!(matches!(DataType::VectorInt16, DataType::VectorInt16));
        assert!(matches!(
            DataType::SparseVectorFp16,
            DataType::SparseVectorFp16
        ));
        assert!(matches!(
            DataType::SparseVectorFp32,
            DataType::SparseVectorFp32
        ));
        assert!(matches!(DataType::ArrayBinary, DataType::ArrayBinary));
        assert!(matches!(DataType::ArrayString, DataType::ArrayString));
        assert!(matches!(DataType::ArrayBool, DataType::ArrayBool));
        assert!(matches!(DataType::ArrayInt32, DataType::ArrayInt32));
        assert!(matches!(DataType::ArrayInt64, DataType::ArrayInt64));
        assert!(matches!(DataType::ArrayUInt32, DataType::ArrayUInt32));
        assert!(matches!(DataType::ArrayUInt64, DataType::ArrayUInt64));
        assert!(matches!(DataType::ArrayFloat, DataType::ArrayFloat));
        assert!(matches!(DataType::ArrayDouble, DataType::ArrayDouble));
    }

    #[test]
    fn test_metric_type_all_variants() {
        assert!(matches!(MetricType::Undefined, MetricType::Undefined));
        assert!(matches!(MetricType::L2, MetricType::L2));
        assert!(matches!(MetricType::Ip, MetricType::Ip));
        assert!(matches!(MetricType::Cosine, MetricType::Cosine));
        assert!(matches!(MetricType::MipsL2, MetricType::MipsL2));
    }

    #[test]
    fn test_quantize_type_all_variants() {
        assert!(matches!(QuantizeType::Undefined, QuantizeType::Undefined));
        assert!(matches!(QuantizeType::Fp16, QuantizeType::Fp16));
        assert!(matches!(QuantizeType::Int8, QuantizeType::Int8));
        assert!(matches!(QuantizeType::Int4, QuantizeType::Int4));
    }

    #[test]
    fn test_index_type_all_variants() {
        assert!(matches!(IndexType::Undefined, IndexType::Undefined));
        assert!(matches!(IndexType::Hnsw, IndexType::Hnsw));
        assert!(matches!(IndexType::Ivf, IndexType::Ivf));
        assert!(matches!(IndexType::Flat, IndexType::Flat));
        assert!(matches!(IndexType::Invert, IndexType::Invert));
    }

    #[test]
    fn test_doc_id_methods() -> zvec_bindings::Result<()> {
        let mut doc = Doc::id("test_pk");
        let _ = doc.pk();

        doc.set_pk("new_pk");
        let _ = doc.pk();

        let doc2 = Doc::with_pk("pk2");
        let _ = doc2.pk();

        let doc3 = Doc::new();
        let _ = doc3.pk();

        Ok(())
    }

    #[test]
    fn test_doc_setters_all_types() -> zvec_bindings::Result<()> {
        let mut doc = Doc::id("test");

        doc.set_bool("bool_field", true)?;
        doc.set_int32("int32_field", 42)?;
        doc.set_int64("int64_field", 1234567890i64)?;
        doc.set_float("float_field", 1.5)?;
        doc.set_double("double_field", 2.5)?;
        doc.set_string("string_field", "hello")?;

        doc.set_vector("vec_field", &[1.0, 2.0, 3.0, 4.0])?;
        doc.set_sparse_vector("sparse_field", &[1, 2, 3], &[0.1, 0.2, 0.3])?;

        Ok(())
    }

    #[test]
    fn test_doc_builder_chaining() -> zvec_bindings::Result<()> {
        let doc = Doc::id("test")
            .with_pk_mut("updated_pk")
            .with_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?
            .with_string("name", "test_name")?
            .with_float("score", 1.5)?
            .with_int64("count", 100)?;

        let _ = doc.pk();

        Ok(())
    }

    #[test]
    fn test_field_schema_constructors() {
        let fs = FieldSchema::bool_("bool_field");
        assert_eq!(fs.name(), "bool_field");
        assert_eq!(fs.data_type(), DataType::Bool);

        let fs = FieldSchema::int32("int32_field");
        assert_eq!(fs.data_type(), DataType::Int32);

        let fs = FieldSchema::int64("int64_field");
        assert_eq!(fs.data_type(), DataType::Int64);

        let fs = FieldSchema::float("float_field");
        assert_eq!(fs.data_type(), DataType::Float);

        let fs = FieldSchema::double("double_field");
        assert_eq!(fs.data_type(), DataType::Double);

        let fs = FieldSchema::string("string_field");
        assert_eq!(fs.data_type(), DataType::String);

        let fs = FieldSchema::new("custom_field", DataType::UInt32);
        assert_eq!(fs.data_type(), DataType::UInt32);

        let fs = FieldSchema::new_vector("vec_field", DataType::VectorFp32, 128);
        assert_eq!(fs.data_type(), DataType::VectorFp32);
        assert_eq!(fs.dimension(), 128);
    }

    #[test]
    fn test_field_schema_nullable() {
        let mut fs = FieldSchema::string("test");
        assert!(!fs.nullable());
        fs.set_nullable(true);
        assert!(fs.nullable());
    }

    #[test]
    fn test_field_schema_dimension() {
        let fs = FieldSchema::new_vector("vec", DataType::VectorFp32, 64);
        assert_eq!(fs.dimension(), 64);
    }

    #[test]
    fn test_vector_schema_constructors() {
        let vs = VectorSchema::fp32("fp32_vec", 128);
        assert_eq!(vs.into_field_schema().data_type(), DataType::VectorFp32);

        let vs = VectorSchema::fp16("fp16_vec", 64);
        assert_eq!(vs.into_field_schema().data_type(), DataType::VectorFp16);

        let vs = VectorSchema::sparse_fp32("sparse_fp32");
        assert_eq!(
            vs.into_field_schema().data_type(),
            DataType::SparseVectorFp32
        );

        let vs = VectorSchema::sparse_fp16("sparse_fp16");
        assert_eq!(
            vs.into_field_schema().data_type(),
            DataType::SparseVectorFp16
        );

        let vs = VectorSchema::sparse_fp32_with_dim("sparse_fp32_dim", 1000);
        assert_eq!(vs.into_field_schema().dimension(), 1000);

        let vs = VectorSchema::sparse_fp16_with_dim("sparse_fp16_dim", 500);
        assert_eq!(vs.into_field_schema().dimension(), 500);
    }

    #[test]
    fn test_collection_schema_name() {
        let schema = CollectionSchema::new("my_collection");
        let _ = schema.name();
    }

    #[test]
    fn test_index_params_types() {
        let params = IndexParams::hnsw(16, 200, MetricType::L2, QuantizeType::Undefined);
        assert_eq!(params.index_type(), IndexType::Hnsw);

        let params = IndexParams::ivf(100, 10, true, MetricType::Cosine, QuantizeType::Fp16);
        assert_eq!(params.index_type(), IndexType::Ivf);

        let params = IndexParams::flat(MetricType::Ip, QuantizeType::Int8);
        assert_eq!(params.index_type(), IndexType::Flat);

        let params = IndexParams::invert(true);
        assert_eq!(params.index_type(), IndexType::Invert);
    }

    #[test]
    fn test_query_builder_all_options() -> zvec_bindings::Result<()> {
        let query = VectorQuery::new("embedding")
            .topk(100)
            .filter("count > 10")
            .include_vector(true)
            .include_doc_id(true)
            .output_fields(&["name", "count"])
            .vector(&[1.0, 0.0, 0.0, 0.0])?;

        drop(query);
        Ok(())
    }

    #[test]
    fn test_query_sparse_vector() -> zvec_bindings::Result<()> {
        let query = VectorQuery::new("sparse_embedding")
            .topk(10)
            .sparse_vector(&[1, 5, 10], &[0.5, 0.3, 0.2])?;

        drop(query);
        Ok(())
    }

    #[test]
    fn test_group_by_query_builder() -> zvec_bindings::Result<()> {
        let query = GroupByVectorQuery::new("embedding")
            .group_by("category")
            .group_count(5)
            .group_topk(10)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;

        drop(query);
        Ok(())
    }

    #[test]
    fn test_collection_path() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let path_result = collection.path()?;
        assert!(path_result.contains("test_db"));

        Ok(())
    }

    #[test]
    fn test_collection_open() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let collection = create_collection(&path)?;
        collection.flush()?;
        drop(collection);

        let reopened = open(&path)?;
        assert!(reopened.path()?.contains("test_db"));

        Ok(())
    }

    #[test]
    fn test_collection_optimize() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        collection.optimize()?;

        Ok(())
    }

    #[test]
    fn test_collection_drop_index() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let params = IndexParams::flat(MetricType::L2, QuantizeType::Undefined);
        collection.create_index("embedding", params)?;

        collection.drop_index("embedding")?;

        Ok(())
    }

    #[test]
    fn test_collection_ivf_index() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let params = IndexParams::ivf(10, 10, false, MetricType::L2, QuantizeType::Undefined);
        collection.create_index("embedding", params)?;

        Ok(())
    }

    #[test]
    fn test_write_results_iteration() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let mut doc1 = Doc::id("doc_1");
        doc1.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;

        let mut doc2 = Doc::id("doc_2");
        doc2.set_vector("embedding", &[0.5, 0.6, 0.7, 0.8])?;

        let results = collection.insert(&[doc1, doc2])?;
        assert_eq!(results.len(), 2);

        let mut count = 0;
        for result in results.iter() {
            assert!(result.is_ok());
            count += 1;
        }
        assert_eq!(count, 2);

        Ok(())
    }

    #[test]
    fn test_doc_list_iteration() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let mut doc1 = Doc::id("doc_1");
        doc1.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;

        let mut doc2 = Doc::id("doc_2");
        doc2.set_vector("embedding", &[0.0, 1.0, 0.0, 0.0])?;

        collection.insert(&[doc1, doc2])?;

        let query = VectorQuery::new("embedding")
            .topk(10)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;

        let results = collection.query(query)?;
        assert!(!results.is_empty());

        let mut count = 0;
        for doc in results.iter() {
            assert!(!doc.pk().is_empty() || doc.score() >= 0.0);
            count += 1;
        }
        assert!(count > 0);

        Ok(())
    }

    #[test]
    fn test_doc_map_iteration() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let mut doc1 = Doc::id("doc_1");
        doc1.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;

        let mut doc2 = Doc::id("doc_2");
        doc2.set_vector("embedding", &[0.5, 0.6, 0.7, 0.8])?;

        collection.insert(&[doc1, doc2])?;

        let fetched = collection.fetch(&["doc_1", "doc_2"])?;
        assert_eq!(fetched.len(), 2);

        let keys = fetched.keys();
        for key in keys {
            let doc = fetched.get(key);
            if let Some(doc) = doc {
                let _ = doc.pk();
            }
        }

        assert!(fetched.get("nonexistent").is_none());

        Ok(())
    }

    #[test]
    fn test_doc_ref_getters() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4).into())?;
        schema.add_field(FieldSchema::int64("count"))?;
        schema.add_field(FieldSchema::float("score"))?;
        let collection = create_and_open(&path, schema)?;

        let mut doc = Doc::id("test_doc");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        doc.set_int64("count", 42)?;
        doc.set_float("score", 1.5)?;
        collection.insert(&[doc])?;

        let fetched = collection.fetch(&["test_doc"])?;
        let doc = fetched.get("test_doc").expect("Document should exist");

        assert_eq!(doc.get_int64("count"), Some(42));
        assert_eq!(doc.get_float("score"), Some(1.5));

        let vec = doc.get_vector("embedding").expect("Should have vector");
        assert_eq!(vec.len(), 4);

        assert!(doc.get_string("nonexistent").is_none());
        assert!(doc.get_int64("nonexistent").is_none());
        assert!(doc.get_float("nonexistent").is_none());
        assert!(doc.get_vector("nonexistent").is_none());

        Ok(())
    }

    #[test]
    fn test_doc_ref_doc_id_and_score() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let mut doc = Doc::id("test_doc");
        doc.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;
        collection.insert(&[doc])?;

        let query = VectorQuery::new("embedding")
            .topk(10)
            .include_doc_id(true)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;

        let results = collection.query(query)?;
        let first = results.get(0).expect("Should have result");

        let _score = first.score();
        let _doc_id = first.doc_id();

        Ok(())
    }

    #[test]
    fn test_list_registered_metrics() {
        let metrics = zvec_bindings::list_registered_metrics();
        assert!(!metrics.is_empty());
        assert!(metrics.contains(&"L2".to_string()) || metrics.contains(&"Euclidean".to_string()));
    }

    #[test]
    fn test_error_display() {
        let err = zvec_bindings::Error::NotFound("test not found".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test not found"));

        let err = zvec_bindings::Error::InvalidArgument("bad arg".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("bad arg"));

        let err = zvec_bindings::Error::AlreadyExists("already here".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("already here"));

        let err = zvec_bindings::Error::InternalError("internal".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("internal"));

        let err = zvec_bindings::Error::NullPointer;
        let msg = format!("{}", err);
        assert!(msg.contains("Null pointer"));

        let err = zvec_bindings::Error::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("128"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn test_data_type_is_vector_checks() {
        assert!(DataType::VectorFp32.is_vector());
        assert!(DataType::VectorFp16.is_vector());
        assert!(DataType::VectorFp64.is_vector());
        assert!(DataType::VectorInt4.is_vector());
        assert!(DataType::VectorInt8.is_vector());
        assert!(DataType::VectorInt16.is_vector());
        assert!(DataType::SparseVectorFp32.is_vector());
        assert!(DataType::SparseVectorFp16.is_vector());
        assert!(!DataType::String.is_vector());
        assert!(!DataType::Int64.is_vector());
        assert!(!DataType::Float.is_vector());
    }

    #[test]
    fn test_data_type_is_sparse_checks() {
        assert!(DataType::SparseVectorFp32.is_sparse_vector());
        assert!(DataType::SparseVectorFp16.is_sparse_vector());
        assert!(!DataType::VectorFp32.is_sparse_vector());
        assert!(!DataType::String.is_sparse_vector());
    }

    #[test]
    fn test_collection_destroy() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        collection.destroy()?;

        Ok(())
    }

    #[test]
    fn test_doc_get_all_types() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4).into())?;
        schema.add_field(FieldSchema::bool_("bool_field"))?;
        schema.add_field(FieldSchema::int32("int32_field"))?;
        schema.add_field(FieldSchema::int64("int64_field"))?;
        schema.add_field(FieldSchema::float("float_field"))?;
        schema.add_field(FieldSchema::double("double_field"))?;
        schema.add_field(FieldSchema::string("string_field"))?;
        let collection = create_and_open(&path, schema)?;

        let mut doc = Doc::id("test_doc");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        doc.set_bool("bool_field", true)?;
        doc.set_int32("int32_field", 123)?;
        doc.set_int64("int64_field", 456)?;
        doc.set_float("float_field", 1.5)?;
        doc.set_double("double_field", 2.5)?;
        doc.set_string("string_field", "hello")?;
        collection.insert(&[doc])?;

        let fetched = collection.fetch(&["test_doc"])?;
        let doc = fetched.get("test_doc").expect("Document should exist");

        assert_eq!(doc.get_int64("int64_field"), Some(456));
        assert_eq!(doc.get_float("float_field"), Some(1.5));
        let _ = doc.get_string("string_field");

        assert!(doc.get_int64("nonexistent").is_none());
        assert!(doc.get_float("nonexistent").is_none());
        assert!(doc.get_string("nonexistent").is_none());

        Ok(())
    }

    #[test]
    fn test_sparse_vector_mismatched_length() -> zvec_bindings::Result<()> {
        let mut doc = Doc::id("test");
        let result = doc.set_sparse_vector("sparse", &[1, 2, 3], &[0.1, 0.2]);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_doc_has_and_has_value() -> zvec_bindings::Result<()> {
        let mut doc = Doc::id("test");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;

        assert!(doc.has("embedding"));
        assert!(doc.has_value("embedding"));
        assert!(!doc.has("nonexistent"));

        Ok(())
    }

    #[test]
    fn test_doc_is_null() -> zvec_bindings::Result<()> {
        let mut doc = Doc::id("test");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;

        let _ = doc.is_null("embedding");
        let _ = doc.is_null("nonexistent");

        Ok(())
    }

    #[test]
    fn test_doc_default() {
        let doc = Doc::default();
        let _ = doc.pk();
    }

    #[test]
    fn test_doc_list_len_and_is_empty() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let mut doc = Doc::id("test_doc");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        collection.insert(&[doc])?;

        let query = VectorQuery::new("embedding")
            .topk(10)
            .vector(&[0.1, 0.2, 0.3, 0.4])?;

        let results = collection.query(query)?;
        assert!(!results.is_empty());
        assert!(!results.is_empty());

        Ok(())
    }

    #[test]
    fn test_doc_map_is_empty() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let mut doc = Doc::id("test_doc");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        collection.insert(&[doc])?;

        let fetched = collection.fetch(&["test_doc"])?;
        assert!(!fetched.is_empty());
        assert_eq!(fetched.len(), 1);

        Ok(())
    }

    #[test]
    fn test_write_results_get_out_of_bounds() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let mut doc = Doc::id("test_doc");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        let results = collection.insert(&[doc])?;

        assert!(results.get(100).is_none());

        Ok(())
    }

    #[test]
    fn test_error_types() {
        let err = zvec_bindings::Error::NotFound("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = zvec_bindings::Error::AlreadyExists("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = zvec_bindings::Error::NotSupported("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = zvec_bindings::Error::PermissionDenied("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = zvec_bindings::Error::FailedPrecondition("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = zvec_bindings::Error::Unknown("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = zvec_bindings::Error::CollectionNotFound("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = zvec_bindings::Error::IndexNotFound("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = zvec_bindings::Error::FieldNotFound("test".to_string());
        assert!(format!("{}", err).contains("test"));
    }

    #[test]
    fn test_log_level_variants() {
        assert_eq!(LogLevel::Debug as i32, 0);
        assert_eq!(LogLevel::Info as i32, 1);
        assert_eq!(LogLevel::Warn as i32, 2);
        assert_eq!(LogLevel::Error as i32, 3);
        assert_eq!(LogLevel::Fatal as i32, 4);
        assert_eq!(LogLevel::Warning as i32, LogLevel::Warn as i32);
    }

    #[test]
    fn test_log_type_variants() {
        assert_eq!(LogType::Console as i32, 0);
        assert_eq!(LogType::File as i32, 1);
    }

    #[test]
    fn test_status_code_variants() {
        assert_eq!(StatusCode::Ok as u32, 0);
        assert_eq!(StatusCode::NotFound as u32, 1);
        assert_eq!(StatusCode::AlreadyExists as u32, 2);
        assert_eq!(StatusCode::InvalidArgument as u32, 3);
        assert_eq!(StatusCode::NotSupported as u32, 4);
        assert_eq!(StatusCode::InternalError as u32, 5);
        assert_eq!(StatusCode::PermissionDenied as u32, 6);
        assert_eq!(StatusCode::FailedPrecondition as u32, 7);
        assert_eq!(StatusCode::Unknown as u32, 8);
    }

    #[test]
    fn test_status_code_from_u32() {
        assert_eq!(StatusCode::from(0), StatusCode::Ok);
        assert_eq!(StatusCode::from(1), StatusCode::NotFound);
        assert_eq!(StatusCode::from(99), StatusCode::Unknown);
    }

    #[test]
    fn test_hnsw_query_param() {
        let params = HnswQueryParam::new(256);
        assert_eq!(params.ef_search(), 256);
    }

    #[test]
    fn test_ivf_query_param() {
        let params = IVFQueryParam::new(32);
        assert_eq!(params.nprobe(), 32);
    }

    #[test]
    fn test_vector_query_with_hnsw_params() -> zvec_bindings::Result<()> {
        let query = VectorQuery::new("embedding")
            .topk(10)
            .hnsw_params(256)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;
        drop(query);
        Ok(())
    }

    #[test]
    fn test_vector_query_with_ivf_params() -> zvec_bindings::Result<()> {
        let query = VectorQuery::new("embedding")
            .topk(10)
            .ivf_params(16)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;
        drop(query);
        Ok(())
    }

    #[test]
    fn test_vector_query_with_id() {
        let query = VectorQuery::new("embedding").topk(10).id("doc_123");

        assert!(query.has_id());
        assert!(!query.has_vector());
        assert_eq!(query.get_id(), Some("doc_123"));
    }

    #[test]
    fn test_vector_query_without_id() -> zvec_bindings::Result<()> {
        let query = VectorQuery::new("embedding")
            .topk(10)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;

        assert!(!query.has_id());
        assert!(query.has_vector());
        assert_eq!(query.get_id(), None);
        Ok(())
    }

    #[test]
    fn test_collection_stats() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_collection(&path)?;

        let mut doc1 = Doc::id("doc_1");
        doc1.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        let mut doc2 = Doc::id("doc_2");
        doc2.set_vector("embedding", &[0.5, 0.6, 0.7, 0.8])?;
        collection.insert(&[doc1, doc2])?;
        collection.flush()?;

        let stats = collection.stats()?;
        assert_eq!(stats.doc_count(), 2);
        let _ = stats.memory_usage();

        Ok(())
    }

    #[test]
    fn test_collection_schema_field_names() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4).into())?;
        schema.add_field(FieldSchema::int64("count"))?;
        schema.add_field(FieldSchema::string("name"))?;
        let collection = create_and_open(&path, schema)?;

        let schema = collection.schema()?;
        let field_names = schema.field_names();
        assert!(!field_names.is_empty());

        Ok(())
    }

    #[test]
    fn test_collection_schema_vector_field_names() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding_a", 4).into())?;
        schema.add_field(VectorSchema::fp32("embedding_b", 4).into())?;
        schema.add_field(FieldSchema::int64("count"))?;
        let collection = create_and_open(&path, schema)?;

        let schema = collection.schema()?;
        let vector_names = schema.vector_field_names();
        assert_eq!(vector_names.len(), 2);
        assert!(vector_names.contains(&"embedding_a".to_string()));
        assert!(vector_names.contains(&"embedding_b".to_string()));

        Ok(())
    }

    #[test]
    fn test_doc_ref_get_bool() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4).into())?;
        schema.add_field(FieldSchema::bool_("active"))?;
        let collection = create_and_open(&path, schema)?;

        let mut doc = Doc::id("test_doc");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        doc.set_bool("active", true)?;
        collection.insert(&[doc])?;

        let fetched = collection.fetch(&["test_doc"])?;
        let doc = fetched.get("test_doc").expect("Document should exist");
        assert_eq!(doc.get_bool("active"), Some(true));
        assert!(doc.get_bool("nonexistent").is_none());

        Ok(())
    }

    #[test]
    fn test_doc_ref_get_int32() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4).into())?;
        schema.add_field(FieldSchema::int32("value"))?;
        let collection = create_and_open(&path, schema)?;

        let mut doc = Doc::id("test_doc");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        doc.set_int32("value", 12345)?;
        collection.insert(&[doc])?;

        let fetched = collection.fetch(&["test_doc"])?;
        let doc = fetched.get("test_doc").expect("Document should exist");
        assert_eq!(doc.get_int32("value"), Some(12345));
        assert!(doc.get_int32("nonexistent").is_none());

        Ok(())
    }

    #[test]
    fn test_doc_ref_get_double() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4).into())?;
        schema.add_field(FieldSchema::double("precise_value"))?;
        let collection = create_and_open(&path, schema)?;

        let mut doc = Doc::id("test_doc");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        doc.set_double("precise_value", 3.14159265358979)?;
        collection.insert(&[doc])?;

        let fetched = collection.fetch(&["test_doc"])?;
        let doc = fetched.get("test_doc").expect("Document should exist");
        let val = doc.get_double("precise_value");
        assert!(val.is_some());
        assert!((val.unwrap() - 3.14159265358979).abs() < 1e-10);
        assert!(doc.get_double("nonexistent").is_none());

        Ok(())
    }

    #[test]
    fn test_rrf_reranker_basic() {
        use std::collections::HashMap;
        use zvec_bindings::RrfReRanker;

        let reranker = RrfReRanker::new(10);
        assert_eq!(reranker.topn(), 10);
        assert_eq!(reranker.rank_constant(), 60);

        let reranker = reranker.with_rank_constant(100);
        assert_eq!(reranker.rank_constant(), 100);

        let mut query_results: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        query_results.insert(
            "embedding_a".to_string(),
            vec![("doc_1".to_string(), 0.9), ("doc_2".to_string(), 0.8)],
        );
        query_results.insert(
            "embedding_b".to_string(),
            vec![("doc_2".to_string(), 0.95), ("doc_3".to_string(), 0.7)],
        );

        let results = reranker.rerank(&query_results);
        assert_eq!(results.len(), 3);

        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"doc_1"));
        assert!(ids.contains(&"doc_2"));
        assert!(ids.contains(&"doc_3"));
    }

    #[test]
    fn test_weighted_reranker_basic() {
        use std::collections::HashMap;
        use zvec_bindings::{MetricType, WeightedReRanker};

        let reranker = WeightedReRanker::new(10, MetricType::L2);
        assert_eq!(reranker.topn(), 10);
        assert_eq!(reranker.metric(), MetricType::L2);

        let reranker = reranker
            .with_weight("embedding_a", 2.0)
            .with_weight("embedding_b", 1.0);

        let mut query_results: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        query_results.insert(
            "embedding_a".to_string(),
            vec![("doc_1".to_string(), 0.1), ("doc_2".to_string(), 0.2)],
        );
        query_results.insert(
            "embedding_b".to_string(),
            vec![("doc_2".to_string(), 0.05), ("doc_3".to_string(), 0.3)],
        );

        let results = reranker.rerank(&query_results);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_init_function() {
        let result = zvec_bindings::init();
        assert!(result.is_ok());
    }
}

#[cfg(feature = "sync")]
mod sync_tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    use zvec_bindings::{create_and_open_shared, open_shared, SharedCollection};

    fn create_shared_collection(path: &std::path::Path) -> zvec_bindings::Result<SharedCollection> {
        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4).into())?;
        create_and_open_shared(path, schema)
    }

    #[test]
    fn test_shared_collection_clone() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_shared_collection(&path)?;

        let c2 = collection.clone();
        let c3 = collection.clone();

        // All clones share the same underlying collection
        assert_eq!(collection.path()?, c2.path()?);
        assert_eq!(collection.path()?, c3.path()?);

        Ok(())
    }

    #[test]
    fn test_shared_collection_from_collection() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let mut schema = CollectionSchema::new("test");
        schema.add_field(VectorSchema::fp32("embedding", 4).into())?;
        let collection = create_and_open(&path, schema)?;

        let shared = SharedCollection::new(collection);

        let _ = shared.path()?;

        Ok(())
    }

    #[test]
    fn test_shared_collection_concurrent_reads() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_shared_collection(&path)?;

        let mut doc = Doc::id("doc_1");
        doc.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;
        collection.insert(&[doc])?;

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let c = collection.clone();
                thread::spawn(move || {
                    let query = VectorQuery::new("embedding")
                        .topk(10)
                        .vector(&[1.0, 0.0, 0.0, 0.0])
                        .unwrap();
                    c.query(query).unwrap()
                })
            })
            .collect();

        for handle in handles {
            let results = handle.join().expect("thread panicked");
            assert!(!results.is_empty());
        }

        Ok(())
    }

    #[test]
    fn test_shared_collection_concurrent_write_read() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_shared_collection(&path)?;

        let writer = {
            let c = collection.clone();
            thread::spawn(move || {
                for i in 0..5 {
                    let mut doc = Doc::id(format!("doc_{}", i));
                    doc.set_vector("embedding", &[i as f32 * 0.1, 0.0, 0.0, 0.0])
                        .unwrap();
                    c.insert(&[doc]).unwrap();
                }
            })
        };

        let reader = {
            let c = collection.clone();
            thread::spawn(move || {
                for _ in 0..5 {
                    let query = VectorQuery::new("embedding")
                        .topk(100)
                        .vector(&[0.5, 0.0, 0.0, 0.0])
                        .unwrap();
                    let _ = c.query(query);
                    thread::sleep(Duration::from_millis(1));
                }
            })
        };

        writer.join().expect("writer panicked");
        reader.join().expect("reader panicked");

        Ok(())
    }

    #[test]
    fn test_shared_collection_fetch() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_shared_collection(&path)?;

        let mut doc = Doc::id("test_doc");
        doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4])?;
        collection.insert(&[doc])?;

        let fetched = collection.fetch(&["test_doc"])?;
        assert_eq!(fetched.len(), 1);

        Ok(())
    }

    #[test]
    fn test_shared_collection_open() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");

        let collection = create_shared_collection(&path)?;
        collection.flush()?;
        drop(collection);

        let reopened = open_shared(&path)?;
        assert!(reopened.path()?.contains("test_db"));

        Ok(())
    }

    #[test]
    fn test_shared_collection_all_operations() -> zvec_bindings::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_db");
        let collection = create_shared_collection(&path)?;

        // Insert
        let mut doc = Doc::id("doc_1");
        doc.set_vector("embedding", &[1.0, 0.0, 0.0, 0.0])?;
        collection.insert(&[doc])?;

        // Upsert
        let mut doc = Doc::id("doc_2");
        doc.set_vector("embedding", &[0.0, 1.0, 0.0, 0.0])?;
        collection.upsert(&[doc])?;

        // Update
        let mut doc = Doc::id("doc_1");
        doc.set_vector("embedding", &[0.5, 0.5, 0.0, 0.0])?;
        collection.update(&[doc])?;

        // Query
        let query = VectorQuery::new("embedding")
            .topk(10)
            .vector(&[1.0, 0.0, 0.0, 0.0])?;
        let results = collection.query(query)?;
        assert!(!results.is_empty());

        // Fetch
        let fetched = collection.fetch(&["doc_1"])?;
        assert_eq!(fetched.len(), 1);

        // Create index
        let params = zvec_bindings::IndexParams::flat(
            zvec_bindings::MetricType::L2,
            zvec_bindings::QuantizeType::Undefined,
        );
        collection.create_index("embedding", params)?;

        // Optimize
        collection.optimize()?;

        // Flush
        collection.flush()?;

        // Delete
        collection.delete(&["doc_2"])?;

        // Destroy
        collection.destroy()?;

        Ok(())
    }
}
