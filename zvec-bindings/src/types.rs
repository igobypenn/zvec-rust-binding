#![allow(non_upper_case_globals)]

use crate::ffi::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[derive(Default)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    #[default]
    Warn = 2,
    Error = 3,
    Fatal = 4,
}

impl LogLevel {
    pub const Warning: Self = Self::Warn;
}

impl From<zvec_log_level_t> for LogLevel {
    fn from(t: zvec_log_level_t) -> Self {
        match t {
            x if x == zvec_log_level_t_ZVEC_LOG_LEVEL_DEBUG => LogLevel::Debug,
            x if x == zvec_log_level_t_ZVEC_LOG_LEVEL_INFO => LogLevel::Info,
            x if x == zvec_log_level_t_ZVEC_LOG_LEVEL_WARN => LogLevel::Warn,
            x if x == zvec_log_level_t_ZVEC_LOG_LEVEL_ERROR => LogLevel::Error,
            x if x == zvec_log_level_t_ZVEC_LOG_LEVEL_FATAL => LogLevel::Fatal,
            _ => LogLevel::Warn,
        }
    }
}

impl From<LogLevel> for zvec_log_level_t {
    fn from(t: LogLevel) -> Self {
        match t {
            LogLevel::Debug => zvec_log_level_t_ZVEC_LOG_LEVEL_DEBUG,
            LogLevel::Info => zvec_log_level_t_ZVEC_LOG_LEVEL_INFO,
            LogLevel::Warn => zvec_log_level_t_ZVEC_LOG_LEVEL_WARN,
            LogLevel::Error => zvec_log_level_t_ZVEC_LOG_LEVEL_ERROR,
            LogLevel::Fatal => zvec_log_level_t_ZVEC_LOG_LEVEL_FATAL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[derive(Default)]
pub enum LogType {
    #[default]
    Console = 0,
    File = 1,
}

impl From<zvec_log_type_t> for LogType {
    fn from(t: zvec_log_type_t) -> Self {
        match t {
            x if x == zvec_log_type_t_ZVEC_LOG_TYPE_CONSOLE => LogType::Console,
            x if x == zvec_log_type_t_ZVEC_LOG_TYPE_FILE => LogType::File,
            _ => LogType::Console,
        }
    }
}

impl From<LogType> for zvec_log_type_t {
    fn from(t: LogType) -> Self {
        match t {
            LogType::Console => zvec_log_type_t_ZVEC_LOG_TYPE_CONSOLE,
            LogType::File => zvec_log_type_t_ZVEC_LOG_TYPE_FILE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DataType {
    Undefined = 0,
    Binary = 1,
    String = 2,
    Bool = 3,
    Int32 = 4,
    Int64 = 5,
    UInt32 = 6,
    UInt64 = 7,
    Float = 8,
    Double = 9,
    VectorBinary32 = 20,
    VectorBinary64 = 21,
    VectorFp16 = 22,
    VectorFp32 = 23,
    VectorFp64 = 24,
    VectorInt4 = 25,
    VectorInt8 = 26,
    VectorInt16 = 27,
    SparseVectorFp16 = 30,
    SparseVectorFp32 = 31,
    ArrayBinary = 40,
    ArrayString = 41,
    ArrayBool = 42,
    ArrayInt32 = 43,
    ArrayInt64 = 44,
    ArrayUInt32 = 45,
    ArrayUInt64 = 46,
    ArrayFloat = 47,
    ArrayDouble = 48,
}

impl From<zvec_data_type_t> for DataType {
    fn from(t: zvec_data_type_t) -> Self {
        match t {
            x if x == ZVEC_DATA_TYPE_UNDEFINED as u32 => DataType::Undefined,
            x if x == ZVEC_DATA_TYPE_BINARY as u32 => DataType::Binary,
            x if x == ZVEC_DATA_TYPE_STRING as u32 => DataType::String,
            x if x == ZVEC_DATA_TYPE_BOOL as u32 => DataType::Bool,
            x if x == ZVEC_DATA_TYPE_INT32 as u32 => DataType::Int32,
            x if x == ZVEC_DATA_TYPE_INT64 as u32 => DataType::Int64,
            x if x == ZVEC_DATA_TYPE_UINT32 as u32 => DataType::UInt32,
            x if x == ZVEC_DATA_TYPE_UINT64 as u32 => DataType::UInt64,
            x if x == ZVEC_DATA_TYPE_FLOAT as u32 => DataType::Float,
            x if x == ZVEC_DATA_TYPE_DOUBLE as u32 => DataType::Double,
            x if x == ZVEC_DATA_TYPE_VECTOR_BINARY32 as u32 => DataType::VectorBinary32,
            x if x == ZVEC_DATA_TYPE_VECTOR_BINARY64 as u32 => DataType::VectorBinary64,
            x if x == ZVEC_DATA_TYPE_VECTOR_FP16 as u32 => DataType::VectorFp16,
            x if x == ZVEC_DATA_TYPE_VECTOR_FP32 as u32 => DataType::VectorFp32,
            x if x == ZVEC_DATA_TYPE_VECTOR_FP64 as u32 => DataType::VectorFp64,
            x if x == ZVEC_DATA_TYPE_VECTOR_INT4 as u32 => DataType::VectorInt4,
            x if x == ZVEC_DATA_TYPE_VECTOR_INT8 as u32 => DataType::VectorInt8,
            x if x == ZVEC_DATA_TYPE_VECTOR_INT16 as u32 => DataType::VectorInt16,
            x if x == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16 as u32 => DataType::SparseVectorFp16,
            x if x == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 as u32 => DataType::SparseVectorFp32,
            x if x == ZVEC_DATA_TYPE_ARRAY_BINARY as u32 => DataType::ArrayBinary,
            x if x == ZVEC_DATA_TYPE_ARRAY_STRING as u32 => DataType::ArrayString,
            x if x == ZVEC_DATA_TYPE_ARRAY_BOOL as u32 => DataType::ArrayBool,
            x if x == ZVEC_DATA_TYPE_ARRAY_INT32 as u32 => DataType::ArrayInt32,
            x if x == ZVEC_DATA_TYPE_ARRAY_INT64 as u32 => DataType::ArrayInt64,
            x if x == ZVEC_DATA_TYPE_ARRAY_UINT32 as u32 => DataType::ArrayUInt32,
            x if x == ZVEC_DATA_TYPE_ARRAY_UINT64 as u32 => DataType::ArrayUInt64,
            x if x == ZVEC_DATA_TYPE_ARRAY_FLOAT as u32 => DataType::ArrayFloat,
            x if x == ZVEC_DATA_TYPE_ARRAY_DOUBLE as u32 => DataType::ArrayDouble,
            _ => DataType::Undefined,
        }
    }
}

impl From<DataType> for zvec_data_type_t {
    fn from(t: DataType) -> Self {
        match t {
            DataType::Undefined => ZVEC_DATA_TYPE_UNDEFINED as u32,
            DataType::Binary => ZVEC_DATA_TYPE_BINARY as u32,
            DataType::String => ZVEC_DATA_TYPE_STRING as u32,
            DataType::Bool => ZVEC_DATA_TYPE_BOOL as u32,
            DataType::Int32 => ZVEC_DATA_TYPE_INT32 as u32,
            DataType::Int64 => ZVEC_DATA_TYPE_INT64 as u32,
            DataType::UInt32 => ZVEC_DATA_TYPE_UINT32 as u32,
            DataType::UInt64 => ZVEC_DATA_TYPE_UINT64 as u32,
            DataType::Float => ZVEC_DATA_TYPE_FLOAT as u32,
            DataType::Double => ZVEC_DATA_TYPE_DOUBLE as u32,
            DataType::VectorBinary32 => ZVEC_DATA_TYPE_VECTOR_BINARY32 as u32,
            DataType::VectorBinary64 => ZVEC_DATA_TYPE_VECTOR_BINARY64 as u32,
            DataType::VectorFp16 => ZVEC_DATA_TYPE_VECTOR_FP16 as u32,
            DataType::VectorFp32 => ZVEC_DATA_TYPE_VECTOR_FP32 as u32,
            DataType::VectorFp64 => ZVEC_DATA_TYPE_VECTOR_FP64 as u32,
            DataType::VectorInt4 => ZVEC_DATA_TYPE_VECTOR_INT4 as u32,
            DataType::VectorInt8 => ZVEC_DATA_TYPE_VECTOR_INT8 as u32,
            DataType::VectorInt16 => ZVEC_DATA_TYPE_VECTOR_INT16 as u32,
            DataType::SparseVectorFp16 => ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16 as u32,
            DataType::SparseVectorFp32 => ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 as u32,
            DataType::ArrayBinary => ZVEC_DATA_TYPE_ARRAY_BINARY as u32,
            DataType::ArrayString => ZVEC_DATA_TYPE_ARRAY_STRING as u32,
            DataType::ArrayBool => ZVEC_DATA_TYPE_ARRAY_BOOL as u32,
            DataType::ArrayInt32 => ZVEC_DATA_TYPE_ARRAY_INT32 as u32,
            DataType::ArrayInt64 => ZVEC_DATA_TYPE_ARRAY_INT64 as u32,
            DataType::ArrayUInt32 => ZVEC_DATA_TYPE_ARRAY_UINT32 as u32,
            DataType::ArrayUInt64 => ZVEC_DATA_TYPE_ARRAY_UINT64 as u32,
            DataType::ArrayFloat => ZVEC_DATA_TYPE_ARRAY_FLOAT as u32,
            DataType::ArrayDouble => ZVEC_DATA_TYPE_ARRAY_DOUBLE as u32,
        }
    }
}

impl DataType {
    pub fn is_vector(&self) -> bool {
        matches!(
            self,
            DataType::VectorBinary32
                | DataType::VectorBinary64
                | DataType::VectorFp16
                | DataType::VectorFp32
                | DataType::VectorFp64
                | DataType::VectorInt4
                | DataType::VectorInt8
                | DataType::VectorInt16
                | DataType::SparseVectorFp16
                | DataType::SparseVectorFp32
        )
    }

    pub fn is_sparse_vector(&self) -> bool {
        matches!(
            self,
            DataType::SparseVectorFp16 | DataType::SparseVectorFp32
        )
    }

    pub fn is_dense_vector(&self) -> bool {
        matches!(
            self,
            DataType::VectorBinary32
                | DataType::VectorBinary64
                | DataType::VectorFp16
                | DataType::VectorFp32
                | DataType::VectorFp64
                | DataType::VectorInt4
                | DataType::VectorInt8
                | DataType::VectorInt16
        )
    }

    pub fn is_array(&self) -> bool {
        matches!(
            self,
            DataType::ArrayBinary
                | DataType::ArrayString
                | DataType::ArrayBool
                | DataType::ArrayInt32
                | DataType::ArrayInt64
                | DataType::ArrayUInt32
                | DataType::ArrayUInt64
                | DataType::ArrayFloat
                | DataType::ArrayDouble
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum IndexType {
    Undefined = 0,
    Hnsw = 1,
    Ivf = 2,
    Flat = 3,
    Invert = 10,
    Fts = 11,
}

impl From<zvec_index_type_t> for IndexType {
    fn from(t: zvec_index_type_t) -> Self {
        match t {
            x if x == ZVEC_INDEX_TYPE_UNDEFINED as u32 => IndexType::Undefined,
            x if x == ZVEC_INDEX_TYPE_HNSW as u32 => IndexType::Hnsw,
            x if x == ZVEC_INDEX_TYPE_IVF as u32 => IndexType::Ivf,
            x if x == ZVEC_INDEX_TYPE_FLAT as u32 => IndexType::Flat,
            x if x == ZVEC_INDEX_TYPE_INVERT as u32 => IndexType::Invert,
            x if x == ZVEC_INDEX_TYPE_FTS as u32 => IndexType::Fts,
            _ => IndexType::Undefined,
        }
    }
}

impl From<IndexType> for zvec_index_type_t {
    fn from(t: IndexType) -> Self {
        match t {
            IndexType::Undefined => ZVEC_INDEX_TYPE_UNDEFINED as u32,
            IndexType::Hnsw => ZVEC_INDEX_TYPE_HNSW as u32,
            IndexType::Ivf => ZVEC_INDEX_TYPE_IVF as u32,
            IndexType::Flat => ZVEC_INDEX_TYPE_FLAT as u32,
            IndexType::Invert => ZVEC_INDEX_TYPE_INVERT as u32,
            IndexType::Fts => ZVEC_INDEX_TYPE_FTS as u32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MetricType {
    Undefined = 0,
    L2 = 1,
    Ip = 2,
    Cosine = 3,
    MipsL2 = 4,
}

impl From<zvec_metric_type_t> for MetricType {
    fn from(t: zvec_metric_type_t) -> Self {
        match t {
            x if x == ZVEC_METRIC_TYPE_UNDEFINED as u32 => MetricType::Undefined,
            x if x == ZVEC_METRIC_TYPE_L2 as u32 => MetricType::L2,
            x if x == ZVEC_METRIC_TYPE_IP as u32 => MetricType::Ip,
            x if x == ZVEC_METRIC_TYPE_COSINE as u32 => MetricType::Cosine,
            x if x == ZVEC_METRIC_TYPE_MIPSL2 as u32 => MetricType::MipsL2,
            _ => MetricType::Undefined,
        }
    }
}

impl From<MetricType> for zvec_metric_type_t {
    fn from(t: MetricType) -> Self {
        match t {
            MetricType::Undefined => ZVEC_METRIC_TYPE_UNDEFINED as u32,
            MetricType::L2 => ZVEC_METRIC_TYPE_L2 as u32,
            MetricType::Ip => ZVEC_METRIC_TYPE_IP as u32,
            MetricType::Cosine => ZVEC_METRIC_TYPE_COSINE as u32,
            MetricType::MipsL2 => ZVEC_METRIC_TYPE_MIPSL2 as u32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum QuantizeType {
    Undefined = 0,
    Fp16 = 1,
    Int8 = 2,
    Int4 = 3,
}

impl From<zvec_quantize_type_t> for QuantizeType {
    fn from(t: zvec_quantize_type_t) -> Self {
        match t {
            x if x == ZVEC_QUANTIZE_TYPE_UNDEFINED as u32 => QuantizeType::Undefined,
            x if x == ZVEC_QUANTIZE_TYPE_FP16 as u32 => QuantizeType::Fp16,
            x if x == ZVEC_QUANTIZE_TYPE_INT8 as u32 => QuantizeType::Int8,
            x if x == ZVEC_QUANTIZE_TYPE_INT4 as u32 => QuantizeType::Int4,
            _ => QuantizeType::Undefined,
        }
    }
}

impl From<QuantizeType> for zvec_quantize_type_t {
    fn from(t: QuantizeType) -> Self {
        match t {
            QuantizeType::Undefined => ZVEC_QUANTIZE_TYPE_UNDEFINED as u32,
            QuantizeType::Fp16 => ZVEC_QUANTIZE_TYPE_FP16 as u32,
            QuantizeType::Int8 => ZVEC_QUANTIZE_TYPE_INT8 as u32,
            QuantizeType::Int4 => ZVEC_QUANTIZE_TYPE_INT4 as u32,
        }
    }
}
