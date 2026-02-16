#![allow(non_upper_case_globals)]

use crate::ffi::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Fatal = 4,
}

impl LogLevel {
    pub const Warning: Self = Self::Warn;
}

impl Default for LogLevel {
    fn default() -> Self {
        Self::Warn
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum LogType {
    Console = 0,
    File = 1,
}

impl Default for LogType {
    fn default() -> Self {
        Self::Console
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

impl From<zvec_data_type> for DataType {
    fn from(t: zvec_data_type) -> Self {
        match t {
            zvec_data_type_ZVEC_DATA_TYPE_UNDEFINED => DataType::Undefined,
            zvec_data_type_ZVEC_DATA_TYPE_BINARY => DataType::Binary,
            zvec_data_type_ZVEC_DATA_TYPE_STRING => DataType::String,
            zvec_data_type_ZVEC_DATA_TYPE_BOOL => DataType::Bool,
            zvec_data_type_ZVEC_DATA_TYPE_INT32 => DataType::Int32,
            zvec_data_type_ZVEC_DATA_TYPE_INT64 => DataType::Int64,
            zvec_data_type_ZVEC_DATA_TYPE_UINT32 => DataType::UInt32,
            zvec_data_type_ZVEC_DATA_TYPE_UINT64 => DataType::UInt64,
            zvec_data_type_ZVEC_DATA_TYPE_FLOAT => DataType::Float,
            zvec_data_type_ZVEC_DATA_TYPE_DOUBLE => DataType::Double,
            zvec_data_type_ZVEC_DATA_TYPE_VECTOR_BINARY32 => DataType::VectorBinary32,
            zvec_data_type_ZVEC_DATA_TYPE_VECTOR_BINARY64 => DataType::VectorBinary64,
            zvec_data_type_ZVEC_DATA_TYPE_VECTOR_FP16 => DataType::VectorFp16,
            zvec_data_type_ZVEC_DATA_TYPE_VECTOR_FP32 => DataType::VectorFp32,
            zvec_data_type_ZVEC_DATA_TYPE_VECTOR_FP64 => DataType::VectorFp64,
            zvec_data_type_ZVEC_DATA_TYPE_VECTOR_INT4 => DataType::VectorInt4,
            zvec_data_type_ZVEC_DATA_TYPE_VECTOR_INT8 => DataType::VectorInt8,
            zvec_data_type_ZVEC_DATA_TYPE_VECTOR_INT16 => DataType::VectorInt16,
            zvec_data_type_ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16 => DataType::SparseVectorFp16,
            zvec_data_type_ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 => DataType::SparseVectorFp32,
            zvec_data_type_ZVEC_DATA_TYPE_ARRAY_BINARY => DataType::ArrayBinary,
            zvec_data_type_ZVEC_DATA_TYPE_ARRAY_STRING => DataType::ArrayString,
            zvec_data_type_ZVEC_DATA_TYPE_ARRAY_BOOL => DataType::ArrayBool,
            zvec_data_type_ZVEC_DATA_TYPE_ARRAY_INT32 => DataType::ArrayInt32,
            zvec_data_type_ZVEC_DATA_TYPE_ARRAY_INT64 => DataType::ArrayInt64,
            zvec_data_type_ZVEC_DATA_TYPE_ARRAY_UINT32 => DataType::ArrayUInt32,
            zvec_data_type_ZVEC_DATA_TYPE_ARRAY_UINT64 => DataType::ArrayUInt64,
            zvec_data_type_ZVEC_DATA_TYPE_ARRAY_FLOAT => DataType::ArrayFloat,
            zvec_data_type_ZVEC_DATA_TYPE_ARRAY_DOUBLE => DataType::ArrayDouble,
            _ => DataType::Undefined,
        }
    }
}

impl From<DataType> for zvec_data_type {
    fn from(t: DataType) -> Self {
        match t {
            DataType::Undefined => zvec_data_type_ZVEC_DATA_TYPE_UNDEFINED,
            DataType::Binary => zvec_data_type_ZVEC_DATA_TYPE_BINARY,
            DataType::String => zvec_data_type_ZVEC_DATA_TYPE_STRING,
            DataType::Bool => zvec_data_type_ZVEC_DATA_TYPE_BOOL,
            DataType::Int32 => zvec_data_type_ZVEC_DATA_TYPE_INT32,
            DataType::Int64 => zvec_data_type_ZVEC_DATA_TYPE_INT64,
            DataType::UInt32 => zvec_data_type_ZVEC_DATA_TYPE_UINT32,
            DataType::UInt64 => zvec_data_type_ZVEC_DATA_TYPE_UINT64,
            DataType::Float => zvec_data_type_ZVEC_DATA_TYPE_FLOAT,
            DataType::Double => zvec_data_type_ZVEC_DATA_TYPE_DOUBLE,
            DataType::VectorBinary32 => zvec_data_type_ZVEC_DATA_TYPE_VECTOR_BINARY32,
            DataType::VectorBinary64 => zvec_data_type_ZVEC_DATA_TYPE_VECTOR_BINARY64,
            DataType::VectorFp16 => zvec_data_type_ZVEC_DATA_TYPE_VECTOR_FP16,
            DataType::VectorFp32 => zvec_data_type_ZVEC_DATA_TYPE_VECTOR_FP32,
            DataType::VectorFp64 => zvec_data_type_ZVEC_DATA_TYPE_VECTOR_FP64,
            DataType::VectorInt4 => zvec_data_type_ZVEC_DATA_TYPE_VECTOR_INT4,
            DataType::VectorInt8 => zvec_data_type_ZVEC_DATA_TYPE_VECTOR_INT8,
            DataType::VectorInt16 => zvec_data_type_ZVEC_DATA_TYPE_VECTOR_INT16,
            DataType::SparseVectorFp16 => zvec_data_type_ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16,
            DataType::SparseVectorFp32 => zvec_data_type_ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32,
            DataType::ArrayBinary => zvec_data_type_ZVEC_DATA_TYPE_ARRAY_BINARY,
            DataType::ArrayString => zvec_data_type_ZVEC_DATA_TYPE_ARRAY_STRING,
            DataType::ArrayBool => zvec_data_type_ZVEC_DATA_TYPE_ARRAY_BOOL,
            DataType::ArrayInt32 => zvec_data_type_ZVEC_DATA_TYPE_ARRAY_INT32,
            DataType::ArrayInt64 => zvec_data_type_ZVEC_DATA_TYPE_ARRAY_INT64,
            DataType::ArrayUInt32 => zvec_data_type_ZVEC_DATA_TYPE_ARRAY_UINT32,
            DataType::ArrayUInt64 => zvec_data_type_ZVEC_DATA_TYPE_ARRAY_UINT64,
            DataType::ArrayFloat => zvec_data_type_ZVEC_DATA_TYPE_ARRAY_FLOAT,
            DataType::ArrayDouble => zvec_data_type_ZVEC_DATA_TYPE_ARRAY_DOUBLE,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum IndexType {
    Undefined = 0,
    Hnsw = 1,
    Ivf = 3,
    Flat = 4,
    Invert = 10,
}

impl From<zvec_index_type> for IndexType {
    fn from(t: zvec_index_type) -> Self {
        match t {
            zvec_index_type_ZVEC_INDEX_TYPE_UNDEFINED => IndexType::Undefined,
            zvec_index_type_ZVEC_INDEX_TYPE_HNSW => IndexType::Hnsw,
            zvec_index_type_ZVEC_INDEX_TYPE_IVF => IndexType::Ivf,
            zvec_index_type_ZVEC_INDEX_TYPE_FLAT => IndexType::Flat,
            zvec_index_type_ZVEC_INDEX_TYPE_INVERT => IndexType::Invert,
            _ => IndexType::Undefined,
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

impl From<zvec_metric_type> for MetricType {
    fn from(t: zvec_metric_type) -> Self {
        match t {
            zvec_metric_type_ZVEC_METRIC_TYPE_UNDEFINED => MetricType::Undefined,
            zvec_metric_type_ZVEC_METRIC_TYPE_L2 => MetricType::L2,
            zvec_metric_type_ZVEC_METRIC_TYPE_IP => MetricType::Ip,
            zvec_metric_type_ZVEC_METRIC_TYPE_COSINE => MetricType::Cosine,
            zvec_metric_type_ZVEC_METRIC_TYPE_MIPS_L2 => MetricType::MipsL2,
            _ => MetricType::Undefined,
        }
    }
}

impl From<MetricType> for zvec_metric_type {
    fn from(t: MetricType) -> Self {
        match t {
            MetricType::Undefined => zvec_metric_type_ZVEC_METRIC_TYPE_UNDEFINED,
            MetricType::L2 => zvec_metric_type_ZVEC_METRIC_TYPE_L2,
            MetricType::Ip => zvec_metric_type_ZVEC_METRIC_TYPE_IP,
            MetricType::Cosine => zvec_metric_type_ZVEC_METRIC_TYPE_COSINE,
            MetricType::MipsL2 => zvec_metric_type_ZVEC_METRIC_TYPE_MIPS_L2,
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

impl From<zvec_quantize_type> for QuantizeType {
    fn from(t: zvec_quantize_type) -> Self {
        match t {
            zvec_quantize_type_ZVEC_QUANTIZE_TYPE_UNDEFINED => QuantizeType::Undefined,
            zvec_quantize_type_ZVEC_QUANTIZE_TYPE_FP16 => QuantizeType::Fp16,
            zvec_quantize_type_ZVEC_QUANTIZE_TYPE_INT8 => QuantizeType::Int8,
            zvec_quantize_type_ZVEC_QUANTIZE_TYPE_INT4 => QuantizeType::Int4,
            _ => QuantizeType::Undefined,
        }
    }
}

impl From<QuantizeType> for zvec_quantize_type {
    fn from(t: QuantizeType) -> Self {
        match t {
            QuantizeType::Undefined => zvec_quantize_type_ZVEC_QUANTIZE_TYPE_UNDEFINED,
            QuantizeType::Fp16 => zvec_quantize_type_ZVEC_QUANTIZE_TYPE_FP16,
            QuantizeType::Int8 => zvec_quantize_type_ZVEC_QUANTIZE_TYPE_INT8,
            QuantizeType::Int4 => zvec_quantize_type_ZVEC_QUANTIZE_TYPE_INT4,
        }
    }
}
