namespace py infinity_thrift_rpc
namespace cpp infinity_thrift_rpc

// https://github.com/apache/thrift/blob/master/test/Recursive.thrift

enum LogicType{
Boolean,
TinyInt,
SmallInt,
Integer,
BigInt,
HugeInt,
Decimal,
Float,
Double,
Float16,
BFloat16,
Varchar,
Embedding,
Tensor,
TensorArray,
Sparse,
MultiVector,
Date,
Time,
DateTime,
Timestamp,
Interval,
Array,
Invalid
}

enum CreateConflict {
Ignore,
Error,
Replace,
}

enum DropConflict {
Ignore,
Error,
}

struct Property {
1:  string key,
2:  string value
}

struct CreateOption {
1:  CreateConflict conflict_type,
2:  list<Property> properties = [],
}

struct DropOption {
1:  DropConflict conflict_type,
}

struct NumberType {}

struct VarcharType {}

enum ElementType {
ElementBit,
ElementUInt8,
ElementInt8,
ElementInt16,
ElementInt32,
ElementInt64,
ElementFloat32,
ElementFloat64,
ElementFloat16,
ElementBFloat16,
}

struct EmbeddingType {
1:  i32 dimension,
2:  ElementType element_type,
}

struct SparseType {
1:  i64 dimension,
2:  ElementType element_type,
3:  ElementType index_type,
}

struct ArrayType {
1:  DataType & element_data_type,
}

union PhysicalType {
1:  NumberType number_type,
2:  VarcharType varchar_type,
3:  EmbeddingType embedding_type,
4:  SparseType sparse_type,
5:  ArrayType array_type,
}

struct DataType {
1:  LogicType logic_type,
2:  PhysicalType physical_type,
}

enum Constraint {
PrimaryKey,
NotNull,
Null,
Unique,
}

enum LiteralType {
Boolean,
Double,
String,
Int64,
Null,
IntegerArray,
DoubleArray,
IntegerTensor,
DoubleTensor,
IntegerTensorArray,
DoubleTensorArray,
SparseIntegerArray,
SparseDoubleArray,
Date,
Time,
Inteval,
DateTime,
Timestamp,
CurlyBracketsArray,
}

union ParsedExprType {
1: ConstantExpr & constant_expr,
2: ColumnExpr & column_expr,
3: FunctionExpr & function_expr,
4: BetweenExpr & between_expr,
5: KnnExpr  & knn_expr,
6: MatchSparseExpr & match_sparse_expr;
7: MatchTensorExpr & match_tensor_expr;
8: MatchExpr  & match_expr,
9: FusionExpr & fusion_expr,
10: SearchExpr & search_expr,
11 : InExpr & in_expr,
}

struct ParsedExpr {
1: ParsedExprType type,
2: string alias_name,
}

struct ColumnExpr {
1: list<string> column_name = [],
2: bool star,
}

enum KnnDistanceType{
L2,
Cosine,
InnerProduct,
Hamming,
}

union EmbeddingData {
1: list<bool> bool_array_value,
2: list<i16> u8_array_value,
3: list<i16> i8_array_value,
4: list<i16> i16_array_value,
5: list<i32> i32_array_value,
6: list<i64> i64_array_value,
7: list<double> f32_array_value,
8: list<double> f64_array_value,
9: list<double> f16_array_value,
10: list<double> bf16_array_value,
}

struct InitParameter {
1: string param_name,
2: string param_value,
}

struct ConstantExpr {
1: LiteralType literal_type,
2: optional bool bool_value,
3: optional i64 i64_value,
4: optional double f64_value,
5: optional string str_value,
6: optional list<i64> i64_array_value,
7: optional list<double> f64_array_value,
8: optional list<list<i64>> i64_tensor_value,
9: optional list<list<double>> f64_tensor_value,
10: optional list<list<list<i64>>> i64_tensor_array_value,
11: optional list<list<list<double>>> f64_tensor_array_value,
12: optional list<i64> i64_array_idx,
13: optional list<ConstantExpr> curly_brackets_array,
}

struct KnnExpr {
1: ColumnExpr  column_expr,
2: EmbeddingData embedding_data,
3: ElementType embedding_data_type,
4: KnnDistanceType distance_type,
5: i64 topn,
6: list<InitParameter> opt_params = [],
7: optional ParsedExpr filter_expr,
}

struct MatchSparseExpr {
1: ColumnExpr  column_expr,
2: ConstantExpr  query_sparse_expr,
3: string  metric_type,
4: i64  topn,
5: list<InitParameter> opt_params = [],
6: optional ParsedExpr filter_expr,
}

struct MatchTensorExpr {
1: string search_method,
2: ColumnExpr column_expr,
3: ElementType embedding_data_type,
4: EmbeddingData embedding_data,
5: string extra_options,
6: optional ParsedExpr filter_expr,
}

struct MatchExpr {
	1: string fields,
	2: string matching_text,
	3: string options_text,
    4: optional ParsedExpr filter_expr,
}

union GenericMatchExpr {
	1: KnnExpr & match_vector_expr;
	2: MatchSparseExpr & match_sparse_expr;
	3: MatchTensorExpr & match_tensor_expr;
	4: MatchExpr & match_text_expr;
}

struct FusionExpr {
	1: string method,
	2: string options_text,
	3: optional MatchTensorExpr optional_match_tensor_expr,
}

struct SearchExpr {
	1: optional list<GenericMatchExpr> match_exprs,
	2: optional list<FusionExpr> fusion_exprs,
}

struct FunctionExpr {
1: string function_name,
2: list<ParsedExpr> arguments,
}


struct BetweenExpr {
1: ParsedExpr value,
2: ParsedExpr upper_bound,
3: ParsedExpr lower_bound,
}

struct UpdateExpr {
1: string column_name,
2: ParsedExpr value,
}

struct OrderByExpr {
1: ParsedExpr expr,
2: bool asc,
}

struct InExpr {
1: ParsedExpr left_operand,
2: list<ParsedExpr> arguments,
3: bool in_type,
}

struct ColumnDef {
1:  i32 id,
2:  string name,
3:  DataType data_type,
4:  list<Constraint> constraints = [],
5:  ConstantExpr constant_expr,
6:  string comment,
}


struct Field {
1: list<string> column_names = [],
2: list<ParsedExpr> parse_exprs = [],
}

enum CopyFileType {
CSV,
JSON,
JSONL,
FVECS,
CSR,
BVECS,
}

enum ColumnType {
ColumnBool,
ColumnInt8,
ColumnInt16,
ColumnInt32,
ColumnInt64,
ColumnFloat32,
ColumnFloat64,
ColumnFloat16,
ColumnBFloat16,
ColumnVarchar,
ColumnEmbedding,
ColumnTensor,
ColumnTensorArray,
ColumnSparse,
ColumnMultiVector,
ColumnRowID,
ColumnDate,
ColumnTime,
ColumnDateTime,
ColumnTimestamp,
ColumnInterval,
ColumnArray,
ColumnInvalid,
}

struct ColumnField {
1: ColumnType column_type,
2: list<binary> column_vectors = [],
3: string column_name,
}

struct ImportOption {
1:  string delimiter,
2:  bool has_header,
3:  CopyFileType copy_file_type,
}

struct ExportOption {
1:  string delimiter,
2:  bool has_header,
3:  CopyFileType copy_file_type,
4:  i64 offset,
5:  i64 limit,
6:  i64 row_limit,
}

struct OptimizeOptions {
1:  string index_name,
2:  list<InitParameter> opt_params = []
}

struct ConnectRequest {
1: i64 client_version,
}

struct CommonRequest {
1:  i64 session_id,
}

struct CommonResponse {
1:  i64 error_code,
2:  string error_msg,
3:  i64 session_id,
}

struct ListDatabaseRequest {
1: i64 session_id,
}

struct ListDatabaseResponse {
1: i64 error_code,
2: string error_msg,
3: list<string> db_names = [],
4: list<string> db_dirs = [],
5: list<string> db_comments = [],
}

struct ListTableRequest {
1: string db_name,
2: i64 session_id,
}

struct ListTableResponse {
1: i64 error_code,
2: string error_msg,
3: list<string> table_names = [],
}

struct ListIndexRequest {
1: string db_name,
2: string table_name,
3: i64 session_id,
}

struct ListIndexResponse {
1: i64 error_code,
2: string error_msg,
3: list<string> index_names = [],
}

struct ShowDatabaseRequest {
1: string db_name,
2: i64 session_id,
}

struct ShowDatabaseResponse {
1: i64 error_code,
2: string error_msg,
3: string database_name,
4: string store_dir,
5: i64 table_count,
6: string comment
}

struct ShowTableRequest {
1: string db_name,
2: string table_name,
3: i64 session_id,
}

struct ShowTableResponse {
1: i64 error_code,
2: string error_msg,
3: string database_name,
4: string table_name,
5: string store_dir,
6: i64 column_count,
7: i64 segment_count,
8: i64 row_count,
}

struct ShowColumnsRequest {
1: string db_name,
2: string table_name,
3: i64 session_id,
}

struct GetTableRequest {
1: string db_name,
2: string table_name,
3: i64 session_id,
}

enum IndexType {
IVF,
Hnsw,
FullText,
BMP,
Secondary,
EMVB,
DiskAnn,
}

struct IndexInfo {
1: string column_name,
2: IndexType index_type,
3: list<InitParameter> index_param_list = [],
}

struct CreateIndexRequest {
1: string db_name,
2: string table_name,
3: string index_name,
4: string index_comment,
5: IndexInfo index_info,
6: i64 session_id,
7: CreateOption create_option,
}

struct DropIndexRequest {
1: string db_name,
2: string table_name,
3: string index_name,
4: i64 session_id,
5: DropOption drop_option,
}

struct ShowIndexRequest {
1: string db_name,
2: string table_name,
3: string index_name,
4: i64 session_id,
}

struct ShowIndexResponse {
1: i64 error_code,
2: string error_msg,
3: string db_name,
4: string table_name,
5: string index_name,
6: string index_comment,
7: string index_type,
8: string index_column_names,
9: string index_column_ids,
10: string other_parameters,
11: string store_dir,
12: string segment_index_count,
}

struct OptimizeRequest {
1: string db_name,
2: string table_name,
3: OptimizeOptions optimize_options,
4: i64 session_id,
}

struct GetDatabaseRequest {
1: string db_name,
2: i64 session_id,
}

struct CreateDatabaseRequest {
1: string db_name,
2: i64 session_id,
3: CreateOption create_option,
4: string db_comment,
}

struct DropDatabaseRequest {
1:  string db_name,
2:  i64 session_id,
3:  DropOption drop_option,
}

struct CreateTableRequest {
1:  string db_name,
2:  string table_name,
3:  list<ColumnDef> column_defs = [],
6:  i64 session_id,
7:  CreateOption create_option,
}

struct DropTableRequest {
1:  string db_name,
2:  string table_name,
3:  i64 session_id,
4:  DropOption drop_option,
}

struct InsertRequest {
1:  string db_name,
2:  string table_name,
3:  list<Field> fields = [],
4:  i64 session_id,
}

struct ImportRequest{
1:  string db_name,
2:  string table_name,
3:  string file_name,
4:  ImportOption import_option,
5:  i64 session_id,
}

struct ExportRequest{
1:  string db_name,
2:  string table_name,
3:  list<string> columns,
4:  string file_name,
5:  ExportOption export_option,
6:  i64 session_id,
}

enum ExplainType {
Analyze,
Ast,
UnOpt,
Opt,
Physical,
Pipeline,
Fragment,
}

struct ExplainRequest {
1:  i64 session_id,
2:  string db_name,
3:  string table_name,
4:  list<ParsedExpr> select_list = [],
5:  optional list<ParsedExpr> highlight_list = [],
6:  optional SearchExpr search_expr,
7:  optional ParsedExpr where_expr,
8:  optional list<ParsedExpr> group_by_list = [],
9:  optional ParsedExpr having_expr,
10:  optional ParsedExpr limit_expr,
11:  optional ParsedExpr offset_expr,
12:  optional list<OrderByExpr> order_by_list = [],
13:  ExplainType explain_type,
}

struct ExplainResponse {
1: i64 error_code,
2: string error_msg,
3: list<ColumnDef> column_defs = [],
4: list<ColumnField> column_fields = [];
}

struct SelectRequest {
1: i64 session_id,
2: string db_name,
3: string table_name,
4: list<ParsedExpr> select_list = [],
5: optional list<ParsedExpr> highlight_list = [],
6: optional SearchExpr search_expr,
7: optional ParsedExpr where_expr,
8: optional list<ParsedExpr> group_by_list = [],
9: optional ParsedExpr having_expr,
10: optional ParsedExpr limit_expr,
11: optional ParsedExpr offset_expr,
12: optional list<OrderByExpr> order_by_list = [],
13: optional bool total_hits_count,
}

struct SelectResponse {
1: i64 error_code,
2: string error_msg,
3: list<ColumnDef> column_defs = [],
4: list<ColumnField> column_fields = [];
5: string extra_result;
}

struct DeleteRequest {
1:  string db_name,
2:  string table_name,
3:  ParsedExpr where_expr,
4:  i64 session_id,
}

struct DeleteResponse {
1:  i64 error_code,
2:  string error_msg,
3:  i64 deleted_rows,
}

struct UpdateRequest {
1:  string db_name,
2:  string table_name,
3:  ParsedExpr where_expr,
4:  list<UpdateExpr> update_expr_array = [],
5:  i64 session_id,
}

struct AddColumnsRequest {
1:  string db_name,
2:  string table_name,
3:  list<ColumnDef> column_defs = [],
4:  i64 session_id,
}

struct DropColumnsRequest {
1:  string db_name,
2:  string table_name,
3:  list<string> column_names = [],
4:  i64 session_id,
}

struct DumpIndexRequest {
1:  string db_name,
2:  string table_name,
3:  string index_name,
4:  i64 session_id,
}

struct ShowTablesRequest{
1: i64 session_id,
2: string db_name,
}

struct ShowSegmentsRequest {
1: i64 session_id,
2: string db_name,
3: string table_name,
}

struct ShowSegmentRequest {
1: i64 session_id,
2: string db_name,
3: string table_name,
4: i64 segment_id,
}

struct ShowSegmentResponse {
1: i64 error_code,
2: string error_msg,
3: i64 segment_id,
4: string status,
5: string path,
6: string size,
7: i64 block_count,
8: i64 row_capacity,
9: i64 row_count,
10: i64 room,
11: i64 column_count,
}

struct ShowBlocksRequest {
1: i64 session_id,
2: string db_name,
3: string table_name,
4: i64 segment_id,
}

struct ShowBlockRequest {
1: i64 session_id,
2: string db_name,
3: string table_name,
4: i64 segment_id,
5: i64 block_id,
}

struct ShowBlockResponse {
1: i64 error_code,
2: string error_msg,
3: i64 block_id,
4: string path,
5: string size,
6: i64 row_capacity,
7: i64 row_count,
8: i64 column_count,
}

struct ShowBlockColumnRequest {
1: i64 session_id,
2: string db_name,
3: string table_name,
4: i64 segment_id,
5: i64 block_id,
6: i64 column_id,
}

struct ShowBlockColumnResponse {
1: i64 error_code,
2: string error_msg,
3: string column_name,
4: i64 column_id,
5: string data_type,
6: string path,
7: i64 extra_file_count,
8: string extra_file_names,
}

struct ShowCurrentNodeRequest {
1: i64 session_id
}

struct ShowCurrentNodeResponse {
1: i64 error_code,
2: string error_msg,
3: string node_role,
4: string server_status
}

struct CommandRequest {
1: i64 session_id
2: string command_type,
3: string test_command_content
}

struct FlushRequest {
1: i64 session_id
2: string flush_type,
}

struct CompactRequest {
1: i64 session_id
2: string db_name,
3: string table_name,
}

struct CreateTableSnapshotRequest {
1: i64 session_id
2: string db_name,
3: string table_name,
4: string snapshot_name,
}

struct CreateDatabaseSnapshotRequest {
1: i64 session_id
2: string db_name,
3: string snapshot_name,
}

struct CreateSystemSnapshotRequest {
1: i64 session_id
2: string snapshot_name,
}

struct RestoreSnapshotRequest {
1: i64 session_id
2: string snapshot_name,
3: string scope,
}

struct SnapshotInfo {
1:  string name,
2:  string scope,
3:  string time,
4:  i64 commit,
5:  string size,
}

struct ShowSnapshotRequest {
1: i64 session_id,
2: string snapshot_name,
}

struct ShowSnapshotResponse {
1: i64 error_code,
2: string error_msg,
3: SnapshotInfo snapshot,
}

struct ListSnapshotsRequest {
1: i64 session_id,
}

struct ListSnapshotsResponse {
1: i64 error_code,
2: string error_msg,
3: list<SnapshotInfo> snapshots = [],
}

struct DropSnapshotRequest {
1: i64 session_id,
2: string snapshot_name,
}

// Service
service InfinityService {
CommonResponse Connect(1:ConnectRequest request),
CommonResponse Disconnect(1:CommonRequest request),

CommonResponse CreateDatabase(1:CreateDatabaseRequest request),
CommonResponse DropDatabase(1:DropDatabaseRequest request),
CommonResponse CreateTable(1:CreateTableRequest request),
CommonResponse DropTable(1:DropTableRequest request),
CommonResponse Insert(1:InsertRequest request),
CommonResponse Import(1:ImportRequest request),
CommonResponse Export(1:ExportRequest request),
SelectResponse Select(1:SelectRequest request),
SelectResponse Explain(1:ExplainRequest request),
DeleteResponse Delete(1:DeleteRequest request),
CommonResponse Update(1:UpdateRequest request),

ListDatabaseResponse ListDatabase(1:ListDatabaseRequest request),
ListTableResponse ListTable(1:ListTableRequest request),
ListIndexResponse ListIndex(1:ListIndexRequest request),

ShowTableResponse ShowTable(1:ShowTableRequest request),
SelectResponse ShowColumns(1:ShowColumnsRequest request),
ShowDatabaseResponse ShowDatabase(1:ShowDatabaseRequest request),
SelectResponse ShowTables(1:ShowTablesRequest request),

SelectResponse ShowSegments(1:ShowSegmentsRequest request),
ShowSegmentResponse ShowSegment(1:ShowSegmentRequest request),

SelectResponse ShowBlocks(1:ShowBlocksRequest request),
ShowBlockResponse ShowBlock(1:ShowBlockRequest request),

ShowBlockColumnResponse ShowBlockColumn(1:ShowBlockColumnRequest request),
ShowCurrentNodeResponse ShowCurrentNode(1:ShowCurrentNodeRequest request),

CommonResponse GetDatabase(1:GetDatabaseRequest request),
CommonResponse GetTable(1:GetTableRequest request),

CommonResponse CreateIndex(1:CreateIndexRequest request),
CommonResponse DropIndex(1:DropIndexRequest request),
ShowIndexResponse ShowIndex(1:ShowIndexRequest request),

CommonResponse Optimize(1:OptimizeRequest request),

CommonResponse AddColumns(1:AddColumnsRequest request),
CommonResponse DropColumns(1:DropColumnsRequest request),

CommonResponse Cleanup(1:CommonRequest request),
CommonResponse DumpIndex(1:DumpIndexRequest request),

CommonResponse Command(1: CommandRequest request),

CommonResponse Flush(1: FlushRequest request),

CommonResponse Compact(1: CompactRequest request),

CommonResponse CreateTableSnapshot(1: CreateTableSnapshotRequest request),
CommonResponse CreateDatabaseSnapshot(1: CreateDatabaseSnapshotRequest request),
CommonResponse CreateSystemSnapshot(1: CreateSystemSnapshotRequest request),
CommonResponse RestoreSnapshot(1: RestoreSnapshotRequest request),

ShowSnapshotResponse ShowSnapshot(1: ShowSnapshotRequest request),
ListSnapshotsResponse ListSnapshots(1: ListSnapshotsRequest request),
CommonResponse DropSnapshot(1: DropSnapshotRequest request),

}
