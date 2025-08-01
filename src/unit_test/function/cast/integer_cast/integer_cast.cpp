// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
import base_test;

import infinity_exception;

import global_resource_usage;
import third_party;

import logger;
import stl;
import infinity_context;

import function_set;
import aggregate_function_set;
import aggregate_function;
import function;
import column_expression;
import value;
import default_values;
import data_block;
import cast_table;
import column_vector;
import integer_cast;
import bound_cast_func;
import internal_types;
import logical_type;
import data_type;

using namespace infinity;
class IntegerCastTest : public BaseTest {};

TEST_F(IntegerCastTest, integer_cast0) {
    using namespace infinity;

    // Integer to Integer, throw exception
    {
        IntegerT source = 0;
        IntegerT target;
        EXPECT_THROW(IntegerTryCastToFixlen::Run(source, target), UnrecoverableException);
    }

    // IntegerT to TinyInt
    {
        IntegerT source = std::numeric_limits<IntegerT>::min();
        TinyIntT target;
        EXPECT_FALSE(IntegerTryCastToFixlen::Run(source, target));

        source = std::numeric_limits<IntegerT>::max();
        EXPECT_FALSE(IntegerTryCastToFixlen::Run(source, target));

        source = std::numeric_limits<TinyIntT>::max();
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target);

        source = std::numeric_limits<TinyIntT>::min();
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target);

        source = 0;
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target);
    }

    // Integer to SmallInt
    {
        IntegerT source = std::numeric_limits<IntegerT>::min();
        SmallIntT target;
        EXPECT_FALSE(IntegerTryCastToFixlen::Run(source, target));

        source = std::numeric_limits<IntegerT>::max();
        EXPECT_FALSE(IntegerTryCastToFixlen::Run(source, target));
        source = 0;
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target);

        source = std::numeric_limits<SmallIntT>::max();
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target);

        source = std::numeric_limits<SmallIntT>::min();
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target);
    }
    // Integer to BigInt
    {
        IntegerT source = std::numeric_limits<IntegerT>::min();
        BigIntT target;
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target);

        source = std::numeric_limits<IntegerT>::max();
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target);

        source = 0;
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target);
    }
    // IntegerT to HugeInt
    {
        IntegerT source = std::numeric_limits<IntegerT>::min();
        HugeIntT target;
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target.lower);
        EXPECT_EQ((static_cast<i32>(source) < 0) * -1, target.upper);

        source = std::numeric_limits<IntegerT>::max();
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target.lower);
        EXPECT_EQ((static_cast<i32>(source) < 0) * -1, target.upper);

        source = 0;
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_EQ(source, target.lower);
        EXPECT_EQ((static_cast<i32>(source) < 0) * -1, target.upper);
    }
    // IntegerT to Float
    {
        IntegerT source = std::numeric_limits<IntegerT>::min();
        FloatT target;
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_FLOAT_EQ(source, target);

        source = std::numeric_limits<IntegerT>::max();
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_FLOAT_EQ(source, target);

        source = 0;
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_FLOAT_EQ(source, target);
    }
    // IntegerT to Double
    {
        IntegerT source = std::numeric_limits<IntegerT>::min();
        DoubleT target;
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_FLOAT_EQ(source, target);

        source = std::numeric_limits<IntegerT>::max();
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_FLOAT_EQ(source, target);

        source = 0;
        EXPECT_TRUE(IntegerTryCastToFixlen::Run(source, target));
        EXPECT_FLOAT_EQ(source, target);
    }

    // TODO:
    // Cast to decimal
    {
        IntegerT source = std::numeric_limits<IntegerT>::lowest();
        DecimalT target;
        EXPECT_THROW(IntegerTryCastToFixlen::Run(source, target), UnrecoverableException);
    }

    // IntegerT to VarcharT
    {
        IntegerT source;
        VarcharT target;
        String src_str, tgt_str;

        SharedPtr<DataType> data_type = MakeShared<DataType>(LogicalType::kVarchar);
        SharedPtr<ColumnVector> col_varchar_ptr = MakeShared<ColumnVector>(data_type);
        col_varchar_ptr->Initialize();

        source = std::numeric_limits<IntegerT>::min();
        EXPECT_TRUE(IntegerTryCastToVarlen::Run(source, target, col_varchar_ptr.get()));

        src_str = std::to_string(source);
        EXPECT_EQ(src_str.size(), 11u);
        EXPECT_STREQ(src_str.c_str(), target.ToString().c_str());

        source = std::numeric_limits<IntegerT>::max();
        EXPECT_TRUE(IntegerTryCastToVarlen::Run(source, target, col_varchar_ptr.get()));
        src_str = std::to_string(source);
        EXPECT_EQ(src_str.size(), 10u);
        EXPECT_STREQ(src_str.c_str(), target.ToString().c_str());

        source = 0;
        EXPECT_TRUE(IntegerTryCastToVarlen::Run(source, target, col_varchar_ptr.get()));
        src_str = std::to_string(source);
        EXPECT_EQ(src_str.size(), 1u);
        EXPECT_STREQ(src_str.c_str(), target.ToString().c_str());

        source = 9;
        EXPECT_TRUE(IntegerTryCastToVarlen::Run(source, target, col_varchar_ptr.get()));
        src_str = std::to_string(source);
        EXPECT_EQ(src_str.size(), 1u);
        EXPECT_STREQ(src_str.c_str(), target.ToString().c_str());

        source = 10;
        EXPECT_TRUE(IntegerTryCastToVarlen::Run(source, target, col_varchar_ptr.get()));
        src_str = std::to_string(source);
        EXPECT_EQ(src_str.size(), 2u);
        EXPECT_STREQ(src_str.c_str(), target.ToString().c_str());

        source = 99;
        EXPECT_TRUE(IntegerTryCastToVarlen::Run(source, target, col_varchar_ptr.get()));
        src_str = std::to_string(source);
        EXPECT_EQ(src_str.size(), 2u);
        EXPECT_STREQ(src_str.c_str(), target.ToString().c_str());

        source = -100;
        EXPECT_TRUE(IntegerTryCastToVarlen::Run(source, target, col_varchar_ptr.get()));
        src_str = std::to_string(source);
        EXPECT_EQ(src_str.size(), 4u);
        EXPECT_STREQ(src_str.c_str(), target.ToString().c_str());

        source = 100;
        EXPECT_TRUE(IntegerTryCastToVarlen::Run(source, target, col_varchar_ptr.get()));
        src_str = std::to_string(source);
        EXPECT_EQ(src_str.size(), 3u);
        EXPECT_STREQ(src_str.c_str(), target.ToString().c_str());
    }
}

TEST_F(IntegerCastTest, integer_cast1) {
    using namespace infinity;

    SharedPtr<DataType> source_type = MakeShared<DataType>(LogicalType::kInteger);
    SharedPtr<ColumnVector> col_source = MakeShared<ColumnVector>(source_type);
    col_source->Initialize();
    for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
        Value v = Value::MakeInt(static_cast<IntegerT>(i));
        col_source->AppendValue(v);
        Value vx = col_source->GetValueByIndex(i);
    }
    for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
        Value vx = col_source->GetValueByIndex(i);
        EXPECT_EQ(vx.type().type(), LogicalType::kInteger);
        EXPECT_EQ(vx.value_.integer, static_cast<IntegerT>(i));
    }

    // cast integer column vector to tiny int column vector
    {
        SharedPtr<DataType> target_type = MakeShared<DataType>(LogicalType::kTinyInt);
        auto int2tiny_ptr = BindIntegerCast<IntegerT>(*source_type, *target_type);
        EXPECT_NE(int2tiny_ptr.function, nullptr);

        SharedPtr<ColumnVector> col_target = MakeShared<ColumnVector>(target_type);
        col_target->Initialize();

        CastParameters cast_parameters;
        bool result = int2tiny_ptr.function(col_source, col_target, DEFAULT_VECTOR_SIZE, cast_parameters);
        EXPECT_FALSE(result);
        for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
            Value vx = col_target->GetValueByIndex(i);
            EXPECT_EQ(vx.type().type(), LogicalType::kTinyInt);
            i32 check_value = 0;
            if (i >= std::numeric_limits<i8>::min() && i <= std::numeric_limits<i8>::max()) {
                check_value = static_cast<i32>(i);
            }
            EXPECT_EQ(vx.value_.tiny_int, static_cast<TinyIntT>(check_value));
        }
    }

    // cast int column vector to small integer column vector
    {
        SharedPtr<DataType> target_type = MakeShared<DataType>(LogicalType::kSmallInt);
        auto int2small_ptr = BindIntegerCast<IntegerT>(*source_type, *target_type);
        EXPECT_NE(int2small_ptr.function, nullptr);

        SharedPtr<ColumnVector> col_target = MakeShared<ColumnVector>(target_type);
        col_target->Initialize();

        CastParameters cast_parameters;
        bool result = int2small_ptr.function(col_source, col_target, DEFAULT_VECTOR_SIZE, cast_parameters);
        EXPECT_TRUE(result);
        for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
            Value vx = col_target->GetValueByIndex(i);
            EXPECT_EQ(vx.type().type(), LogicalType::kSmallInt);
            i32 check_value = 0;
            if (i >= std::numeric_limits<i16>::min() && i <= std::numeric_limits<i16>::max()) {
                check_value = static_cast<i32>(i);
            }
            EXPECT_EQ(vx.value_.small_int, static_cast<SmallIntT>(check_value));
        }
    }

    // cast int column vector to big int column vector
    {
        SharedPtr<DataType> target_type = MakeShared<DataType>(LogicalType::kBigInt);
        auto int2bigint_ptr = BindIntegerCast<IntegerT>(*source_type, *target_type);
        EXPECT_NE(int2bigint_ptr.function, nullptr);

        SharedPtr<ColumnVector> col_target = MakeShared<ColumnVector>(target_type);
        col_target->Initialize();

        CastParameters cast_parameters;
        bool result = int2bigint_ptr.function(col_source, col_target, DEFAULT_VECTOR_SIZE, cast_parameters);
        EXPECT_TRUE(result);
        for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
            Value vx = col_target->GetValueByIndex(i);
            EXPECT_EQ(vx.type().type(), LogicalType::kBigInt);
            i32 check_value = static_cast<i32>(i);
            EXPECT_EQ(vx.value_.big_int, static_cast<BigIntT>(check_value));
        }
    }

    // cast int column vector to huge int column vector
    {
        SharedPtr<DataType> target_type = MakeShared<DataType>(LogicalType::kHugeInt);
        auto int2hugeint_ptr = BindIntegerCast<IntegerT>(*source_type, *target_type);
        EXPECT_NE(int2hugeint_ptr.function, nullptr);

        SharedPtr<ColumnVector> col_target = MakeShared<ColumnVector>(target_type);
        col_target->Initialize();

        CastParameters cast_parameters;
        bool result = int2hugeint_ptr.function(col_source, col_target, DEFAULT_VECTOR_SIZE, cast_parameters);
        EXPECT_TRUE(result);
        for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
            Value vx = col_target->GetValueByIndex(i);
            EXPECT_EQ(vx.type().type(), LogicalType::kHugeInt);
            HugeIntT check_value((static_cast<i32>(i) < 0) * -1, static_cast<i32>(i));
            EXPECT_EQ(vx.value_.huge_int, check_value);
        }
    }

    // cast int column vector to float column vector
    {
        SharedPtr<DataType> target_type = MakeShared<DataType>(LogicalType::kFloat);
        auto int2float_ptr = BindIntegerCast<IntegerT>(*source_type, *target_type);
        EXPECT_NE(int2float_ptr.function, nullptr);

        SharedPtr<ColumnVector> col_target = MakeShared<ColumnVector>(target_type);
        col_target->Initialize();

        CastParameters cast_parameters;
        bool result = int2float_ptr.function(col_source, col_target, DEFAULT_VECTOR_SIZE, cast_parameters);
        EXPECT_TRUE(result);
        for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
            Value vx = col_target->GetValueByIndex(i);
            EXPECT_EQ(vx.type().type(), LogicalType::kFloat);
            i32 check_value = static_cast<i32>(i);
            EXPECT_FLOAT_EQ(vx.value_.float32, static_cast<FloatT>(check_value));
        }
    }

    // cast int column vector to double column vector
    {
        SharedPtr<DataType> target_type = MakeShared<DataType>(LogicalType::kDouble);
        auto int2double_ptr = BindIntegerCast<IntegerT>(*source_type, *target_type);
        EXPECT_NE(int2double_ptr.function, nullptr);

        SharedPtr<ColumnVector> col_target = MakeShared<ColumnVector>(target_type);
        col_target->Initialize();

        CastParameters cast_parameters;
        bool result = int2double_ptr.function(col_source, col_target, DEFAULT_VECTOR_SIZE, cast_parameters);
        EXPECT_TRUE(result);
        for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
            Value vx = col_target->GetValueByIndex(i);
            EXPECT_EQ(vx.type().type(), LogicalType::kDouble);
            i32 check_value = static_cast<i32>(i);
            EXPECT_FLOAT_EQ(vx.value_.float64, static_cast<DoubleT>(check_value));
        }
    }

    // cast integer column vector to decimal128 column vector
    {
        SharedPtr<DataType> target_type = MakeShared<DataType>(LogicalType::kDecimal);
        auto int2decimal_ptr = BindIntegerCast<IntegerT>(*source_type, *target_type);
        EXPECT_NE(int2decimal_ptr.function, nullptr);

        SharedPtr<ColumnVector> col_target = MakeShared<ColumnVector>(target_type);
        col_target->Initialize();

        CastParameters cast_parameters;
        EXPECT_THROW(int2decimal_ptr.function(col_source, col_target, DEFAULT_VECTOR_SIZE, cast_parameters), UnrecoverableException);
    }

    // cast int column vector to Varchar vector
    {
        SharedPtr<DataType> target_type = MakeShared<DataType>(LogicalType::kVarchar);
        auto int2varchar_ptr = BindIntegerCast<IntegerT>(*source_type, *target_type);
        EXPECT_NE(int2varchar_ptr.function, nullptr);

        SharedPtr<ColumnVector> col_target = MakeShared<ColumnVector>(target_type);
        col_target->Initialize();

        CastParameters cast_parameters;
        bool result = int2varchar_ptr.function(col_source, col_target, DEFAULT_VECTOR_SIZE, cast_parameters);
        // Not all values are cast, then it's false.
        EXPECT_TRUE(result);
        for (i64 i = 0; i < DEFAULT_VECTOR_SIZE; ++i) {
            i32 check_value = static_cast<i32>(i);
            String check_str(std::to_string(check_value));
            Value vx = col_target->GetValueByIndex(i);
            const String &s2 = vx.GetVarchar();
            EXPECT_STREQ(s2.c_str(), check_str.c_str());
        }
    }

    // Throw exception when cast int to other types.
    {
        DataType source(LogicalType::kInteger);
        DataType target(LogicalType::kTimestamp);
        EXPECT_THROW(BindIntegerCast<IntegerT>(source, target), RecoverableException);
    }
}
