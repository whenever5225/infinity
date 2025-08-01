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

module;

#include <cassert>
#include <ostream>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <xmmintrin.h>
#elif defined(__GNUC__) && defined(__aarch64__)
#include <simde/x86/sse.h>
#endif

export module lvq_vec_store;

import stl;
import local_file_handle;
import hnsw_common;
import serialize;
import data_store_util;

namespace infinity {

export template <typename DataType, typename LocalCacheType, typename CompressType>
struct LVQData {
    DataType scale_;
    DataType bias_;
    LocalCacheType local_cache_;
    CompressType compress_vec_[];
};

export template <typename DataType, typename CompressType, typename LVQCache, bool OwnMem>
class LVQVecStoreInner;

export template <typename DataType, typename CompressType, typename LVQCache>
class LVQVecStoreMetaType {
public:
    using LocalCacheType = LVQCache::LocalCacheType;
    using LVQData = LVQData<DataType, LocalCacheType, CompressType>;
    struct LVQQuery {
        UniquePtr<LVQData> inner_;
        operator const LVQData *() const { return inner_.get(); }

        LVQQuery(SizeT compress_data_size) : inner_(new(new char[compress_data_size]) LVQData) {}
        LVQQuery(LVQQuery &&other) = default;
        ~LVQQuery() { delete[] reinterpret_cast<char *>(inner_.release()); }
    };

    using StoreType = const LVQData *;
    using QueryType = LVQQuery;
    using DistanceType = f32;
};

template <typename DataType, typename CompressType, typename LVQCache, bool OwnMem>
class LVQVecStoreMetaBase {
public:
    // Compress type must be i8 temporarily
    static_assert(std::is_same<CompressType, i8>() || std::is_same<CompressType, void>());
    constexpr static SizeT max_bucket_idx_ = std::numeric_limits<CompressType>::max() - std::numeric_limits<CompressType>::min(); // 255 for i8

    using This = LVQVecStoreMetaBase<DataType, CompressType, LVQCache, OwnMem>;
    using Inner = LVQVecStoreInner<DataType, CompressType, LVQCache, OwnMem>;
    using LocalCacheType = LVQCache::LocalCacheType;
    using GlobalCacheType = LVQCache::GlobalCacheType;
    using LVQData = LVQVecStoreMetaType<DataType, CompressType, LVQCache>::LVQData;
    using LVQQuery = LVQVecStoreMetaType<DataType, CompressType, LVQCache>::LVQQuery;

public:
    LVQVecStoreMetaBase() : dim_(0), compress_data_size_(0), normalize_(false) {}
    LVQVecStoreMetaBase(This &&other)
        : dim_(std::exchange(other.dim_, 0)), compress_data_size_(std::exchange(other.compress_data_size_, 0)), mean_(std::move(other.mean_)),
          global_cache_(std::exchange(other.global_cache_, GlobalCacheType())), normalize_(other.normalize_) {}
    LVQVecStoreMetaBase &operator=(This &&other) {
        if (this != &other) {
            dim_ = std::exchange(other.dim_, 0);
            compress_data_size_ = std::exchange(other.compress_data_size_, 0);
            mean_ = std::move(other.mean_);
            global_cache_ = std::exchange(other.global_cache_, GlobalCacheType());
            normalize_ = other.normalize_;
        }
        return *this;
    }

    SizeT GetSizeInBytes() const { return sizeof(dim_) + sizeof(MeanType) * dim_ + sizeof(GlobalCacheType); }

    // Get size of vector in search
    SizeT GetVecSizeInBytes() const { return compress_data_size_; }

    void Save(LocalFileHandle &file_handle) const {
        file_handle.Append(&dim_, sizeof(dim_));
        file_handle.Append(mean_.get(), sizeof(MeanType) * dim_);
        if constexpr (!std::same_as<GlobalCacheType, Tuple<>>) {
            file_handle.Append(&global_cache_, sizeof(GlobalCacheType));
        }
    }

    LVQQuery MakeQuery(const DataType *vec) const {
        LVQQuery query(compress_data_size_);
        CompressTo(vec, query.inner_.get());
        return query;
    }

    void CompressTo(const DataType *src, LVQData *dest) const {
        UniquePtr<DataType[]> normalized;
        if (normalize_) {
            normalized = MakeUniqueForOverwrite<DataType[]>(this->dim_);
            DataType norm = 0;
            for (SizeT j = 0; j < this->dim_; ++j) {
                DataType x = src[j];
                norm += x * x;
            }
            norm = std::sqrt(norm);
            if (norm == 0) {
                std::fill(normalized.get(), normalized.get() + this->dim_, 0);
            } else {
                for (SizeT j = 0; j < this->dim_; ++j) {
                    normalized[j] = src[j] / norm;
                }
            }
            src = normalized.get();
        }

        CompressType *compress = dest->compress_vec_;

        DataType lower = std::numeric_limits<DataType>::max();
        DataType upper = -std::numeric_limits<DataType>::max();
        for (SizeT j = 0; j < dim_; ++j) {
            auto x = static_cast<DataType>(src[j] - mean_[j]);
            lower = std::min(lower, x);
            upper = std::max(upper, x);
        }
        DataType scale = (upper - lower) / max_bucket_idx_;
        DataType bias = lower - std::numeric_limits<CompressType>::min() * scale;
        if (scale == 0) {
            std::fill(compress, compress + dim_, 0);
        } else {
            DataType scale_inv = 1 / scale;
            for (SizeT j = 0; j < dim_; ++j) {
                auto c = std::floor((src[j] - mean_[j] - bias) * scale_inv + 0.5);
                assert(c <= std::numeric_limits<CompressType>::max() && c >= std::numeric_limits<CompressType>::min());
                compress[j] = c;
            }
        }
        dest->scale_ = scale;
        dest->bias_ = bias;
        dest->local_cache_ = LVQCache::MakeLocalCache(compress, scale, dim_, mean_.get());
    }

    SizeT dim() const { return dim_; }
    SizeT compress_data_size() const { return compress_data_size_; }

    const GlobalCacheType &global_cache() const { return global_cache_; }

    // for unit test
    const MeanType *mean() const { return mean_.get(); }

protected:
    void DecompressByMeanTo(const LVQData *src, const MeanType *mean, DataType *dest) const {
        const CompressType *compress = src->compress_vec_;
        DataType scale = src->scale_;
        DataType bias = src->bias_;
        for (SizeT i = 0; i < dim_; ++i) {
            dest[i] = scale * compress[i] + bias + mean[i];
        }
    }

    void DecompressTo(const LVQData *src, DataType *dest) const { DecompressByMeanTo(src, mean_.get(), dest); };

protected:
    SizeT dim_;
    SizeT compress_data_size_;

    ArrayPtr<MeanType, OwnMem> mean_;
    GlobalCacheType global_cache_;

    bool normalize_{false};

public:
    void Dump(std::ostream &os) const {
        os << "[CONST] dim: " << dim_ << ", compress_data_size: " << compress_data_size_ << std::endl;
        os << "mean: ";
        for (SizeT i = 0; i < dim_; ++i) {
            os << mean_[i] << " ";
        }
        os << std::endl;
        LVQCache::DumpGlobalCache(os, global_cache_);
    }
};

export template <typename DataType, typename CompressType, typename LVQCache, bool OwnMem>
class LVQVecStoreMeta : public LVQVecStoreMetaBase<DataType, CompressType, LVQCache, OwnMem> {
    using This = LVQVecStoreMeta<DataType, CompressType, LVQCache, OwnMem>;
    using Inner = LVQVecStoreInner<DataType, CompressType, LVQCache, OwnMem>;
    using LocalCacheType = LVQCache::LocalCacheType;
    using LVQData = LVQData<DataType, LocalCacheType, CompressType>;
    using GlobalCacheType = LVQCache::GlobalCacheType;

private:
    LVQVecStoreMeta(SizeT dim) {
        this->dim_ = dim;
        this->compress_data_size_ = sizeof(LVQData) + sizeof(CompressType) * dim;
        this->mean_ = MakeUnique<MeanType[]>(dim);
        std::fill(this->mean_.get(), this->mean_.get() + dim, 0);
        this->global_cache_ = LVQCache::MakeGlobalCache(this->mean_.get(), dim);
    }

public:
    LVQVecStoreMeta() = default;
    static This Make(SizeT dim) { return This(dim); }
    static This Make(SizeT dim, bool normalize) {
        This ret(dim);
        ret.normalize_ = normalize;
        return ret;
    }

    static This Load(LocalFileHandle &file_handle) {
        SizeT dim;
        file_handle.Read(&dim, sizeof(dim));
        This meta(dim);
        file_handle.Read(meta.mean_.get(), sizeof(MeanType) * dim);
        if constexpr (!std::is_same_v<GlobalCacheType, Tuple<>>) {
            file_handle.Read(&meta.global_cache_, sizeof(GlobalCacheType));
        }
        return meta;
    }

    static This LoadFromPtr(const char *&ptr) {
        SizeT dim = ReadBufAdv<SizeT>(ptr);
        This meta(dim);
        std::memcpy(meta.mean_.get(), ptr, sizeof(MeanType) * dim);
        ptr += sizeof(MeanType) * dim;
        if constexpr (!std::is_same_v<GlobalCacheType, Tuple<>>) {
            std::memcpy(&meta.global_cache_, ptr, sizeof(GlobalCacheType));
            ptr += sizeof(GlobalCacheType);
        }
        return meta;
    }

    template <typename LabelType, DataIteratorConcept<const DataType *, LabelType> Iterator>
    void Optimize(Iterator &&query_iter, const Vector<Pair<Inner *, SizeT>> &inners, SizeT &mem_usage) {
        auto new_mean = MakeUnique<MeanType[]>(this->dim_);
        auto temp_decompress = MakeUnique<DataType[]>(this->dim_);
        SizeT cur_vec_num = 0;
        for (const auto [inner, size] : inners) {
            for (SizeT i = 0; i < size; ++i) {
                this->DecompressTo(inner->GetVec(i, *this), temp_decompress.get());
                for (SizeT j = 0; j < this->dim_; ++j) {
                    new_mean[j] += temp_decompress[j];
                }
            }
            cur_vec_num += size;
        }
        while (true) {
            if (auto ret = query_iter.Next(); ret) {
                auto &[vec, _] = *ret;
                for (SizeT i = 0; i < this->dim_; ++i) {
                    new_mean[i] += vec[i];
                }
                ++cur_vec_num;
            } else {
                break;
            }
        }
        for (SizeT i = 0; i < this->dim_; ++i) {
            new_mean[i] /= cur_vec_num;
        }
        new_mean = this->mean_.exchange(std::move(new_mean)); //

        for (auto [inner, size] : inners) {
            for (SizeT i = 0; i < size; ++i) {
                this->DecompressByMeanTo(inner->GetVec(i, *this), new_mean.get(), temp_decompress.get());
                inner->SetVec(i, temp_decompress.get(), *this, mem_usage);
            }
        }
        this->global_cache_ = LVQCache::MakeGlobalCache(this->mean_.get(), this->dim_);
    }
};

export template <typename DataType, typename CompressType, typename LVQCache>
class LVQVecStoreMeta<DataType, CompressType, LVQCache, false> : public LVQVecStoreMetaBase<DataType, CompressType, LVQCache, false> {
    using This = LVQVecStoreMeta<DataType, CompressType, LVQCache, false>;
    using LocalCacheType = LVQCache::LocalCacheType;
    using LVQData = LVQData<DataType, LocalCacheType, CompressType>;
    using GlobalCacheType = LVQCache::GlobalCacheType;

private:
    LVQVecStoreMeta(SizeT dim, MeanType *mean, GlobalCacheType global_cache) {
        this->dim_ = dim;
        this->compress_data_size_ = sizeof(LVQData) + sizeof(CompressType) * dim;
        this->mean_ = mean;
        this->global_cache_ = global_cache;
    }

public:
    LVQVecStoreMeta() = default;

    static This LoadFromPtr(const char *&ptr) {
        SizeT dim = ReadBufAdv<SizeT>(ptr);
        auto *mean = reinterpret_cast<MeanType *>(const_cast<char *>(ptr));
        ptr += sizeof(MeanType) * dim;
        GlobalCacheType global_cache = ReadBufAdv<GlobalCacheType>(ptr);
        This meta(dim, mean, global_cache);
        return meta;
    }
};

template <typename DataType, typename CompressType, typename LVQCache, bool OwnMem>
class LVQVecStoreInnerBase {
public:
    using This = LVQVecStoreInnerBase<DataType, CompressType, LVQCache, OwnMem>;
    using Meta = LVQVecStoreMetaBase<DataType, CompressType, LVQCache, OwnMem>;
    // Decompress: Q = scale * C + bias + Mean
    using LocalCacheType = LVQCache::LocalCacheType;
    using LVQData = LVQData<DataType, LocalCacheType, CompressType>;

public:
    LVQVecStoreInnerBase() = default;

    SizeT GetSizeInBytes(SizeT cur_vec_num, const Meta &meta) const { return cur_vec_num * meta.compress_data_size(); }

    void Save(LocalFileHandle &file_handle, SizeT cur_vec_num, const Meta &meta) const {
        file_handle.Append(ptr_.get(), cur_vec_num * meta.compress_data_size());
    }

    static void
    SaveToPtr(LocalFileHandle &file_handle, const Vector<const This *> &inners, const Meta &meta, SizeT ck_size, SizeT chunk_num, SizeT last_chunk_size) {
        for (SizeT i = 0; i < chunk_num; ++i) {
            SizeT chunk_size = (i < chunk_num - 1) ? ck_size : last_chunk_size;
            file_handle.Append(inners[i]->ptr_.get(), chunk_size * meta.compress_data_size());
        }
    }

    const LVQData *GetVec(SizeT idx, const Meta &meta) const {
        return reinterpret_cast<const LVQData *>(ptr_.get() + idx * meta.compress_data_size());
    }

    void Prefetch(VertexType vec_i, const Meta &meta) const { _mm_prefetch(reinterpret_cast<const char *>(GetVec(vec_i, meta)), _MM_HINT_T0); }

protected:
    ArrayPtr<char, OwnMem> ptr_;

public:
    void Dump(std::ostream &os, SizeT offset, SizeT chunk_size, const Meta &meta) const {
        for (int i = 0; i < (int)chunk_size; ++i) {
            os << "vec " << i << "(" << offset + i << "): ";
            const LVQData *vec = GetVec(i, meta);
            os << "scale: " << vec->scale_ << ", bias: " << vec->bias_ << std::endl;
            os << "compress_vec: ";
            for (SizeT j = 0; j < meta.dim(); ++j) {
                os << static_cast<int>(vec->compress_vec_[j]) << " ";
            }
            os << std::endl;
            LVQCache::DumpLocalCache(os, vec->local_cache_);
        }
    }
};

export template <typename DataType, typename CompressType, typename LVQCache, bool OwnMem>
class LVQVecStoreInner : public LVQVecStoreInnerBase<DataType, CompressType, LVQCache, OwnMem> {
public:
    using This = LVQVecStoreInner<DataType, CompressType, LVQCache, OwnMem>;
    using Meta = LVQVecStoreMetaBase<DataType, CompressType, LVQCache, OwnMem>;
    using LocalCacheType = LVQCache::LocalCacheType;
    using LVQData = LVQData<DataType, LocalCacheType, CompressType>;
    using Base = LVQVecStoreInnerBase<DataType, CompressType, LVQCache, OwnMem>;

private:
    LVQVecStoreInner(SizeT max_vec_num, const Meta &meta) { this->ptr_ = MakeUnique<char[]>(max_vec_num * meta.compress_data_size()); }

public:
    LVQVecStoreInner() = default;

    static This Make(SizeT max_vec_num, const Meta &meta, SizeT &mem_usage) {
        auto ret = This(max_vec_num, meta);
        mem_usage += max_vec_num * meta.compress_data_size();
        return ret;
    }

    static This Load(LocalFileHandle &file_handle, SizeT cur_vec_num, SizeT max_vec_num, const Meta &meta, SizeT &mem_usage) {
        assert(cur_vec_num <= max_vec_num);
        This ret(max_vec_num, meta);
        file_handle.Read(ret.ptr_.get(), cur_vec_num * meta.compress_data_size());
        mem_usage += max_vec_num * meta.compress_data_size();
        return ret;
    }

    static This LoadFromPtr(const char *&ptr, SizeT cur_vec_num, SizeT max_vec_num, const Meta &meta, SizeT &mem_usage) {
        This ret(max_vec_num, meta);
        std::memcpy(ret.ptr_.get(), ptr, cur_vec_num * meta.compress_data_size());
        ptr += cur_vec_num * meta.compress_data_size();
        mem_usage += max_vec_num * meta.compress_data_size();
        return ret;
    }

    void SetVec(SizeT idx, const DataType *vec, const Meta &meta, SizeT &mem_usage) { meta.CompressTo(vec, GetVecMut(idx, meta)); }

private:
    LVQData *GetVecMut(SizeT idx, const Meta &meta) { return reinterpret_cast<LVQData *>(this->ptr_.get() + idx * meta.compress_data_size()); }
};

export template <typename DataType, typename CompressType, typename LVQCache>
class LVQVecStoreInner<DataType, CompressType, LVQCache, false> : public LVQVecStoreInnerBase<DataType, CompressType, LVQCache, false> {
public:
    using This = LVQVecStoreInner<DataType, CompressType, LVQCache, false>;
    using Meta = LVQVecStoreMetaBase<DataType, CompressType, LVQCache, false>;
    using Base = LVQVecStoreInnerBase<DataType, CompressType, LVQCache, false>;

private:
    LVQVecStoreInner(const char *ptr) { this->ptr_ = ptr; }

public:
    LVQVecStoreInner() = default;

    static This LoadFromPtr(const char *&ptr, SizeT cur_vec_num, const Meta &meta) {
        const char *p = ptr;
        This ret(p);
        ptr += cur_vec_num * meta.compress_data_size();
        return ret;
    }
};

} // namespace infinity