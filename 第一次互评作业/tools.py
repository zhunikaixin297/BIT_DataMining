import numpy as np
import pandas as pd
from pandas import json_normalize
from pyarrow.parquet import ParquetFile
from pyarrow import parquet as pq
from pyarrow import Table
from hashlib import sha256
from collections import defaultdict
import os
import json

# 数据分块加载器
def column_loader(parquet_dir, columns=None, chunksize=4e6):
    """分块加载器"""
    expanded_fields = {'purchase_average_price', 'purchase_category', 'purchase_item_ids'}

    for file in os.listdir(parquet_dir):
        if not file.endswith('.parquet'):
            continue

        file_path = os.path.join(parquet_dir, file)
        pf = ParquetFile(file_path)

        # 智能检测处理模式
        has_raw_field = 'purchase_history' in pf.schema.names
        has_expanded_fields = len(expanded_fields & set(pf.schema.names)) > 0

        # 动态确定加载列
        if has_raw_field:
            if columns:
                need_process = len(expanded_fields & set(columns)) > 0
                actual_columns = [col for col in columns if col not in expanded_fields]
                if need_process:
                    actual_columns.append('purchase_history')
            else:
                actual_columns = pf.schema.names  # 加载全部原始字段
            process_mode = 'raw'
        else:
            actual_columns = columns
            process_mode = 'clean'

        # 分块加载与处理
        for batch in pf.iter_batches(columns=actual_columns, batch_size=chunksize):
            df = batch.to_pandas()

            # 字段展开逻辑
            if process_mode == 'raw' and 'purchase_history' in df.columns:
                df = process_expansion(df, required_fields=columns)

            # 类型优化
            df = apply_dtype_optimization(df)

            # 动态确定输出列
            if columns:
                final_columns = [col for col in columns if col in df.columns]
            else:
                final_columns = df.columns.tolist()

            yield df[final_columns]

def process_expansion(df, required_fields=None):
    """处理字段展开逻辑"""
    try:
        # 解析JSON字段
        parsed = df['purchase_history'].apply(
            lambda x: json.loads(x) if pd.notnull(x) else {}
        )

        # 展开JSON字段
        df_purchase = json_normalize(parsed).rename(columns={
            'average_price': 'purchase_average_price',
            'category': 'purchase_category',
            'items': 'purchase_items'
        })

        # 处理items数组
        if 'purchase_items' in df_purchase:
            df_purchase['purchase_item_ids'] = df_purchase['purchase_items'].apply(
                lambda x: tuple(item['id'] for item in x) if isinstance(x, list) else ()
            )
            df_purchase = df_purchase.drop('purchase_items', axis=1)

        # 合并到原始数据
        merged = pd.concat([
            df.drop('purchase_history', axis=1),
            df_purchase[[col for col in ['purchase_average_price', 'purchase_category', 'purchase_item_ids'] if
                         (required_fields is None) or (col in required_fields)]]
        ], axis=1)

        return merged

    except Exception as e:
        print(f"字段展开失败: {str(e)}")
        return df.drop('purchase_history', axis=1, errors='ignore')


def apply_dtype_optimization(df):
    """应用类型优化"""
    dtype_map = {
        'id': 'uint32',
        'age': 'uint8',
        'income': 'float32',
        'gender': 'category',
        'country': 'category',
        'is_active': 'bool',
        'credit_score': 'uint16',
        'purchase_average_price': 'float32',
        'purchase_category': 'category',
        'purchase_item_ids': 'object'
    }

    # 仅转换存在的列
    valid_columns = {col: dtype for col, dtype in dtype_map.items() if col in df.columns}
    return df.astype(valid_columns, errors='ignore')


def preprocess(parquet_dir, output_dir):
    """预处理"""
    # ================== 配置参数 ==================
    COLUMNS_TO_DROP = ['email', 'chinese_address', 'registration_date', 'phone_number', 'purchase_item_ids']
    NUMERIC_COLS = ['age', 'income', 'credit_score', 'purchase_average_price']

    # ================== 初始化 ==================
    dedup_filter = ScalableBloomFilter(error_rate=0.0001)
    stats = {
        'duplicates_removed': 0,
        'missing_filled': defaultdict(int),
        'outliers_removed': defaultdict(int)
    }

    # ================== 处理流程 ==================
    chunk_gen = column_loader(parquet_dir)
    for chunk_idx, chunk in enumerate(chunk_gen):
        # 移除无用字段
        chunk = chunk.drop(columns=[col for col in COLUMNS_TO_DROP if col in chunk.columns])

        # 全局去重
        hash_str = chunk['id'].astype(str) + "_" + chunk['timestamp'] + "_" + chunk['chinese_name']
        hashes = hash_str.apply(lambda x: sha256(x.encode()).hexdigest())
        is_dup = hashes.apply(lambda x: dedup_filter.add(x))
        chunk = chunk[~is_dup]
        stats['duplicates_removed'] += is_dup.sum()

        # 缺失值处理
        key_cols = ['id', 'timestamp', 'chinese_name'] + NUMERIC_COLS
        valid_cols = [col for col in key_cols if col in chunk.columns]
        original_size = len(chunk)
        chunk = chunk.dropna(subset=valid_cols)
        stats['missing_filled']['total'] += (original_size - len(chunk))

        # 异常值处理
        for col in NUMERIC_COLS:
            if col in chunk.columns:
                q1 = chunk[col].quantile(0.25)
                q3 = chunk[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (chunk[col] >= lower) & (chunk[col] <= upper)
                stats['outliers_removed'][col] += (~mask).sum()
                chunk = chunk[mask]

        # 保存数据
        save_clean_chunk(chunk, output_dir, chunk_idx)

    return stats


def save_clean_chunk(df, output_dir, chunk_idx):
    """优化存储函数"""
    os.makedirs(output_dir, exist_ok=True)
    table = Table.from_pandas(df)
    pq.write_table(
        table,
        os.path.join(output_dir, f"clean_{chunk_idx}.parquet"),
        compression='brotli',
        use_dictionary=False  # 禁用字典编码加速存储
    )


class ScalableBloomFilter:
    """可扩展布隆过滤器"""

    def __init__(self, error_rate=0.001):
        from pybloom_live import ScalableBloomFilter
        self.filter = ScalableBloomFilter(error_rate=error_rate)

    def add(self, item):
        exists = item in self.filter
        self.filter.add(item)
        return exists


def get_dataset_summary(parquet_dir):
    # 初始化统计存储
    total_records = 0
    sample_data = []
    numerical_data = defaultdict(list)
    categorical_counts = defaultdict(lambda: defaultdict(int))

    # 预定义字段分类
    num_fields = ['id', 'age', 'income', 'credit_score', 'purchase_average_price']
    cat_fields = ['gender', 'country', 'is_active', 'purchase_category']

    # 初始化分块加载器
    chunk_generator = column_loader(parquet_dir)

    for i, chunk in enumerate(chunk_generator):

        # 累积统计信息
        total_records += len(chunk)

        # 收集样本数据
        if i == 0 and not chunk.empty:
            sample_data.append(chunk.sample(5).copy())

        # 处理数值字段
        for col in num_fields:
            numerical_data[col].extend(chunk[col].dropna().values)

        # 处理分类字段
        for col in cat_fields:
            counts = chunk[col].value_counts(dropna=False)
            for val, cnt in counts.items():
                key = val if pd.notnull(val) else 'NaN'
                categorical_counts[col][key] += cnt

    # 构建最终统计结果
    numerical_summary = {}
    for col, values in numerical_data.items():
        if values:
            s = pd.Series(values)
            numerical_summary[col] = {
                'min': s.min(),
                'q1': s.quantile(0.25),
                'median': s.median(),
                'q3': s.quantile(0.75),
                'max': s.max()
            }

    categorical_summary = {
        col: dict(counts)
        for col, counts in categorical_counts.items()
    }

    return {
        'total_records': total_records,
        'sample': pd.concat(sample_data, ignore_index=True) if sample_data else pd.DataFrame(),
        'numerical': numerical_summary,
        'categorical': categorical_summary
    }


def enhanced_summary(parquet_dir, num_fields, cat_fields):
    # 初始化统计容器
    stats = {
        'total': 0,
        'histograms': defaultdict(lambda: defaultdict(int)),
        'corr_stats': defaultdict(lambda: {'sum': 0, 'sum2': 0, 'n': 0}),
        'corr_pairs': defaultdict(lambda: {'sum_x': 0, 'sum_y': 0, 'sum_xy': 0, 'sum_x2': 0, 'sum_y2': 0, 'n': 0}),
        'samples': defaultdict(list),
        'numerical': defaultdict(dict),
        'categorical': defaultdict(dict),
        'full_data': defaultdict(list)  # 新增：累积全量数据
    }

    # 初始化分块加载器
    chunk_generator = column_loader(parquet_dir)

    # 第一遍：确定直方图区间
    # print("正在确定数据分布范围...")
    ranges = {}
    for chunk in chunk_generator:
        for col in num_fields:
            col_min = chunk[col].min()
            col_max = chunk[col].max()
            if col not in ranges:
                ranges[col] = (col_min, col_max)
            else:
                ranges[col] = (min(ranges[col][0], col_min),
                               max(ranges[col][1], col_max))
        break  # 仅用第一个块确定范围

    # 生成直方图区间
    bins = {col: np.linspace(r[0], r[1], 21) for col, r in ranges.items()}

    # 第二遍：累积统计量
    # print("正在计算统计量...")
    chunk_generator = column_loader(parquet_dir)
    for chunk_idx, chunk in enumerate(chunk_generator):
        stats['total'] += len(chunk)

        # 数值字段处理
        for col in num_fields:
            data = chunk[col].dropna()

            # 累积全量数据
            stats['full_data'][col].extend(data.tolist())

            # 直方图统计
            hist, _ = np.histogram(data, bins=bins[col])
            for i in range(len(hist)):
                stats['histograms'][col][i] += hist[i]

            # 相关性统计
            stats['corr_stats'][col]['sum'] += data.sum()
            stats['corr_stats'][col]['sum2'] += (data  ** 2).sum()
            stats['corr_stats'][col]['n'] += len(data)

            # 交叉乘积
            for i, col1 in enumerate(num_fields):
                for col2 in num_fields[i + 1:]:
                    mask = chunk[col1].notna() & chunk[col2].notna()
                    valid_data = chunk.loc[mask, [col1, col2]]
                    if not valid_data.empty:
                        x = valid_data[col1].astype(float)
                        y = valid_data[col2].astype(float)
                        pair_key = (col1, col2)
                        stats['corr_pairs'][pair_key]['sum_x'] += x.sum()
                        stats['corr_pairs'][pair_key]['sum_y'] += y.sum()
                        stats['corr_pairs'][pair_key]['sum_xy'] += (x * y).sum()
                        stats['corr_pairs'][pair_key]['sum_x2'] += (x  ** 2).sum()
                        stats['corr_pairs'][pair_key]['sum_y2'] += (y  ** 2).sum()
                        stats['corr_pairs'][pair_key]['n'] += len(x)

        # 分类字段处理
        for col in cat_fields:
            counts = chunk[col].value_counts(dropna=False)
            for val, cnt in counts.items():
                key = val if pd.notnull(val) else 'NaN'
                stats['categorical'][col][key] = stats['categorical'][col].get(key, 0) + cnt

        # 采样数据
        if chunk_idx % 100 == 0:
            sample = chunk.sample(frac=0.001)
            for col in num_fields:
                stats['samples'][col].extend(sample[col].dropna().tolist())

    # 全量数据计算五数概括
    for col in num_fields:
        if stats['full_data'][col]:
            s = pd.Series(stats['full_data'][col])
            stats['numerical'][col] = {
                'min': s.min(),
                'q1': s.quantile(0.25),
                'median': s.median(),
                'q3': s.quantile(0.75),
                'max': s.max()
            }

    return {
        'bins': bins,
        'stats': stats,
    }