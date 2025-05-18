from collections import defaultdict

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from tqdm import tqdm
import gc


def distributed_fpgrowth(encoded_df, chunk_size=1000_000, min_support=0.02, max_len=3):
    """分布式FP-Growth算法实现"""
    # 全局单项计数
    print("计算全局单项频次...")
    item_counts = encoded_df.sum(axis=0)
    total_transactions = len(encoded_df)
    freq_columns = item_counts[item_counts / total_transactions >= min_support].index.tolist()

    # 初始化全局计数器
    support_counter = defaultdict(int)

    # 分块处理
    print("分块执行FP-Growth...")
    for i in tqdm(range(0, len(encoded_df), chunk_size),
                  desc="处理分块",
                  unit="chunk",
                  dynamic_ncols=True,
                  mininterval=0.5,
                  leave=True,
                  ):
        chunk = encoded_df.iloc[i:i + chunk_size][freq_columns]

        # 动态调整局部支持度阈值
        local_min_support = max(min_support * 0.5, 10 / len(chunk))  # 确保最小出现次数

        # 执行FP-Growth
        itemsets = fpgrowth(chunk,
                            min_support=local_min_support,
                            use_colnames=True,
                            max_len=max_len)

        # 转换支持度为绝对计数
        itemsets['count'] = (itemsets['support'] * len(chunk)).astype(int)

        # 累加到全局计数器
        for _, row in itemsets.iterrows():
            support_counter[frozenset(row['itemsets'])] += row['count']

        # 及时释放内存
        del chunk, itemsets
        gc.collect()


    # 生成最终结果
    print("生成最终项集...")
    results = []
    for itemset, count in support_counter.items():
        support = count / total_transactions
        if support >= min_support:
            results.append({'itemsets': itemset, 'support': support})

    return pd.DataFrame(results)


def _chunk_generator(data, chunk_size):
    """生成数据块迭代器"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

import psutil

def auto_chunk_size(encoded_df):
    """根据内存自动调整分块大小（修正版）"""
    mem_available = psutil.virtual_memory().available / 1e9  # 转换为GB
    cols = len(encoded_df.columns)
    # 正确计算：分母应为 cols × 1e-9（每列1字节）
    return int((mem_available * 0.4) / (cols * 1e-9))

if __name__ == "__main__":
    # 加载数据
    df = pd.read_parquet("../data/preprocessed.parquet")
    transactions = df['categories'].tolist()

    # 执行分布式FP-Growth
    frequent_itemsets = distributed_fpgrowth(transactions,
                                             chunk_size=50000,
                                             min_support=0.02)

    # 生成关联规则
    from mlxtend.frequent_patterns import association_rules

    rules = association_rules(frequent_itemsets,
                              metric="confidence",
                              min_threshold=0.5)

    print(f"发现有效规则 {len(rules)} 条")