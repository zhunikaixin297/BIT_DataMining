import numpy as np
import pandas as pd
from pathlib import Path


def create_sample(raw_dir: str, sample_dir: str, n_samples=1_000_000):
    """从单个Parquet文件抽取指定数量样本"""
    # 读取原始文件
    df = pd.read_parquet(Path(raw_dir) / 'part-00000.parquet')

    # 随机抽样
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    # 分块保存（每文件1万条）
    Path(sample_dir).mkdir(exist_ok=True)
    for i, chunk in enumerate(np.array_split(sample, 10)):
        chunk.to_parquet(
            Path(sample_dir) / f'sample_{i + 1}.parquet',
            engine='pyarrow',
            compression='snappy'
        )


# 使用示例
create_sample('../data/30G_data_new', '../data/sample_data_s', n_samples=1_000_000)