import os

import pandas as pd
import pyarrow as pa
import json
from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm
from datetime import datetime


def preprocess_parquet_files(
        input_dir: str,
        output_path: str,
        compression: str = 'snappy',
        chunk_size: int = 1_000_000
) -> None:
    """
    分块预处理Parquet格式的购物数据

    参数：
    input_dir: 输入目录路径
    output_path: 输出目录路径
    compression: 压缩算法（snappy/zstd/gzip）
    chunk_size: 每个处理块的行数
    """
    start_time = datetime.now()
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历所有Parquet文件
    for file_path in Path(input_dir).glob("*.parquet"):
        file_name = file_path.name
        output_file = output_dir / f"processed_{file_name}"
        writer = None  # Parquet写入器

        try:
            # 初始化进度条
            parquet_file = pq.ParquetFile(file_path)
            with tqdm(
                    total=parquet_file.num_row_groups,
                    desc=f"处理 {file_name}",
                    unit="rowgroup",
                    dynamic_ncols=True,
                    mininterval=0.5,
                    leave=True
            ) as progress:

                # 处理每个行组
                for rg in range(parquet_file.num_row_groups):
                    table = parquet_file.read_row_group(rg, columns=['id', 'purchase_history'])

                    # 分块处理
                    for batch in table.to_batches(max_chunksize=chunk_size):
                        df = batch.to_pandas()
                        processed = _process_chunk(df)

                        if not processed.empty:
                            # 初始化写入器（首次写入时创建）
                            if writer is None:
                                schema = pa.Schema.from_pandas(processed)
                                writer = pq.ParquetWriter(
                                    output_file,
                                    schema,
                                    compression=compression
                                )

                            # 追加写入数据
                            writer.write_table(pa.Table.from_pandas(processed))

                    # 更新进度
                    progress.update(1)
                    progress.set_postfix(rows=table.num_rows, refresh=True)

        except Exception as e:
            tqdm.write(f"[ERROR] 处理失败 {file_name}: {str(e)}")
        finally:
            # 关闭写入器
            if writer is not None:
                writer.close()

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n总耗时: {duration // 3600:.0f}h {duration % 3600 // 60:.0f}m {duration % 60:.2f}s")


def _process_chunk(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """处理单个数据块"""
    records = []
    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator,
                        total=len(df),
                        desc="处理数据块",
                        unit="行",
                        dynamic_ncols=True)
    for _, row in iterator:
        try:
            purchase = json.loads(row['purchase_history'])
            records.append({
                "user_id": row['id'],
                "purchase_date": pd.to_datetime(purchase['purchase_date']).normalize(),
                "payment_method": purchase.get('payment_method'),
                "payment_status": purchase.get('payment_status'),
                "item_ids": [item['id'] for item in purchase.get('items', [])],
            })
        except Exception as e:
            tqdm.write(f"[ERROR] 处理记录失败 user_id={row['id']}: {str(e)}")
    return pd.DataFrame(records)

def preprocess_in_one_go(raw_dir: str, output_dir: str):
    """直接处理所有数据"""
    df = pd.read_parquet(raw_dir, engine='pyarrow', columns=['id', 'purchase_history'])
    processed = _process_chunk(df, show_progress=True)
    os.makedirs(output_dir, exist_ok=True)
    processed.to_parquet(output_dir+ "/preprocessed.parquet", engine='pyarrow', compression='snappy')


if __name__ == "__main__":
    dataset_name = 'sample_data_s'
    raw_dir = f'../data/{dataset_name}'
    processed_dir = f'../data/{dataset_name}_processed'
    # preprocess_parquet_files(
    #     input_dir=raw_dir,
    #     output_path=processed_dir,
    # )
    preprocess_in_one_go(raw_dir, processed_dir)
    df = pd.read_parquet(processed_dir, engine='pyarrow')
    print(f"总记录数: {len(df)}")
    print(f"日期范围: {df['purchase_date'].min()} ~ {df['purchase_date'].max()}")
    print(f"唯一用户数: {df['user_id'].nunique()}")
