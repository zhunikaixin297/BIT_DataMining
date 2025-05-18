"""
时间序列模式分析模块
"""
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List
import pyarrow.parquet as pq
import seaborn as sns
from tqdm import tqdm

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示异常
# ----------------------
# 核心功能函数（步骤1-3）
# ----------------------
def load_time_series_data(data_path: str) -> pd.DataFrame:
    """步骤1：加载时序数据"""
    print("正在加载时序数据...")
    return pd.read_parquet(data_path, columns=['item_ids', 'purchase_date'])


def create_time_mapping(catalog_path: str) -> Dict[int, str]:
    """步骤2：创建时间分析映射"""
    print("创建商品类别映射...")
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)

    mapping = {}
    for item in catalog['products']:
        try:
            item_id = int(item['id'])
            mapping[item_id] = item['category']
        except (KeyError, ValueError, TypeError) as e:
            print(f"异常商品条目：{item}，错误：{str(e)}")
    return mapping


def build_time_dataset_chunk(chunk: pd.DataFrame, id_map: Dict[int, str]) -> pd.DataFrame:
    """步骤3：分块构建时序数据集"""

    # 转换商品类别
    def convert_categories(item_ids):
        categories = []
        for item_id in item_ids:
            try:
                categories.append(id_map.get(int(item_id), '未知'))
            except ValueError:
                continue
        return list(set(categories))  # 去重

    # 处理时间特征
    chunk = chunk.copy()
    chunk['purchase_date'] = pd.to_datetime(chunk['purchase_date'])
    chunk['year_quarter'] = chunk['purchase_date'].dt.to_period('Q').astype(str)
    chunk['year_month'] = chunk['purchase_date'].dt.strftime('%Y-%m')
    chunk['day_of_week'] = chunk['purchase_date'].dt.day_name()

    # 转换商品ID为类别并删除原始字段
    chunk['categories'] = chunk['item_ids'].apply(convert_categories)
    return chunk.drop(columns=['item_ids'])




# ----------------------
# 分块处理框架
# ----------------------
def chunk_processor(
        processed_dir: str,
        id_map: Dict[int, str],
        process_fn: callable,
        merge_fn: callable,
        chunk_size: int = 1000_000
) -> None:
    """分块处理框架"""
    # 获取所有Parquet文件
    parquet_files = list(Path(processed_dir).glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"目录中未找到Parquet文件: {processed_dir}")
    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in parquet_files)
    results = []
    with tqdm(total=total_rows, desc="数据分析", unit='row') as pbar:
        for file_path in parquet_files:
            # 创建PyArrow文件读取器
            with pq.ParquetFile(file_path) as parquet_file:
                    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=['item_ids', 'purchase_date']):
                        # 转换为Pandas DataFrame
                        raw_chunk = batch.to_pandas()
                        # 步骤3：构建时序数据
                        processed_chunk = build_time_dataset_chunk(raw_chunk, id_map)
                        # 执行处理逻辑
                        chunk_result = process_fn(processed_chunk)
                        results.append(chunk_result)

                        pbar.update(len(raw_chunk))
                        # 主动释放内存
                        del raw_chunk, processed_chunk

    merge_fn(results)


# ----------------------
# 季节性模式分析（步骤4）
# ----------------------
def seasonal_process(processed_chunk: pd.DataFrame) -> tuple:
    """处理单个分块，返回三个独立DataFrame"""
    # 季度统计
    quarterly = processed_chunk.groupby('year_quarter').size().reset_index(name='count')

    # 月度统计
    monthly = processed_chunk.groupby('year_month').size().reset_index(name='count')

    # 周内统计（保持顺序）
    week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly = (processed_chunk['day_of_week'].value_counts()
              .reindex(week_order, fill_value=0)
              .reset_index()
              .rename(columns={'index': 'day_of_week'}))

    return quarterly, monthly, weekly


def seasonal_merge(results: list) -> None:
    """合并分块结果并保存"""
    output_dir = Path("../data/output/seasonal")
    output_dir.mkdir(exist_ok=True)

    # 分离三类结果
    all_quarterly = [q for q, _, _ in results]
    all_monthly = [m for _, m, _ in results]
    all_weekly = [w for _, _, w in results]

    # 合并季度数据
    if all_quarterly:
        final_q = pd.concat(all_quarterly).groupby('year_quarter')['count'].sum().reset_index()
        final_q.to_csv(output_dir / "quarterly_stats.csv", index=False)

    # 合并月度数据
    if all_monthly:
        final_m = pd.concat(all_monthly).groupby('year_month')['count'].sum().reset_index()
        final_m.to_csv(output_dir / "monthly_trend.csv", index=False)

    # 合并周数据
    if all_weekly:
        final_w = pd.concat(all_weekly).groupby('day_of_week')['count'].sum().reset_index()
        final_w.to_csv(output_dir / "weekly_density.csv", index=False)

# ----------------------
# 品类时段分析（步骤5）
# ----------------------
def category_process(processed_chunk: pd.DataFrame) -> dict:
    """处理单个分块的品类数据（优化内存版）"""
    exploded = processed_chunk.explode('categories')
    return {
        'quarterly': exploded[['year_quarter', 'categories']],
        'monthly': exploded[['year_month', 'categories']]
    }


def category_merge(results: list) -> None:
    """合并品类分析结果"""
    # 合并季度数据
    quarterly = pd.concat([r['quarterly'] for r in results])
    quarterly_pct = (
        quarterly.value_counts(['year_quarter', 'categories'], normalize=True)
        .reset_index(name='percentage')
    )

    # 计算月度增长
    monthly_counts = pd.concat([r['monthly'] for r in results]).value_counts(['year_month', 'categories'])
    monthly_growth = (
        monthly_counts
        .groupby('categories', group_keys=False)
        .apply(lambda x: x.pct_change().mean())
        .nlargest(5)
        .reset_index(name='growth_rate')
        [['categories', 'growth_rate']]
    )


    # 保存结果
    output_dir = Path("../data/output/category/")
    output_dir.mkdir(exist_ok=True)
    quarterly_pct.to_csv(output_dir / "category_heatmap.csv", index=False)
    monthly_growth.to_json(output_dir / "top_categories.json", orient='records')

# ----------------------
# 可视化函数（步骤8）
# ----------------------
def visualize_seasonal_patterns():
    """可视化季节性模式"""
    print("生成可视化图表...")

    # 加载数据
    seasonal_dir = Path("../data/output/seasonal/")
    quarterly = pd.read_csv(seasonal_dir / "quarterly_stats.csv")
    monthly = pd.read_csv(seasonal_dir / "monthly_trend.csv")
    weekly = pd.read_csv(seasonal_dir / "weekly_density.csv")

    plt.figure(figsize=(18, 6))

    # 季度趋势
    plt.subplot(131)
    plt.plot(quarterly['year_quarter'], quarterly['count'], marker='o')
    plt.xticks(rotation=45)
    plt.title('季度购买趋势')

    # 月度趋势
    plt.subplot(132)
    monthly['date'] = pd.to_datetime(monthly['year_month'])
    plt.plot(monthly['date'], monthly['count'])
    plt.gcf().autofmt_xdate()
    plt.title('月度购买趋势')

    # 周内分布
    plt.subplot(133)
    week_order = ['Monday', 'Tuesday', 'Wednesday',
                  'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly['day_of_week'] = pd.Categorical(weekly['day_of_week'], categories=week_order)
    weekly = weekly.sort_values('day_of_week')
    plt.bar(weekly['day_of_week'], weekly['count'])
    plt.title('周内购买分布')

    plt.tight_layout()
    plt.show()


def visualize_category_patterns():
    """可视化品类时段特征"""
    category_dir = Path("../data/output/category/")
    quarterly_pct = pd.read_csv(category_dir / "category_heatmap.csv")
    top_cats = pd.read_json(category_dir / "top_categories.json")

    # 绘制热力图
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    pivot_table = quarterly_pct.pivot_table(
        index='year_quarter',
        columns='categories',
        values='percentage'
    ).fillna(0)
    sns.heatmap(pivot_table.T, cmap="YlGnBu", fmt=".1%")

    # 绘制Top增长
    plt.subplot(122)
    if not top_cats.empty:
        sns.barplot(
            x='growth_rate',
            y='categories',
            data=top_cats.sort_values('growth_rate', ascending=False),
            palette="Blues_d"
        )
        plt.title('Top5增长品类')
        plt.xlabel('平均月增长率')
    plt.tight_layout()
    plt.show()


# ----------------------
# 执行入口
# ----------------------
if __name__ == "__main__":
    # with open("../data/product_catalog.json") as f:
    #     catalog = json.load(f)
    # id_map = {int(item['id']): item['category'] for item in catalog['products']}
    # processed_dir = "../data/30G_data_new_processed"
    # # 执行季节性分析
    # chunk_processor(
    #     processed_dir,
    #     id_map,
    #     seasonal_process,
    #     seasonal_merge,
    #     chunk_size=800_000
    # )
    # # 执行品类分析
    # chunk_processor(
    #     processed_dir,
    #     id_map,
    #     category_process,
    #     category_merge,
    #     chunk_size=800_000
    # )

    # # 可视化展示
    visualize_seasonal_patterns()
    visualize_category_patterns()