"""
退款关联分析流程
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules


# ----------------------
# 核心功能函数（步骤1-5）
# ----------------------
def load_refund_data(data_path: str) -> pd.DataFrame:
    """步骤1：加载退款数据"""
    df = pd.read_parquet(
        data_path,
        columns=['payment_status', 'item_ids']
    )
    # 过滤有效退款状态
    valid_status = ["已退款", "部分退款"]
    return df[df['payment_status'].isin(valid_status)]


def create_refund_mapping(catalog_path: str) -> Dict[int, str]:
    """步骤2：创建退款分析映射"""
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)

    mapping = {}
    for item in catalog['products']:
        try:
            mapping[int(item['id'])] = item['category']
        except (KeyError, ValueError) as e:
            print(f"异常商品条目：{item}，错误：{str(e)}")
    return mapping


def build_refund_dataset(df: pd.DataFrame, id_map: Dict[int, str]) -> List[list]:
    """步骤3：构建退款关联数据集"""
    transactions = []
    for _, row in df.iterrows():
        # 转换支付状态为特征项
        status_feature = f"STATUS_{row['payment_status']}"

        # 转换商品ID为类别
        categories = []
        for item_id in row['item_ids']:
            if item_id in id_map:
                categories.append(id_map[item_id])

        if categories:  # 过滤空交易
            transactions.append([status_feature] + categories)
    return transactions


def analyze_refund_associations(transactions: List[list], chunk_size: int = 100_000) -> (pd.DataFrame, pd.DataFrame):
    """步骤4：执行退款关联分析"""
    # 数据编码
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

    # 分布式FP-Growth挖掘
    from distributed_fpgrowth import distributed_fpgrowth
    fis = distributed_fpgrowth(
        encoded_df,
        min_support=0.005,
        chunk_size=chunk_size,
        max_len=3  # 限制最大组合长度
    )

    # 生成关联规则
    rules = association_rules(
        fis,
        metric="confidence",
        min_threshold=0.4
    )

    def _count_categories(itemset) -> int:
        """统计纯商品类别的数量"""
        return sum(1 for item in itemset if 'STATUS_' not in item)
    # 过滤有效规则
    valid_rules = rules[
        # 必须包含支付状态
        (rules['antecedents'].apply(lambda x: any('STATUS_' in s for s in x)) |
         rules['consequents'].apply(lambda x: any('STATUS_' in s for s in x))) &
        # 商品组合数量≥2（排除单品类规则）
        (rules.apply(lambda x:
                     _count_categories(x['antecedents']) >= 2 or
                     _count_categories(x['consequents']) >= 2, axis=1))
        ]
    return fis, valid_rules


def export_refund_results(fis: pd.DataFrame, rules: pd.DataFrame, analysis_type: str) -> None:
    """步骤5：导出分析结果"""
    output_dir = Path("../data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    fis.to_csv(output_dir / f"{analysis_type}_frequent_itemsets.csv", index=False)
    rules.to_csv(output_dir / f"{analysis_type}_association_rules.csv", index=False)


def load_analysis_results(analysis_type: str) -> (pd.DataFrame, pd.DataFrame):
    """加载已存储的频繁项集和关联规则"""
    output_dir = Path("../data/output")

    # 构建完整文件路径
    fis_path = output_dir / f"{analysis_type}_frequent_itemsets.csv"
    rules_path = output_dir / f"{analysis_type}_association_rules.csv"

    # 带类型转换的读取
    fis = pd.read_csv(fis_path, converters={'itemsets': eval})
    rules = pd.read_csv(rules_path, converters={
        'antecedents': eval,
        'consequents': eval
    })

    return fis, rules

# ----------------------
# 分析展示函数（步骤6-7）
# ----------------------
def analyze_refund_patterns(rules: pd.DataFrame) -> pd.DataFrame:
    """步骤6：分析退款组合模式"""

    # 筛选高风险规则（前件为商品组合，后件为退款状态）
    high_risk = rules[
        rules['consequents'].apply(lambda x: any('STATUS_' in s for s in x)) &
        rules['antecedents'].apply(lambda x: all('STATUS_' not in s for s in x))
        ]

    # 添加风险评分
    high_risk['risk_score'] = high_risk['support'] * high_risk['lift']
    return high_risk.sort_values('risk_score', ascending=False)


def summarize_qualified_itemsets(fis: pd.DataFrame) -> None:
    """总结包含支付状态和商品组合的频繁项集"""
    qualified_items = []

    for _, row in fis.iterrows():
        itemset = row['itemsets']
        status_items = [item for item in itemset if item.startswith('STATUS_')]
        category_items = [item for item in itemset if not item.startswith('STATUS_')]

        # 条件：包含至少1个支付状态且至少2个商品类别
        if len(status_items) >= 1 and len(category_items) >= 1:
            qualified_items.append({
                'itemset': itemset,
                'support': row['support'],
                'status_count': len(status_items),
                'category_count': len(category_items)
            })

    if not qualified_items:
        print("未找到包含支付状态和商品组合的频繁项集")
        return

    # 转换为DataFrame并排序
    summary_df = pd.DataFrame(qualified_items).sort_values('support', ascending=False)

    # 控制台输出格式化结果
    print("\n符合要求的频繁项集总结（包含支付状态且商品组合≥1）：")
    print(f"发现 {len(summary_df)} 个有效项集")
    print("Top 5  高频项集：")

    for idx, row in summary_df.head(5).iterrows():
        status_str = ", ".join([s.replace("STATUS_", "") for s in row['itemset'] if s.startswith("STATUS_")])
        categories_str = ", ".join([c for c in row['itemset'] if not c.startswith("STATUS_")])
        print(f"[支持度 {row['support']:.4f}]")
        print(f"  支付状态：{status_str}")
        print(f"  商品组合：{categories_str}\n")


def visualize_refund_analysis(rules: pd.DataFrame) -> None:
    """步骤7：可视化退款分析"""
    import matplotlib.pyplot as plt
    import networkx as nx

    try:
        if len(rules) == 0:
            print("没有有效的退款关联规则，无法进行可视化。")
            return

        plt.figure(figsize=(15, 6))

        # 子图1：规则指标分布
        plt.subplot(121)
        plt.scatter(rules['support'], rules['confidence'],
                    c=rules['lift'], cmap='viridis', alpha=0.6)
        plt.colorbar(label='提升度')
        plt.xlabel('支持度')
        plt.ylabel('置信度')
        plt.title('退款规则分布')

        # 子图2：规则网络
        plt.subplot(122)
        G = nx.from_pandas_edgelist(
            rules.head(10),
            'antecedents',
            'consequents',
            edge_attr=['lift', 'confidence']
        )
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=800)
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                               width=[d['lift'] * 0.3 for u, v, d in G.edges(data=True)])
        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title("Top10退款规则网络")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"可视化失败：{str(e)}")


# ----------------------
# 执行入口
# ----------------------
if __name__ == "__main__":
    # 执行核心流程
    refund_df = load_refund_data("../data/sample_data_processed")
    refund_map = create_refund_mapping("../data/product_catalog.json")
    refund_trans = build_refund_dataset(refund_df, refund_map)
    fis, rules = analyze_refund_associations(refund_trans, chunk_size=800_000)
    export_refund_results(fis, rules, "refund_analysis")


    # 执行分析展示
    loaded_fis, loaded_rules = load_analysis_results("refund_analysis")
    summarize_qualified_itemsets(loaded_fis)
    high_risk = analyze_refund_patterns(loaded_rules)
    print("\n高风险退款规则：")
    if len(high_risk) > 0:
        print(high_risk[['antecedents', 'consequents', 'risk_score']].head(5))
    else:
        print("未找到导致退款的可能商品组合模式。")

    # visualize_refund_analysis(loaded_rules)