import gc
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import association_rules


# ----------------------
# 核心功能函数（步骤1-5）
# ----------------------
def load_payment_data(processed_dir: str) -> pd.DataFrame:
    """步骤1：加载预处理数据"""
    print("正在加载支付数据...")
    return pd.read_parquet(processed_dir, columns=['payment_method', 'item_ids'])


def create_product_mapping(catalog_path: str) -> Dict[int, dict]:
    """步骤2：创建商品映射字典"""
    print("创建商品映射字典...")
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    return {
        item['id']: {
            'category': item['category'],
            'price': item['price']
        }
        for item in catalog['products']
    }


def build_association_dataset(df: pd.DataFrame, id_map: dict) -> List[list]:
    """步骤3：构建关联分析数据集"""
    transactions = []
    print("正在构建关联分析数据集...")
    for _, row in df.iterrows():
        # 转换商品ID为具体类别
        categories = []
        for item_id in row['item_ids']:
            if item_id in id_map:
                categories.append(id_map[item_id]['category'])
        # 添加支付方式特征
        transaction = [f"支付方式_{row['payment_method']}"] + categories
        transactions.append(transaction)
    return transactions


def analyze_associations(transactions: List[list], chunk_size) -> (pd.DataFrame, pd.DataFrame):
    """步骤4：执行关联分析"""
    from mlxtend.preprocessing import TransactionEncoder
    print("正在执行关联分析...")
    # 新增数据编码步骤
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)
    del transactions , te_ary
    gc.collect()

    # 使用分布式FP-Growth
    from distributed_fpgrowth import distributed_fpgrowth
    frequent_itemsets = distributed_fpgrowth(encoded_df,
                                             min_support=0.01,
                                             chunk_size= chunk_size,
                                             max_len=2)  # 限制为二元关联

    # 生成关联规则
    rules = association_rules(frequent_itemsets,
                              metric="confidence",
                              min_threshold=0.6)
    print('过滤有效规则...')
    # 过滤有效规则（支付方式与商品关联）
    payment_rules = rules[
        rules['antecedents'].apply(lambda x: '支付方式_' in next(iter(x))) &
        rules['consequents'].apply(lambda x: '支付方式_' not in next(iter(x)))
        ]
    return frequent_itemsets, payment_rules


def export_results(fis: pd.DataFrame, rules: pd.DataFrame, analysis_type: str) -> None:
    """步骤5：导出结果文件"""
    print("导出分析结果...")
    output_dir = Path("../data/output")
    output_dir.mkdir(exist_ok=True)

    fis.to_csv(output_dir / f"{analysis_type}_frequent_itemsets.csv", index=False)
    rules.to_csv(output_dir / f"{analysis_type}_association_rules.csv", index=False)


# ----------------------
# 分析展示函数（步骤6-7）
# ----------------------
def analyze_high_value_payments(df: pd.DataFrame, id_map: dict) -> pd.Series:
    """步骤6：高价值商品支付分析"""
    # 获取高价值商品ID
    high_value_ids = {id for id, info in id_map.items() if info['price'] > 5000}

    # 展开item_ids并过滤高价值商品
    exploded_df = df.explode('item_ids')
    high_value_items = exploded_df[exploded_df['item_ids'].isin(high_value_ids)]

    # 统计支付方式（每个高价值商品单独计数）
    return high_value_items['payment_method'].value_counts(normalize=True)


def load_results(analysis_type: str) -> (pd.DataFrame, pd.DataFrame):
    """加载已存储的关联分析结果"""
    output_dir = Path("../data/output")

    def _safe_load(path: Path) -> pd.DataFrame:
        if not path.exists():
            print(f"警告: 文件 {path.name} 不存在")
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            # 转换itemsets列为实际集合类型
            if 'itemsets' in df.columns:
                df['itemsets'] = df['itemsets'].apply(
                    lambda x: frozenset(eval(x)) if isinstance(x, str) else x
                )
            return df
        except Exception as e:
            print(f"加载文件错误 {path.name}: {str(e)}")
            return pd.DataFrame()

    print("正在加载存储的分析结果...")
    fis_path = output_dir / f"{analysis_type}_frequent_itemsets.csv"
    rules_path = output_dir / f"{analysis_type}_association_rules.csv"

    return _safe_load(fis_path), _safe_load(rules_path)


def print_summary(fis: pd.DataFrame, rules: pd.DataFrame, payment_dist: pd.Series) -> None:
    """控制台输出分析摘要"""

    print("\n" + "=" * 50)
    print("分析结果摘要")
    print("=" * 50)

    # 符合要求的频繁项集统计
    valid_fis = fis[
        fis['itemsets'].apply(lambda x: any(item.startswith('支付方式_') for item in x)) &
        (fis['itemsets'].apply(len) >= 2)
        ]

    print("\n符合要求的频繁项集统计:")
    print(f"总数量: {len(valid_fis)}")
    if not valid_fis.empty:
        print(f"最高支持度: {valid_fis['support'].max():.3f}")
        print(f"平均支持度: {valid_fis['support'].mean():.3f}")

        # 显示前5项集
        top_fis = valid_fis.sort_values('support', ascending=False).head(5)
        print("\nTop 5频繁项集:")
        for i, (_, row) in enumerate(top_fis.iterrows()):
            items = list(row['itemsets'])
            payment = next(item for item in items if item.startswith('支付方式_'))
            category = next(item for item in items if not item.startswith('支付方式_'))
            print(f"{i + 1}. {payment.replace('支付方式_', '')} → {category}")
            print(f"   支持度: {row['support']:.3f}")
    else:
        print("未找到符合要求的频繁项集")


    # 关联规则统计
    if len(rules) > 0:
        print(f"\n关联规则总数量: {len(rules)}")
        print(f"最高支持度: {rules['support'].max():.3f}")
        print(f"平均置信度: {rules['confidence'].mean():.3f}")

        # 显示前3条强规则
        print("\nTop 3强关联规则:")
        top_rules = rules.sort_values(['lift', 'support'], ascending=False).head(3)
        for i, (_, rule) in enumerate(top_rules.iterrows()):
            ant = list(rule['antecedents'])[0].replace("支付方式_", "")
            cons = list(rule['consequents'])[0]
            print(f"{i + 1}. {ant} → {cons}")
            print(f"   支持度: {rule['support']:.3f} | 置信度: {rule['confidence']:.3f} | 提升度: {rule['lift']:.2f}")
    else:
        print("\n未找到符合条件的关联规则")

    # 始终显示高价值支付分布
    print("\n高价值商品支付方式分布:")
    if len(payment_dist) == 0:
        print("  未找到高价值商品交易记录")
    else:
        for method, ratio in payment_dist.items():
            print(f"  - {method}: {ratio * 100:.1f}%")


def visualize_analysis(rules: pd.DataFrame, payment_dist: pd.Series) -> None:
    """可视化展示"""
    try:

        if len(rules) == 0:
            print("\n无符合条件的关联规则，无法进行可视化")
            return

        plt.figure(figsize=(15, 6))

        # 子图1：关联规则展示
        plt.subplot(121)
        top_rules = rules.sort_values('support', ascending=False).head(10)
        x = range(len(top_rules))
        plt.bar(x, top_rules['support'], width=0.4, label='支持度')
        plt.bar([i + 0.4 for i in x], top_rules['confidence'], width=0.4, label='置信度')
        plt.xticks([i + 0.2 for i in x],
                   [f"{a}→{c}" for a, c in zip(top_rules['antecedents'], top_rules['consequents'])],
                   rotation=45, ha='right')
        plt.title('Top 10关联规则指标')
        plt.legend()

        # 子图2：高价值支付分布
        plt.subplot(122)
        if len(payment_dist) > 0:
            payment_dist.plot(kind='pie', autopct='%1.1f%%')
            plt.title('高价值商品支付方式分布')
            plt.ylabel('')
        else:
            plt.text(0.5, 0.5, '无高价值商品数据', ha='center')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    except pd.errors.EmptyDataError:
        print("\n关联规则文件为空，请检查分析流程")
    except Exception as e:
        print(f"\n数据加载异常: {str(e)}")


# ----------------------
# 执行入口
# ----------------------
if __name__ == "__main__":
    # 执行核心流程
    df = load_payment_data("../data/30G_data_new_processed")
    product_map = create_product_mapping("../data/product_catalog.json")
    # transactions = build_association_dataset(df, product_map)
    # del df, product_map  # 释放内存
    # gc.collect()
    # fis, rules = analyze_associations(transactions, chunk_size=800_000)
    # export_results(fis, rules, "payment_and_category")

    # 执行分析展示
    loaded_fis, loaded_rules = load_results("payment_and_category")
    payment_dist = analyze_high_value_payments(df, product_map)
    print_summary(loaded_fis, loaded_rules, payment_dist)
    visualize_analysis(loaded_rules, payment_dist)