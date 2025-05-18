"""
商品大类关联分析流程
"""
import os
import json
from typing import Dict

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import gc


# ----------------------
# 核心功能函数（步骤1-5）
# ----------------------
def load_transaction_data(data_path: str) -> pd.DataFrame:
    """步骤1：加载交易数据"""
    df = pd.read_parquet(data_path, engine='pyarrow', columns=['user_id', 'item_ids'])
    print(f"总记录数: {len(df)}")
    print(f"唯一用户数: {df['user_id'].nunique()}")
    df.drop(columns=['user_id'], inplace=True)
    return df


def create_category_mapping() -> Dict[str, str]:
    """步骤2：创建商品大类映射规则"""
    return {
        # 电子产品大类
        "智能手机": "电子产品", "笔记本电脑": "电子产品", "平板电脑": "电子产品", "智能手表": "电子产品",
        "耳机": "电子产品", "音响": "电子产品", "相机": "电子产品", "摄像机": "电子产品", "游戏机": "电子产品",
        # 服装大类
        "上衣": "服装", "裤子": "服装", "裙子": "服装", "内衣": "服装", "鞋子": "服装", "帽子": "服装", "手套": "服装",
        "围巾": "服装", "外套": "服装",
        # 食品大类
        "零食": "食品", "饮料": "食品", "调味品": "食品", "米面": "食品", "水产": "食品", "肉类": "食品",
        "蛋奶": "食品", "水果": "食品", "蔬菜": "食品",
        # 家居大类
        "家具": "家居", "床上用品": "家居", "厨具": "家居", "卫浴用品": "家居",
        # 办公大类
        "文具": "办公", "办公用品": "办公",
        # 运动户外大类
        "健身器材": "运动户外", "户外装备": "运动户外",
        # 玩具大类
        "玩具": "玩具", "模型": "玩具", "益智玩具": "玩具",
        # 母婴大类（根据用户图片示例修正）
        "婴儿用品": "母婴", "儿童课外读物": "母婴",
        # 汽车用品大类
        "车载电子": "汽车用品", "汽车装饰": "汽车用品"
    }


def build_id_category_map(catalog_path: str) -> Dict[int, str]:
    """步骤3：创建ID到类别的映射"""
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)

    id_map = {}
    for item in catalog['products']:
        try:
            id_map[int(item['id'])] = item['category']
        except (KeyError, ValueError) as e:
            print(f"商品数据异常：{item}，错误：{str(e)}")
    return id_map


def enrich_categories(df: pd.DataFrame,
                      id_map: Dict[int, str],
                      cat_map: Dict[str, str]) -> pd.DataFrame:
    """步骤4：丰富商品类别信息"""

    def _convert_items(item_ids):
        categories = []
        for item_id in item_ids:
            try:
                sub_cat = id_map.get(int(item_id), "未知")
                main_cat = cat_map.get(sub_cat, "其他")
                categories.extend([f"{main_cat}_{sub_cat}", main_cat])
            except (ValueError, TypeError):
                continue
        return list(set(categories))

    print("转换商品ID为类别...")
    df['categories'] = df['item_ids'].apply(_convert_items)
    df.drop(columns=['item_ids'], inplace=True)
    return df


def generate_encoded_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """步骤5：生成编码后的交易矩阵"""
    print("生成交易矩阵...")
    transactions = df['categories'].tolist()

    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

    # 内存清理
    del df, te_ary, transactions
    gc.collect()
    return encoded_df


# ----------------------
# 关联分析函数（步骤6-8）
# ----------------------
def mine_association_rules(encoded_df: pd.DataFrame,
                           min_support: float = 0.02,
                           chunk_size: int = 100_000) -> (pd.DataFrame, pd.DataFrame):
    """返回频繁项集和关联规则"""
    from distributed_fpgrowth import distributed_fpgrowth

    frequent_itemsets = distributed_fpgrowth(
        encoded_df,
        min_support=min_support,
        chunk_size=chunk_size
    )
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    return frequent_itemsets, rules


def filter_hierarchical_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """步骤7：过滤层级关系规则"""

    def _is_valid(rule):
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])

        # 过滤大类-子类关系
        for cons in consequents:
            if '_' not in cons:
                for ant in antecedents:
                    if '_' in ant and ant.split('_')[0] == cons:
                        return False

        for ant in antecedents:
            if '_' not in ant:
                for cons in consequents:
                    if '_' in cons and cons.split('_')[0] == ant:
                        return False
        return True

    return rules[rules.apply(_is_valid, axis=1)]


def format_rule_items(rules: pd.DataFrame) -> pd.DataFrame:
    """步骤8：格式化规则展示"""

    def _format(itemset):
        formatted = []
        for item in itemset:
            if '_' in item:
                main, sub = item.split('_', 1)
                formatted.append(f"{sub}（{main}）")
            else:
                formatted.append(item)
        return frozenset(formatted)

    rules['antecedents'] = rules['antecedents'].apply(_format)
    rules['consequents'] = rules['consequents'].apply(_format)
    return rules


# ----------------------
# 分析展示函数（步骤9-10）
# ----------------------
def analyze_specific_category(rules: pd.DataFrame, target: str) -> pd.DataFrame:
    """步骤9：分析指定品类关联规则"""
    return rules[rules.apply(
        lambda x: target in x['antecedents'] or target in x['consequents'],
        axis=1
    )]


def visualize_category_rules(rules: pd.DataFrame, target: str):
    """步骤10：可视化品类关联规则"""
    if rules.empty:
        print(f"没有有效的{target}关联规则，无法进行可视化")
        return

    plt.figure(figsize=(15, 6))

    # 规则指标分布
    plt.subplot(121)
    plt.scatter(rules['support'], rules['confidence'],
                c=rules['lift'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='提升度')
    plt.xlabel('支持度')
    plt.ylabel('置信度')
    plt.title(f'{target}关联规则分布')

    # 规则网络图
    plt.subplot(122)
    try:
        G = nx.from_pandas_edgelist(
            rules.nlargest(10, 'lift'),
            'antecedents',
            'consequents',
            edge_attr=['lift', 'confidence']
        )
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=1500)
        nx.draw_networkx_edges(G, pos, edge_color='grey', width=rules['lift'] * 0.5)
        nx.draw_networkx_labels(G, pos, font_size=8,
                                labels={n: '\n'.join(n) for n in G.nodes()})
        plt.title("Top10关联规则网络")
        plt.axis('off')
    except Exception as e:
        print(f"网络图生成失败: {str(e)}")

    plt.tight_layout()
    plt.show()


def export_results(fis: pd.DataFrame, rules: pd.DataFrame, analysis_type: str) -> None:
    """导出频繁项集和关联规则到CSV"""
    output_dir = Path("../data/output")
    output_dir.mkdir(exist_ok=True)

    # 转换集合类型为字符串
    fis = fis.copy()
    if 'itemsets' in fis.columns:
        fis['itemsets'] = fis['itemsets'].apply(lambda x: ', '.join(map(str, x)))

    rules = rules.copy()
    for col in ['antecedents', 'consequents']:
        rules[col] = rules[col].apply(lambda x: ', '.join(map(str, x)))

    fis.to_csv(output_dir / f"{analysis_type}_frequent_itemsets.csv", index=False)
    rules.to_csv(output_dir / f"{analysis_type}_association_rules.csv", index=False)

def load_processed_results(analysis_type: str) -> (pd.DataFrame, pd.DataFrame):
    output_dir = Path("../data/output")
    def convert_to_frozenset(s: str) -> frozenset:
        return frozenset(s.split(', ')) if pd.notna(s) else frozenset()
    fis = pd.read_csv(output_dir / f"{analysis_type}_frequent_itemsets.csv")
    if 'itemsets' in fis.columns:
        fis['itemsets'] = fis['itemsets'].apply(convert_to_frozenset)
    rules = pd.read_csv(output_dir / f"{analysis_type}_association_rules.csv")
    for col in ['antecedents', 'consequents']:
        rules[col] = rules[col].apply(convert_to_frozenset)
    return fis, rules


def is_valid_itemset(itemset: frozenset) -> bool:
    """判断频繁项集是否符合要求：大小≥2且不包含大类和其子类"""
    # 条件1：项集大小必须≥2
    if len(itemset) < 2:
        return False

    main_cats = set()  # 存储纯大类（无下划线）
    sub_main_cats = set()  # 存储子类的大类部分

    for item in itemset:
        if '_' in item:
            main_part, _ = item.split('_', 1)
            sub_main_cats.add(main_part)
        else:
            main_cats.add(item)

    # 条件2：大类与子类的大类部分不能有交集
    return main_cats.isdisjoint(sub_main_cats)


def filter_frequent_itemsets(fis_df: pd.DataFrame) -> pd.DataFrame:
    """从DataFrame中过滤符合条件的频繁项集"""
    return fis_df[fis_df['itemsets'].apply(is_valid_itemset)]


def summarize_filtered_fis(filtered_fis: pd.DataFrame) -> None:
    """输出过滤后的频繁项集信息到控制台"""
    if filtered_fis.empty:
        print("没有符合条件的频繁项集。")
        return

    print(f"\n过滤后的频繁项集总数: {len(filtered_fis)}")

    # 显示前5个项集的信息
    print("\n前5个符合条件的频繁项集：")
    for idx, row in filtered_fis.head().iterrows():
        itemset = row['itemsets']
        support = row['support']

        # 提取所有涉及的大类
        main_cats = set()
        for item in itemset:
            main = item.split('_')[0] if '_' in item else item
            main_cats.add(main)

        print(f"项集: {', '.join(itemset)}")
        print(f"涉及大类: {', '.join(main_cats)}, 支持度: {support:.4f}\n")

    # 统计大类组合的出现次数
    from collections import defaultdict
    combination_counts = defaultdict(int)
    for itemset in filtered_fis['itemsets']:
        main_cats = set()
        for item in itemset:
            main = item.split('_')[0] if '_' in item else item
            main_cats.add(main)
        combination = frozenset(main_cats)
        combination_counts[combination] += 1

    # 按出现次数降序排列
    sorted_combinations = sorted(combination_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n大类组合统计（Top10）：")
    for comb, count in sorted_combinations[:10]:
        print(f"组合: {set(comb)} - 出现次数: {count}")



def process_filtering_fis(fis_df):
    """从CSV文件加载并处理过滤"""
    # 执行过滤
    valid_fis = filter_frequent_itemsets(fis_df)
    # 输出总结信息
    summarize_filtered_fis(valid_fis)


# ----------------------
# 执行入口
# ----------------------
if __name__ == "__main__":
    # 核心分析流程
    # raw_df = load_transaction_data("../data/30G_data_new_processed")
    # cat_mapping = create_category_mapping()
    # id_category_map = build_id_category_map("../data/product_catalog.json")
    # enriched_df = enrich_categories(raw_df, id_category_map, cat_mapping)
    # encoded_trans = generate_encoded_transactions(enriched_df)
    # del raw_df, enriched_df
    # gc.collect()
    # # 关联分析与数据清洗
    # frequent_itemsets, initial_rules = mine_association_rules(encoded_trans, chunk_size = 800_000)
    # filtered_rules = filter_hierarchical_rules(initial_rules)
    # formatted_rules = format_rule_items(filtered_rules)
    #
    # # 导出清洗后的结果
    # export_results(frequent_itemsets, formatted_rules, "category")


    # 加载清洗后的数据
    fis_df, rules_df = load_processed_results("category")

    process_filtering_fis(fis_df)

    # 结果展示
    print("\n关联规则摘要：")
    print(f"总规则数: {len(rules_df)}")
    if len(rules_df) > 0:
        print(f"最大支持度: {rules_df['support'].max():.3f}")
        print(f"平均置信度: {rules_df['confidence'].mean():.3f}")

        # 打印前10条高质量规则
        print("Top 10 强关联规则：")
        top_rules = rules_df.sort_values(by=['lift', 'confidence'], ascending=False).head(10)
        for i, (_, row) in enumerate(top_rules.iterrows()):
            print(f"{i + 1}. {set(row['antecedents'])} → {set(row['consequents'])}")
            print(f"   支持度: {row['support']:.3f} | 置信度: {row['confidence']:.3f} | 提升度: {row['lift']:.2f}\n")
    else:
        print("没有有效的关联规则。")

    electronics_rules = analyze_specific_category(rules_df, "电子产品")
    print(f"电子产品相关规则总数: {len(electronics_rules)}")

    if not electronics_rules.empty:
        print("\nTop 3电子产品规则：")
        top_rules = electronics_rules.nlargest(3, 'lift')
        for i, (_, row) in enumerate(top_rules.iterrows()):
            print(f"{i + 1}. {set(row['antecedents'])} → {set(row['consequents'])}")
            print(f"   支持度: {row['support']:.3f} | 置信度: {row['confidence']:.3f}")
    else:
        print("没有发现有效的电子产品关联规则。")

    # visualize_category_rules(electronics_rules, "电子产品")