"""
short/inter/long 각 클러스터의 ClueGO 결과(functional_analysis/cluego_result/*.xls,
p-value cutoff 적용된 버전)에서 공통으로 나타나는 GO term 중,
인터페론/항바이러스 관련 23개 term만 골라 그 term에 연결된 유전자 +
3개 샘플 모두에서 발현하는 유전자만 추출한다.

DBN-DDE_inputs/{short,inter,long}_input.csv를 이걸로 덮어쓴다 (25개 유전자).
extract_common_genes.py(단순 교집합, 36개 유전자)는 예전 버전이라 더 이상 안 씀.

출력 형식: 가로(열)=POD, 세로(행)=유전자
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "DBN-DDE_inputs"

CLUEGO_SOURCES = {
    "short": ROOT / "functional_analysis/cluego_result/cluego_table_short_c2.xls",
    "inter": ROOT / "functional_analysis/cluego_result/cluego_table_inter_c1.xls",
    "long":  ROOT / "functional_analysis/cluego_result/cluego_table_long_c2.xls",
}

EXPR_SOURCES = {
    "short": ROOT / "functional_analysis/short/cluster_lists/short_cluster2_expr.csv",
    "inter": ROOT / "functional_analysis/inter/cluster_lists/inter_cluster1_expr.csv",
    "long":  ROOT / "functional_analysis/long/cluster_lists/long_cluster2_expr.csv",
}

# short_c2 / inter_c1 / long_c2 공통 GO term(p-value cutoff 적용 후 26개) 중,
# 인터페론/항바이러스 관련 23개
# (natural killer cell mediated immunity, regulation of nuclease activity,
#  interleukin-27-mediated signaling pathway 3개 term은 제외)
SELECTED_GO_IDS = {
    "GO:0001819", "GO:0002831", "GO:0009615", "GO:0019079", "GO:0032481",
    "GO:0032608", "GO:0032648", "GO:0032728", "GO:0034340", "GO:0035455",
    "GO:0039528", "GO:0045069", "GO:0045071", "GO:0045088", "GO:0045824",
    "GO:0048525", "GO:0050792", "GO:0051607", "GO:0060700", "GO:0098586",
    "GO:0140546", "GO:1900246", "GO:1903900",
}


def genes_from_cell(cell: str) -> set[str]:
    # ClueGO cell format: "[GENE1, GENE2, GENE3]"
    return {g.strip() for g in cell.strip("[]").split(",") if g.strip()}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1) 샘플별로: 선택된 51개 term에 연결된 유전자들의 합집합
    term_genes_by_sample = {}
    for name, path in CLUEGO_SOURCES.items():
        df = pd.read_excel(path)
        df["ID"] = df["ID"].astype(str).str.strip()
        selected = df[df["ID"].isin(SELECTED_GO_IDS)]
        genes = set()
        for cell in selected["Associated Genes Found"]:
            genes |= genes_from_cell(cell)
        term_genes_by_sample[name] = genes
        print(f"{name}: term-associated genes = {len(genes)}")

    # 2) 발현 데이터(유전자 x POD)를 로드하고, term-연관 유전자만 우선 필터
    expr_dfs = {
        name: pd.read_csv(path, index_col="GeneName")
        for name, path in EXPR_SOURCES.items()
    }
    filtered_by_sample = {}
    for name in EXPR_SOURCES:
        present = set(expr_dfs[name].index) & term_genes_by_sample[name]
        filtered_by_sample[name] = present
        print(f"{name}: term genes also in expr data = {len(present)}")

    # 3) 3개 샘플 모두에서 발현하는 유전자만 최종 선택
    common_genes = sorted(
        filtered_by_sample["short"] & filtered_by_sample["inter"] & filtered_by_sample["long"]
    )
    print(f"\n최종 공통 유전자 수 (인터페론/항바이러스 term 필터): {len(common_genes)}")
    print(common_genes)

    # 4) 저장 (DBN-DDE_inputs/{short,inter,long}_input.csv -- extract_common_genes.py의
    #    단순 교집합 결과를 대체하는 현재 기본 파이프라인)
    for name, expr_df in expr_dfs.items():
        filtered = expr_df.loc[common_genes]
        out_path = OUTPUT_DIR / f"{name}_input.csv"
        filtered.to_csv(out_path)
        print(f"{name}: {filtered.shape[0]} genes x {filtered.shape[1]} POD -> {out_path}")


if __name__ == "__main__":
    main()
