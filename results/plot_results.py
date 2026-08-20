#!/usr/bin/env python3
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# スタイル設定
sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 12

def load_data(results_dir="results"):
    """ results ディレクトリ配下の JSON をロードして DataFrame 化 """
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        print(f"⚠️ '{results_dir}' に結果データが見つかりません。")
        return None
    
    data = []
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data.append(json.load(fp))
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return pd.DataFrame(data)

def plot_ppl(df, output_dir="plots"):
    """ Perplexity (PPL) の比較棒グラフ（低い方が高性能） """
    ppl_df = df[df["eval_type"] == "ppl"].copy() if "eval_type" in df.columns else df.dropna(subset=["ppl"])
    if ppl_df.empty:
        return

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=ppl_df,
        x="method",
        y="ppl",
        palette="viridis"
    )
    plt.title("Perplexity Comparison (Lower is Better)", fontsize=14, fontweight="bold")
    plt.xlabel("Quantization Method", fontsize=12)
    plt.ylabel("Perplexity (PPL)", fontsize=12)
    
    # バーの上に数値を表示
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                        textcoords='offset points')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ppl_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 PPL のグラフを保存しました: {out_path}")

def plot_niah(df, output_dir="plots"):
    """ NIAH (Needle In A Haystack) 正解率の比較棒グラフ（高い方が高性能） """
    niah_df = df[df["eval_type"] == "niah"].copy() if "eval_type" in df.columns else df.dropna(subset=["accuracy"])
    if niah_df.empty:
        return

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=niah_df,
        x="method",
        y="accuracy",
        palette="magma"
    )
    plt.title("Needle In A Haystack Accuracy (Higher is Better)", fontsize=14, fontweight="bold")
    plt.xlabel("Quantization Method", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.annotate(f'{height:.2%}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                        textcoords='offset points')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "niah_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 NIAH のグラフを保存しました: {out_path}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_ppl(df)
        plot_niah(df)
        print("✅ すべてのグラフ描画が完了しました。")