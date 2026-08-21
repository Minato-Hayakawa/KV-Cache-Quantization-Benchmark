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

def load_and_merge_results(results_dir="results/ai-l40s"):
    """
    results フォルダ内の全 JSON ファイルを読み込み、
    model_name と method、そしてベンチマーク（またはファイルの種類）ごとに綺麗にマージする
    """
    json_files = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
    
    if not json_files:
        json_files = glob.glob("results/*.json")
        results_dir = "results"

    if not json_files:
        print(f"⚠️ 結果ファイル (JSON) が見つかりませんでした。")
        return None

    records = {}

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            filename = os.path.basename(file_path)
            data["file_name"] = filename

            # キーの特定：model_name, method に加え、どのベンチマーク/評価ファイルか（filenameのサフィックス等）で分ける場合
            # もし「1つのモデル×手法につき1行にすべてをまとめたい」場合は model_name と method をキーにします。
            # ただし、leval_finance のような別タスクの結果がある場合、混ざると上書きされる可能性があるためタスク別に持たせるのが安全です。
            
            model_name = data.get("model_name", "unknown_model")
            method = data.get("method", "unknown_method")
            
            # ファイル名からベンチマーク名や種類を大まかに判定してキーに含める
            # 例: _leval_finance.json, _longbench.json など
            task_type = "general"
            for benchmark_name in ["leval_finance", "longbench", "niah", "ppl", "speed", "compression", "fidelity"]:
                if benchmark_name in filename:
                    task_type = benchmark_name
                    data["benchmark"] = benchmark_name
                    break

            key = (model_name, method, task_type)

            if key not in records:
                records[key] = {}

            # データを結合
            for k, v in data.items():
                if v is not None:
                    records[key][k] = v

        except Exception as e:
            print(f"❌ ファイルの読み込みエラー ({file_path}): {e}")

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(list(records.values()))

def format_and_save_summary(df):
    """
    データを整形してコンソールに表示し、CSVとして保存する
    """
    print("\n" + "=" * 80)
    print(" 📊 KV-Cache Quantization Benchmark Summary (Task-aware Merged)")
    print("=" * 80)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    print(df.head(10))
    print("=" * 80)

    output_csv = "summary_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ タスク別に整理されたサマリーデータを '{output_csv}' に保存しました。\n")

def generate_plots(df, output_dir="plots"):
    """
    主要な指標の比較グラフを生成する
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. PPL のグラフ
    if "perplexity" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df.dropna(subset=["perplexity"]), x="method", y="perplexity", hue="model_name", palette="viridis")
        plt.title("Perplexity (PPL) Comparison (Lower is Better)", fontsize=14, fontweight="bold")
        plt.xlabel("Quantization Method", fontsize=12)
        plt.ylabel("Perplexity", fontsize=12)
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ppl_comparison.png"), dpi=300)
        plt.close()

    # 2. 圧縮率のグラフ
    if "compression_ratio" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df.dropna(subset=["compression_ratio"]), x="method", y="compression_ratio", hue="model_name", palette="magma")
        plt.title("Compression Ratio Comparison", fontsize=14, fontweight="bold")
        plt.xlabel("Quantization Method", fontsize=12)
        plt.ylabel("Compression Ratio", fontsize=12)
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "compression_comparison.png"), dpi=300)
        plt.close()

    # 3. 速度グラフ
    if "tokens_per_second" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df.dropna(subset=["tokens_per_second"]), x="method", y="tokens_per_second", hue="model_name", palette="coolwarm")
        plt.title("Inference Speed Comparison (Higher is Better)", fontsize=14, fontweight="bold")
        plt.xlabel("Quantization Method", fontsize=12)
        plt.ylabel("Tokens / Second", fontsize=12)
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "speed_comparison.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    df_results = load_and_merge_results()
    if df_results is not None and not df_results.empty:
        format_and_save_summary(df_results)
        generate_plots(df_results)
        print("✅ すべての集計とグラフ描画が完了しました！")
    else:
        print("❌ 処理できるデータが見つかりませんでした。")