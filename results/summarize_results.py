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

# ファイル名サフィックスから判定するタスク種別
TASK_TYPES = ["longbench", "niah", "ppl", "speed", "compression", "fidelity", "theoretical", "sanity"]


def load_and_merge_results(results_dir="results/ai-l40s"):
    """
    results フォルダ内の全 JSON ファイルを読み込み、
    model_name と method、ベンチマーク種別ごとにマージする
    """
    json_files = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)

    if not json_files:
        json_files = glob.glob("results/*.json")
        results_dir = "results"

    if not json_files:
        print("[WARN] 結果ファイル(JSON)が見つかりませんでした。")
        return None

    records = {}

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            filename = os.path.basename(file_path)

            # サマリ系（all_methods_footprint_summary 等）の list 形式は
            # メソッド別 JSON に内容が含まれているので集計対象から外す
            if isinstance(data, list):
                print(f"[INFO] スキップ (list形式のサマリ): {filename}")
                continue

            data["file_name"] = filename

            model_name = data.get("model_name", "unknown_model")
            method = data.get("method", "unknown_method")

            task_type = "general"
            for benchmark_name in TASK_TYPES:
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
            print(f"[ERROR] ファイルの読み込みエラー ({file_path}): {e}")

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(list(records.values()))


def format_and_save_summary(df):
    """
    データを整形してコンソールに表示し、CSVとして保存する
    """
    print("\n" + "=" * 80)
    print(" [SUMMARY] KV-Cache Quantization Benchmark Summary (Task-aware Merged)")
    print("=" * 80)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: "%.4f" % x)

    print(df.head(10))
    print("=" * 80)

    output_csv = "summary_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] タスク別に整理されたサマリーデータを '{output_csv}' に保存しました。\n")


def _bar(df, column, title, ylabel, filename, output_dir, palette="viridis"):
    """共通の棒グラフヘルパー（対象列が無い場合は静かにスキップ）"""
    if column not in df.columns:
        return
    plot_df = df.dropna(subset=[column]).drop_duplicates(subset=["model_name", "method"])
    if plot_df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="method", y=column, hue="model_name", palette=palette)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Quantization Method", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def generate_plots(df, output_dir="plots"):
    """
    主要な指標の比較グラフを生成する（新スキーマ対応）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. PPL
    _bar(
        df, "perplexity",
        "Perplexity (PPL) Comparison (Lower is Better)",
        "Perplexity", "ppl_comparison.png", output_dir,
    )

    # 2. 圧縮率（designed footprint 基準: _theoretical.json / _compression.json 由来）
    _bar(
        df, "designed_compression_ratio",
        "Compression Ratio (Designed Footprint, Higher is Better)",
        "Compression Ratio (x)", "compression_comparison.png", output_dir, palette="magma",
    )

    # 3. 速度
    _bar(
        df, "tokens_per_second",
        "Inference Speed Comparison (Higher is Better)",
        "Tokens / Second", "speed_comparison.png", output_dir, palette="coolwarm",
    )

    # 4. NIAH 成功率
    _bar(
        df, "niah_success_rate",
        "Needle In A Haystack Success Rate (Higher is Better)",
        "Success Rate", "niah_success_rate.png", output_dir, palette="crest",
    )

    # 5. LongBench Qasper QA-F1
    _bar(
        df, "longbench_qasper_f1",
        "LongBench Qasper QA-F1 (Higher is Better)",
        "QA-F1", "longbench_f1.png", output_dir, palette="flare",
    )

    # 6. 忠実度（logit / KV の cos-sim を並べて表示）
    fidelity_cols = [
        ("logit_cosine_similarity", "Logit Cosine Similarity"),
        ("kv_cosine_similarity", "KV Cosine Similarity"),
    ]
    available = [(c, t) for c, t in fidelity_cols if c in df.columns]
    if available:
        plot_df = df.dropna(subset=[c for c, _ in available]).drop_duplicates(
            subset=["model_name", "method"]
        )
        if not plot_df.empty:
            fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 6), sharey=True)
            if len(available) == 1:
                axes = [axes]
            for ax, (col, title) in zip(axes, available):
                sns.barplot(
                    data=plot_df, x="method", y=col, hue="model_name",
                    palette="rocket", ax=ax, legend=(ax is axes[-1]),
                )
                ax.set_title(title, fontsize=13, fontweight="bold")
                ax.set_xlabel("Quantization Method", fontsize=11)
                ax.set_ylabel("Cosine Similarity" if ax is axes[0] else "", fontsize=11)
            fig.suptitle("Fidelity Comparison (Higher is Better)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fidelity_comparison.png"), dpi=300)
            plt.close(fig)


if __name__ == "__main__":
    df_results = load_and_merge_results()
    if df_results is not None and not df_results.empty:
        format_and_save_summary(df_results)
        generate_plots(df_results)
        print("[OK] すべての集計とグラフ描画が完了しました。")
    else:
        print("[ERROR] 処理できるデータが見つかりませんでした。")
