#!/usr/bin/env python3
import os
import glob
import json
import pandas as pd

def load_results(results_dir="results"):
    """
    results ディレクトリ内の全 JSON ファイルを読み込み、リスト化する
    """
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not json_files:
        print(f"⚠️ '{results_dir}' ディレクトリに結果ファイル (JSON) が見つかりませんでした。")
        return None

    data_list = []
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # ファイル名も情報として保持
                data["file_name"] = os.path.basename(file_path)
                data_list.append(data)
        except Exception as e:
            print(f"❌ ファイルの読み込みエラー ({file_path}): {e}")

    return pd.DataFrame(data_list)


def format_and_display(df):
    """
    データを整形して表示・保存する
    """
    print("\n" + "=" * 80)
    print(" 📊 KV-Cache Quantization Benchmark Summary")
    print("=" * 80)

    # 主要なカラムを優先的に並べ替え（存在するもののみ）
    priority_cols = ["eval_type", "method", "model_name", "ppl", "cossim", "topk_match", "accuracy", "seq_len", "context_len"]
    existing_cols = [col for col in priority_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_cols and col != "file_name"]
    
    formatted_df = df[existing_cols + other_cols]

    # 小数点の表示精度を調整
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    print(formatted_df.to_string(index=False))
    print("=" * 80)

    # CSV として保存
    output_csv = "summary_results.csv"
    formatted_df.to_csv(output_csv, index=False)
    print(f"\n✅ サマリーを '{output_csv}' に保存しました。\n")


if __name__ == "__main__":
    # 1. 結果の読み込み
    df_results = load_results()

    # 2. 整形と表示
    if df_results is not None:
        format_and_display(df_results)