"""
ログから results/ai-l40s の JSON を再構築する復元ツール。

背景:
  2026-08-26 の full_eval (job 373124) の結果 JSON が rm 等で消えた場合でも、
  各評価スクリプトは全スカラー指標を標準出力(log)にも出力している。
  本スクリプトはそのログを走査し、元の eval スクリプトと同一スキーマの
  JSON を再生成する。

使い方:
  python results/rebuild_from_log.py logs/full_eval_XXXXXX.log [logs/missing_eval_YYYYYY.log ...]
    --outdir results/ai-l40s   # 既定
    --force                    # 既存JSONも上書き(既定は既存スキップ)

注意(ログに出ていない情報は復元できない):
  - NIAH の各試行 "response" 文 → "(response not present in log)" ダミー
  - LongBench の "question"/"references"、prediction の81文字目以降
  - これら以外の数値指標はログと同等に復元される
"""

import argparse
import ast
import json
import os
import re


def save_json(outdir, filename, data, force, report):
    path = os.path.join(outdir, filename)
    if os.path.exists(path) and not force:
        report.append(f"SKIP (exists): {filename}")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    report.append(f"WRITE: {filename}")


def safe_model(model_name):
    return model_name.replace("/", "_")


def rebuild_from_log(log_path, outdir, force, report):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    section = None          # "compression" | "speed"
    current_model = None
    lb_ctx = None           # LongBench 集計用
    niah_ctx = None         # NIAH 集計用
    sanity_ctx = None       # Sanity 集計用
    theo_ctx = None         # 理論フットプリント集計用

    for line in lines:
        m = re.match(r"^#+\s*$", line)
        if m:
            continue

        mo = re.match(r"^ Model: (\S+)\s*$", line)
        if mo:
            current_model = mo.group(1)
            continue

        # ---- セクションヘッダ検出 ----
        if "=== Measuring KV Cache Compression" in line:
            section = "compression"
            continue
        if "=== Measuring Inference Speed" in line:
            section = "speed"
            continue

        # ---- Compression / Speed: Result: {dict} ----
        mr = re.match(r"^Result: (\{.*\})\s*$", line.strip())
        if mr and section in ("compression", "speed"):
            data = ast.literal_eval(mr.group(1))
            fn = f"{safe_model(data['model_name'])}_{data['method']}_{section}.json"
            save_json(outdir, fn, data, force, report)
            section = None
            continue

        # ---- PPL: Final Result [model | method]: float ----
        mp = re.match(r"^Final Result \[(.*?) \| (\w+)\]: ([\d.]+)\s*$", line.strip())
        if mp:
            model, method, ppl = mp.group(1), mp.group(2), float(mp.group(3))
            data = {
                "model_name": model,
                "method": method,
                "stride": 512,
                "seq_len": 2048,
                "perplexity": ppl,
            }
            save_json(outdir, f"{safe_model(model)}_{method}_ppl.json", data, force, report)
            continue

        # ---- Fidelity: Final Result [model | method]: {dict} ----
        mf = re.match(r"^Final Result \[(.*?) \| (\w+)\]: (\{.*\})\s*$", line.strip())
        if mf:
            model, method = mf.group(1), mf.group(2)
            metrics = ast.literal_eval(mf.group(3))
            data = {"model_name": model, "method": method, "seq_len": 1024, **metrics}
            save_json(outdir, f"{safe_model(model)}_{method}_fidelity.json", data, force, report)
            continue

        # ---- NIAH ----
        mn = re.match(r"^=== Needle In A Haystack \| Model: (.*?) \| Context: (\d+) \| Method: (\w+)", line)
        if mn:
            niah_ctx = {
                "model": mn.group(1), "context_len": int(mn.group(2)), "method": mn.group(3),
                "trials": [],
            }
            continue
        if niah_ctx is not None:
            mt = re.match(r"^\s+depth=(\d+)% trial=(\d+) key=(\d+) success=(True|False)", line)
            if mt:
                niah_ctx["trials"].append({
                    "depth": int(mt.group(1)) / 100.0,
                    "trial": int(mt.group(2)),
                    "secret_key": int(mt.group(3)),
                    "response": "(response not present in log; rebuilt from log)",
                    "success": mt.group(4) == "True",
                })
                continue
            ms = re.match(r"^=== NIAH success rate: (\d+)/(\d+) = ([\d.]+)%", line)
            if ms:
                n_success, n = int(ms.group(1)), int(ms.group(2))
                data = {
                    "model_name": niah_ctx["model"],
                    "method": niah_ctx["method"],
                    "context_len": niah_ctx["context_len"],
                    "depths": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "trials_per_depth": 1,
                    "n_trials": n,
                    "n_success": n_success,
                    "niah_success_rate": float(ms.group(3)) / 100.0,
                    "trials": niah_ctx["trials"],
                }
                fn = f"{safe_model(niah_ctx['model'])}_{niah_ctx['method']}_niah.json"
                save_json(outdir, fn, data, force, report)
                niah_ctx = None
            continue

        # ---- LongBench ----
        ml = re.match(r"^=== LongBench \(Qasper QA-F1\) \| Model: (.*?) \| Method: (\w+)", line)
        if ml:
            lb_ctx = {"model": ml.group(1), "method": ml.group(2), "samples": []}
            continue
        if lb_ctx is not None:
            mls = re.match(r"^\s+\[(\d+)\] F1=([\d.]+) \| pred: (.*)$", line)
            if mls:
                pred_raw = mls.group(3)
                try:
                    pred = ast.literal_eval(pred_raw)
                except (ValueError, SyntaxError):
                    pred = pred_raw
                lb_ctx["samples"].append({
                    "index": int(mls.group(1)),
                    "question": "(question not present in log; rebuilt from log)",
                    "prediction": pred,
                    "references": "(references not present in log; rebuilt from log)",
                    "qa_f1": float(mls.group(2)),
                })
                continue
            mm = re.match(r"^=== LongBench Qasper mean QA-F1 over (\d+) samples: ([\d.]+)", line)
            if mm:
                data = {
                    "model_name": lb_ctx["model"],
                    "method": lb_ctx["method"],
                    "task": "qasper",
                    "num_samples": int(mm.group(1)),
                    "max_ctx_tokens": 6144,
                    "longbench_qasper_f1": float(mm.group(2)),
                    "samples": lb_ctx["samples"],
                }
                fn = f"{safe_model(lb_ctx['model'])}_{lb_ctx['method']}_longbench.json"
                save_json(outdir, fn, data, force, report)
                lb_ctx = None
            continue

        # ---- Sanity Check ----
        msc = re.match(r"^=== Sanity Check \| Model: (.*?) \| Device:", line)
        if msc:
            sanity_ctx = {"model": msc.group(1), "results": {}, "last_ids_key": None}
            continue
        if sanity_ctx is not None:
            mpas_g = re.match(r"^\[passthrough\] generated: (\[.*\])", line)
            m7_g = re.match(r"^\[(\w+) 7bit\] generated: (\[.*\])", line)
            mpas_i = re.match(r"^\[passthrough\] identical: (True|False) \(match ([\d.]+)%\)", line)
            m7_r = re.match(r"^\[(\w+) 7bit\] match rate: ([\d.]+)%", line)
            mpass = re.match(r"^=== Sanity Check Overall: (PASS|FAIL)", line)
            if mpas_g:
                sanity_ctx["pt_ids"] = ast.literal_eval(mpas_g.group(1))
                continue
            if m7_g:
                sanity_ctx["last_ids_key"] = m7_g.group(1)
                sanity_ctx.setdefault("ids7", {})[m7_g.group(1)] = ast.literal_eval(m7_g.group(2))
                continue
            if mpas_i:
                sanity_ctx["results"]["passthrough"] = {
                    "generated_ids": sanity_ctx.get("pt_ids", []),
                    "identical_to_fp16": mpas_i.group(1) == "True",
                    "token_match_rate": float(mpas_i.group(2)) / 100.0,
                    "criterion": "identical_to_fp16 == True",
                }
                continue
            if m7_r:
                method = m7_r.group(1)
                sanity_ctx["results"][f"{method}_7bit"] = {
                    "generated_ids": sanity_ctx.get("ids7", {}).get(method, []),
                    "token_match_rate_vs_fp16": float(m7_r.group(2)) / 100.0,
                    "criterion": "token_match_rate >= 0.5 (near-lossless at 7bit)",
                }
                continue
            if mpass:
                overall = mpass.group(1) == "PASS"
                # 7bit 結果が全て揃っていなくても書き出す(ログに応じる)
                data = {"overall_pass": overall, "results": sanity_ctx["results"]}
                fn = f"{safe_model(sanity_ctx['model'])}_sanity.json"
                save_json(outdir, fn, data, force, report)
                sanity_ctx = None
            continue

        # ---- 理論フットプリント ----
        mft = re.match(r"^=== Calculating Analytical Footprints \| Model: (.*?) \| SeqLen: (\d+)", line)
        if mft:
            theo_ctx = {"model": mft.group(1), "seq_len": int(mft.group(2)), "rows": []}
            continue
        if theo_ctx is not None:
            mtb = re.match(
                r"^Method: (\w+)\s+\| Bits:\s*([\d.]+) \| Data:\s*([\d.]+) MB "
                r"\| Meta:\s*([\d.]+) MB \| Total:\s*([\d.]+) MB \| Compression:\s*([\d.]+)x",
                line.strip(),
            )
            if mtb:
                theo_ctx["rows"].append({
                    "method": mtb.group(1),
                    "bits": float(mtb.group(2)),
                    "data": float(mtb.group(3)),
                    "meta": float(mtb.group(4)),
                    "total": float(mtb.group(5)),
                    "ratio": float(mtb.group(6)),
                })
                continue
            if "All analytical footprint results successfully saved" in line:
                fp16_base = next((r["total"] for r in theo_ctx["rows"] if r["method"] == "fp16"), 0.0)
                all_metrics = []
                for r in theo_ctx["rows"]:
                    metrics = {
                        "model_name": theo_ctx["model"],
                        "method": r["method"],
                        "designed_bits": r["bits"],
                        "sequence_length": theo_ctx["seq_len"],
                        "designed_data_mb": r["data"],
                        "designed_meta_mb": r["meta"],
                        "designed_footprint_mb": r["total"],
                        "fp16_baseline_mb": fp16_base,
                        "designed_compression_ratio": r["ratio"],
                        "calculation_type": "analytical",
                    }
                    all_metrics.append(metrics)
                    fn = f"{safe_model(theo_ctx['model'])}_{r['method']}_theoretical.json"
                    save_json(outdir, fn, metrics, force, report)
                sfn = f"{safe_model(theo_ctx['model'])}_all_methods_footprint_summary.json"
                save_json(outdir, sfn, all_metrics, force, report)
                theo_ctx = None
            continue


def main():
    parser = argparse.ArgumentParser(description="Rebuild results/ai-l40s JSONs from eval logs")
    parser.add_argument("logs", nargs="+", help="full_eval / missing_eval などのログファイル")
    parser.add_argument("--outdir", default="results/ai-l40s")
    parser.add_argument("--force", action="store_true", help="既存JSONも上書き(既定:既存はスキップ)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    report = []
    for log in args.logs:
        rebuild_from_log(log, args.outdir, args.force, report)

    for r in report:
        print(r)
    n_write = sum(1 for r in report if r.startswith("WRITE"))
    n_skip = sum(1 for r in report if r.startswith("SKIP"))
    print(f"\nDone: {n_write} rebuilt, {n_skip} skipped (existing). Outdir: {args.outdir}")


if __name__ == "__main__":
    main()
