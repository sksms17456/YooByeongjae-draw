"""Auto improvement loop for mosaic.

Loop:
  1. mosaic.py with current params
  2. score.py
  3. if total >= target: success exit
  4. if iter >= max_iter or stale 2x: stop, return best
  5. identify weakest 1-2 dimensions
  6. apply prescription, repeat
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils import OUTPUT_DIR, PROJECT_ROOT, ensure_dirs

PRESCRIPTION = {
    # weakest dim -> (param, delta, max_or_min, kind)
    "ssim": ("grid", 20, 150, "max"),
    "perceptual": ("blend", 0.1, 0.6, "max"),
    "color": ("blend", 0.05, 0.6, "max"),
    "diversity": ("usage_penalty", 0.5, 5.0, "max"),
    "adjacency": ("neighbor_penalty", 1.0, 5.0, "max"),
    "sharpness": ("__user_action", 0, 0, "user"),  # curate strengthen
    "identifiability": ("__user_action", 0, 0, "user"),
}


def _weakest(normalized: dict) -> list[str]:
    """Return up to 2 weakest dimensions (lowest normalized score)."""
    items = sorted(normalized.items(), key=lambda kv: kv[1])
    return [k for k, _ in items[:2]]


def _apply_prescription(params: dict, weakest: list[str]) -> tuple[dict, str]:
    new_params = dict(params)
    notes = []
    for w in weakest[:1]:  # apply only top-1 prescription per iter to allow attribution
        rule = PRESCRIPTION.get(w)
        if not rule:
            continue
        param_name, delta, cap, kind = rule
        if param_name == "__user_action":
            notes.append(f"{w}↑필요(사용자 조치): curate 강화 또는 추가 수집")
            continue
        cur = new_params.get(param_name, 0)
        nxt = round(cur + delta, 3)
        if kind == "max":
            nxt = min(nxt, cap)
        if nxt == cur:
            notes.append(f"{w}: {param_name} 상한({cap}) 도달")
        else:
            new_params[param_name] = nxt
            notes.append(f"{w}: {param_name} {cur}→{nxt}")
    return new_params, "; ".join(notes)


def _run_mosaic_score(target: Path, params: dict, iter_idx: int, log_dir: Path) -> dict:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "mosaic.py"),
        "--target",
        str(target),
        "--grid",
        str(int(params["grid"])),
        "--blend",
        str(params["blend"]),
        "--neighbor-penalty",
        str(params["neighbor_penalty"]),
        "--usage-penalty",
        str(params["usage_penalty"]),
    ]
    print(f"\n[improve][iter {iter_idx}] mosaic.py {params}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    # Find latest layout for this target
    layouts = sorted(
        OUTPUT_DIR.glob(f"mosaic_{target.stem}_*_layout.json"), key=lambda p: p.stat().st_mtime
    )
    layout = layouts[-1]

    score_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "score.py"),
        "--layout",
        str(layout),
        "--target",
        str(target),
    ]
    print(f"[improve][iter {iter_idx}] score.py")
    subprocess.run(score_cmd, check=True, cwd=PROJECT_ROOT)

    score_path = layout.with_name(layout.stem.replace("_layout", "") + "_score.json")
    score_data = json.loads(score_path.read_text(encoding="utf-8"))
    return {"layout": str(layout), "score_path": str(score_path), "score": score_data}


def _make_chart(log_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    log = json.loads(log_path.read_text(encoding="utf-8"))
    iters = [it["iter"] for it in log["iterations"]]
    totals = [it["score"]["total"] for it in log["iterations"]]
    plt.figure(figsize=(8, 4))
    plt.plot(iters, totals, marker="o")
    plt.axhline(log["target_score"], color="r", linestyle="--", label=f"target {log['target_score']}")
    plt.xlabel("iteration")
    plt.ylabel("total score")
    plt.title(f"improve: {log['target']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    chart_path = log_path.with_suffix(".png").with_name(log_path.stem.replace("_log", "_chart") + ".png")
    plt.savefig(chart_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[improve] chart → {chart_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="자동 개선 루프")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-score", type=float, default=70)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--initial-grid", type=int, default=60)
    parser.add_argument("--initial-blend", type=float, default=0.3)
    args = parser.parse_args()

    ensure_dirs()
    if not args.target.exists():
        raise SystemExit(f"타겟 없음: {args.target}")

    params = {
        "grid": args.initial_grid,
        "blend": args.initial_blend,
        "neighbor_penalty": 1.0,
        "usage_penalty": 0.5,
    }

    log = {
        "target": str(args.target),
        "target_score": args.target_score,
        "started_at": datetime.now().isoformat(),
        "iterations": [],
        "best": None,
        "stopped_reason": None,
    }
    best_total = -1.0
    prev_total = -1.0
    stale = 0

    for it in range(args.max_iter):
        result = _run_mosaic_score(args.target, params, it, OUTPUT_DIR)
        score = result["score"]
        total = score["total"]
        weakest = _weakest(score["normalized"])
        new_params, prescription_note = _apply_prescription(params, weakest)

        log["iterations"].append(
            {
                "iter": it,
                "params": params.copy(),
                "score": score,
                "weakest": weakest,
                "prescription": prescription_note or "(none)",
                "layout": result["layout"],
            }
        )
        if total > best_total:
            best_total = total
            log["best"] = {"iter": it, "total": total, "layout": result["layout"]}

        print(f"\n[improve][iter {it}] total={total} grade={score['grade']} weakest={weakest}")
        print(f"[improve][iter {it}] prescription: {prescription_note}")

        if total >= args.target_score:
            log["stopped_reason"] = "target_reached"
            break

        if total <= prev_total:
            stale += 1
        else:
            stale = 0
        if stale >= 2:
            log["stopped_reason"] = "stale"
            break

        prev_total = total
        if new_params == params:
            log["stopped_reason"] = "params_capped"
            break
        params = new_params
    else:
        log["stopped_reason"] = "max_iter"

    log["finished_at"] = datetime.now().isoformat()
    log_path = OUTPUT_DIR / f"{args.target.stem}_improve_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[improve] 종료: {log['stopped_reason']} | best total={best_total}")
    print(f"[improve] log → {log_path}")
    _make_chart(log_path)

    if best_total < args.target_score:
        print(f"\n[improve] 목표 {args.target_score} 미달. mosaic-tuner 에이전트 호출 권장.")


if __name__ == "__main__":
    main()
