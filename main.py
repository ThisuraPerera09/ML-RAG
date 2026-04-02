"""
main.py  —  Sri Lankan Event Planning Cost Pipeline
----------------------------------------------------
Generates the event dataset and runs the ML pipeline
(clean → feature engineer → train Random Forest → evaluate).

Usage:
  python main.py
"""

import time
import pandas as pd

from generate_data   import *           # creates data/raw_events.csv
from rag_ml_runner   import run_full_pipeline


if __name__ == "__main__":
    total_start = time.time()
    print("=" * 60)
    print("  Event Planning Cost — End-to-End Pipeline")
    print("=" * 60)

    # ── Step 1: Generate data ─────────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("#  Generating raw event dataset")
    print("#" * 60)
    # generate_data.py already ran on import above and saved data/raw_events.csv

    # ── Step 2: Run ML pipeline ───────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("#  Running ML Pipeline  (clean → features → train → evaluate)")
    print("#" * 60)

    t = time.time()
    results = run_full_pipeline("data/raw_events.csv")
    elapsed = time.time() - t

    # ── Step 3: Print results ─────────────────────────────────────────────────
    m = results["metrics"]
    s = results["dataset_stats"]

    print(f"\n  Pipeline completed in {elapsed:.1f}s")

    print(f"""
{"=" * 60}
  DATASET SUMMARY
{"=" * 60}
  Total events      : {s['total_events']:,}
  Cities            : {', '.join(s['cities'])}
  Avg estimated cost: LKR {s['avg_estimated_cost']:,}
  Min estimated cost: LKR {s['min_estimated_cost']:,}
  Max estimated cost: LKR {s['max_estimated_cost']:,}
  Avg guests        : {s['avg_guests']:,}
""")

    if s["event_type_avg_costs"]:
        print("  Avg cost by event type:")
        for etype, cost in sorted(s["event_type_avg_costs"].items(), key=lambda x: -x[1]):
            print(f"    {etype:<22}: LKR {cost:,}")

    if s["city_avg_costs"]:
        print("\n  Avg cost by city:")
        for city, cost in sorted(s["city_avg_costs"].items(), key=lambda x: -x[1]):
            print(f"    {city:<20}: LKR {cost:,}")

    if m:
        print(f"""
{"=" * 60}
  MODEL RESULTS  (Random Forest)
{"=" * 60}
  R²        : {m['r2']:.4f}   → explains {m['r2']*100:.1f}% of cost variance
  Accuracy  : {m['accuracy']:.1f}%
  MAE       : LKR {m['mae']:,.0f}
  RMSE      : LKR {m['rmse']:,.0f}
  Train rows: {m['train_size']:,}
  Test rows : {m['test_size']:,}
""")

        print("  Top features driving cost:")
        top_feats = sorted(results["feature_importance"].items(), key=lambda x: -x[1])[:8]
        for feat, imp in top_feats:
            bar = "█" * int(imp * 100)
            print(f"    {feat:<28}: {bar} {imp:.4f}")

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  ALL DONE  ({total:.1f}s total)")
    print(f"  To launch the RAG chat app:")
    print(f"    Backend : uvicorn api:app --reload --port 8000")
    print(f"    Frontend: cd frontend && npm run dev")
    print(f"{'=' * 60}")
