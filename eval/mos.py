"""Mean Opinion Score.

  python eval/mos.py --init     # creates eval/mos.csv with 10 empty rater rows
  python eval/mos.py            # prints mean MOS per model from the filled-in file

Each rater listens to the three wavs (shuffle the order / hide model names) and gives 1-5:
5 Excellent, 4 Good, 3 Fair, 2 Poor, 1 Bad.
"""
import argparse
from pathlib import Path

import pandas as pd

CSV = Path(__file__).parent / "mos.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    args = ap.parse_args()
    if args.init:
        pd.DataFrame({"rater": [f"r{i + 1}" for i in range(10)],
                      "svr": "", "nn": "", "seq2seq": ""}).to_csv(CSV, index=False)
        print("created", CSV)
        return
    df = pd.read_csv(CSV)
    scores = df[["svr", "nn", "seq2seq"]].apply(pd.to_numeric, errors="coerce")
    print(scores.mean().round(2).to_string())
    print("raters:", int(scores.notna().all(axis=1).sum()))
    scores.mean().round(2).to_frame("MOS").to_csv(CSV.parent / "mos_summary.csv")


if __name__ == "__main__":
    main()
