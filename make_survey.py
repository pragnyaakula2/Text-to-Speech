import random, shutil
from pathlib import Path
import pandas as pd

OUT = Path("outputs")
sets = {"seen": "as_a_result_oswald_was_not_hir", "unseen": "hello_world"}
dest = Path("survey"); dest.mkdir(exist_ok=True)
items = [(s, m, OUT / f"{base}_{m}.wav") for s, base in sets.items() for m in ("svr", "nn", "seq2seq")]
random.Random(1).shuffle(items)
rows = []
for i, (s, m, p) in enumerate(items, 1):
    shutil.copy(p, dest / f"clip{i}.wav")
    rows.append({"clip": f"clip{i}", "set": s, "model": m})
pd.DataFrame(rows).to_csv(dest / "key.csv", index=False)
print(pd.DataFrame(rows))
