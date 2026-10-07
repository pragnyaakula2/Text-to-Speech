import json, numpy as np, config as C, models
from postprocess import spec_to_wav, save_wav

d = np.load(C.DATA_DIR / "dataset.npz")
norm = json.load(open(C.DATA_DIR / "meta.json"))["norm"]
i = 0
ids = d["X"][i:i+1]
print("text:", d["texts"][i])
for name in models.NAMES:
    m = models.load(name)
    S = models.predict(name, m, ids)[0]
    print(name, "std of prediction:", round(float(S.std()), 4), "| real:", round(float(d["Y"][i].std()), 4))
    save_wav(C.WAV_DIR / f"train{i}_{name}.wav", spec_to_wav(S, norm, 100))