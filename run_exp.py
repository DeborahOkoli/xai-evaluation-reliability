"""
RQ5 experiment: how much do random seeds, the number of evaluation
instances, and the masking baseline change faithfulness scores and the
ranking of feature-attribution methods?

Datasets : Breast Cancer Wisconsin (healthcare, tabular),
           NASA C-MAPSS FD001 (manufacturing / predictive maintenance),
           Digits 8x8 (image).
Models   : LightGBM, MLP.
Methods  : KernelSHAP, LIME, Occlusion, Random (reference).
Metrics  : Deletion faithfulness, Faithfulness correlation.
Factors  : seed (1..5; controls split, model, explainer, instance draw),
           baseline (zero, mean, random training row),
           number of evaluation instances n (analysed later by subsampling).
Output   : results/scores_<dataset>_<model>_<seed>.npz
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import shap
from lime.lime_tabular import LimeTabularExplainer

POOL = int(os.environ.get("POOL", 300))
METHODS = ["KernelSHAP", "LIME", "Occlusion", "Random"]
BASELINES = ["zero", "mean", "sample"]
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5]
N_SUBSETS = 30
# ---- set these two paths for your computer ----
CMAPSS_FILE = "train_FD001.txt"   # NASA C-MAPSS training file (FD001)
OUT = "results"                    # folder where scores are saved
os.makedirs(OUT, exist_ok=True)

def load(name, seed):
    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    elif name == "digits":
        X, y = load_digits(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    elif name == "cmapss":
        cols = ["unit", "cycle"] + [f"set{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
        df = pd.read_csv(CMAPSS_FILE, sep=r"\s+", header=None, names=cols)
        df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
        keep = [f"s{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
        X = df[keep].values.astype(float); y = (df["rul"] <= 30).astype(int).values
        g = df["unit"].values
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed).split(X, y, g))
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        # keep the training set manageable
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xtr), size=min(6000, len(Xtr)), replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    return Xtr, Xte, ytr, yte

def fit(model, Xtr, ytr, seed):
    if model == "lightgbm":
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=seed, verbose=-1, n_jobs=1)
    else:
        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                          max_iter=400, early_stopping=True, random_state=seed))
    m.fit(Xtr, ytr)
    return m

def main(dataset, model, seed):
    t0 = time.time()
    rng = np.random.default_rng(1000 + seed)
    Xtr, Xte, ytr, yte = load(dataset, seed)
    m = fit(model, Xtr, ytr, seed)
    acc = (m.predict(Xte) == yte).mean()
    pool = rng.choice(len(Xte), size=min(POOL, len(Xte)), replace=False)
    Xp = Xte[pool]
    d = Xtr.shape[1]
    proba = lambda z: m.predict_proba(z)   # plain function: works across shap versions
    pred = proba(Xp).argmax(1)

    # ---- attributions (towards the predicted class) ----
    A = {k: np.zeros((len(Xp), d)) for k in METHODS}
    bg = shap.sample(Xtr, 50, random_state=seed)
    kx = shap.KernelExplainer(proba, bg, seed=seed)
    lx = LimeTabularExplainer(Xtr, discretize_continuous=False, random_state=seed, mode="classification")
    mu = Xtr.mean(0)
    for i, x in enumerate(Xp):
        c = pred[i]
        sv = kx.shap_values(x[None, :], nsamples=256, silent=True)
        sv = np.asarray(sv)
        A["KernelSHAP"][i] = sv[0, :, c] if sv.ndim == 3 else np.asarray(sv[c])[0]
        e = lx.explain_instance(x, proba, labels=(c,), num_features=d, num_samples=500)
        for f, w in e.as_map()[c]:
            A["LIME"][i, f] = w
        Xo = np.repeat(x[None, :], d, 0); Xo[np.arange(d), np.arange(d)] = mu
        A["Occlusion"][i] = proba(x[None, :])[0, c] - proba(Xo)[:, c]
        A["Random"][i] = rng.normal(size=d)

    # ---- metrics ----
    # scores[method, instance, baseline, metric]  metric 0 = deletion, 1 = faithfulness correlation
    S = np.zeros((len(METHODS), len(Xp), len(BASELINES), 2))
    subsets = [rng.choice(d, size=max(1, int(np.ceil(0.2 * d))), replace=False) for _ in range(N_SUBSETS)]
    for i, x in enumerate(Xp):
        c = pred[i]; p0 = proba(x[None, :])[0, c]
        for b, bl in enumerate(BASELINES):
            if bl == "zero":   base = np.zeros(d)
            elif bl == "mean": base = mu
            else:              base = Xtr[rng.integers(len(Xtr))]
            # faithfulness correlation: same random subsets for every method
            Xs = np.repeat(x[None, :], N_SUBSETS, 0)
            for j, sidx in enumerate(subsets): Xs[j, sidx] = base[sidx]
            drops_fc = p0 - proba(Xs)[:, c]
            for mi, meth in enumerate(METHODS):
                a = A[meth][i]
                order = np.argsort(-a)
                Xd = np.repeat(x[None, :], len(FRACS), 0)
                for j, f in enumerate(FRACS):
                    k = int(np.ceil(f * d)); Xd[j, order[:k]] = base[order[:k]]
                S[mi, i, b, 0] = np.mean(p0 - proba(Xd)[:, c])
                sums = np.array([a[s].sum() for s in subsets])
                if sums.std() < 1e-12 or drops_fc.std() < 1e-12: S[mi, i, b, 1] = 0.0
                else: S[mi, i, b, 1] = np.corrcoef(sums, drops_fc)[0, 1]
    np.savez_compressed(f"{OUT}/scores_{dataset}_{model}_{seed}.npz", S=S, acc=acc,
                        methods=METHODS, baselines=BASELINES, d=d, n_pool=len(Xp))
    print(f"{dataset} {model} seed={seed} acc={acc:.3f} d={d} n={len(Xp)} {time.time()-t0:.0f}s", flush=True)

def run_all(seeds=range(1, 11)):
    """Run every dataset/model/seed. Skips results that already exist."""
    global POOL
    for ds in ["breast_cancer", "cmapss", "digits"]:
        for mdl in ["lightgbm", "mlp"]:
            POOL = 150 if (ds == "digits" and mdl == "lightgbm") else 300
            for s in seeds:
                if not os.path.exists(f"{OUT}/scores_{ds}_{mdl}_{s}.npz"):
                    main(ds, mdl, s)

if __name__ == "__main__" and len(sys.argv) == 4 and not sys.argv[1].startswith("-"):
    ds, mdl = sys.argv[1], sys.argv[2]
    for s in [int(v) for v in sys.argv[3].split(",")]:
        if not os.path.exists(f"{OUT}/scores_{ds}_{mdl}_{s}.npz"):
            main(ds, mdl, s)
