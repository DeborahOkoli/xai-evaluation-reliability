"""Analyse RQ5 experiment outputs -> summary JSON + pgfplots coordinates."""
import glob, json, itertools
import numpy as np, pandas as pd

R = "results"   # same folder as OUT in run_exp.py
METHODS = ["KernelSHAP", "LIME", "Occlusion", "Random"]
BASELINES = ["zero", "mean", "sample"]
METRICS = ["Deletion", "FaithCorr"]
NGRID = [10, 25, 50, 100, 150]
B = 300
rng = np.random.default_rng(2026)

data = {}
for f in sorted(glob.glob(f"{R}/scores_*.npz")):
    z = np.load(f)
    parts = f.split("/")[-1][7:-4].rsplit("_", 2)
    ds, mdl, s = parts[0], parts[1], int(parts[2])
    data[(ds, mdl, s)] = (z["S"], float(z["acc"]))

seeds = list(range(1, 11))
dsm = sorted({(k[0], k[1]) for k in data if all((k[0], k[1], s) in data for s in seeds)})
if not dsm:
    raise SystemExit("No complete results yet. Run the experiment first:  python -c \"import run_exp; run_exp.run_all()\"  "
                     "(each dataset/model needs all 10 seeds in the results folder).")
out = {"configs": [f"{a}/{b}" for a, b in dsm], "seeds": seeds, "acc": {}}
for a, b in dsm:
    out["acc"][f"{a}/{b}"] = round(float(np.mean([data[(a, b, s)][1] for s in seeds])), 3)

rows = []          # long table of full-pool mean scores
stab = []          # rank stability vs n
for (ds, mdl) in dsm:
    for mi_, met in enumerate(METRICS):
        for bi, bl in enumerate(BASELINES):
            # full-pool mean score per seed & method
            M = np.array([[data[(ds, mdl, s)][0][m, :, bi, mi_].mean() for m in range(4)] for s in seeds])
            for si, s in enumerate(seeds):
                for m in range(4):
                    rows.append(dict(ds=ds, mdl=mdl, metric=met, baseline=bl, seed=s, method=METHODS[m], score=M[si, m]))
            ref = M.mean(0)
            spread = ref[:3].max() - ref[:3].min()
            degenerate = bool(spread < 1e-4)
            ref_top = int(np.argmax(ref[:3]))           # best of the three real methods
            # how often does each seed alone (full pool) pick the same winner?
            for n in NGRID:
                agree, taus, ci_excl = [], [], []
                for si, s in enumerate(seeds):
                    S = data[(ds, mdl, s)][0][:3, :, bi, mi_]   # real methods only
                    npool = S.shape[1]
                    if n > npool: continue
                    for _ in range(B):
                        idx = rng.choice(npool, size=n, replace=False)
                        mean = S[:, idx].mean(1)
                        agree.append(int(np.argmax(mean) == ref_top))
                        r1 = np.argsort(np.argsort(-mean)); r0 = np.argsort(np.argsort(-ref[:3]))
                        taus.append(1 - 2 * np.sum(np.abs(r1 - r0) > 0) / 3)  # crude agreement
                stab.append(dict(ds=ds, mdl=mdl, metric=met, baseline=bl, n=n,
                                 top1=float(np.mean(agree)), degenerate=degenerate))
df = pd.DataFrame(rows); st = pd.DataFrame(stab)
df.to_csv("full_pool_scores.csv", index=False)
st.to_csv("stability.csv", index=False)

valid = st[~st.degenerate]
cells = valid.groupby(["ds", "mdl", "metric", "baseline"])
out["n_cells_total"] = int(st.groupby(["ds", "mdl", "metric", "baseline"]).ngroups)
out["n_cells_degenerate"] = int(st[st.degenerate].groupby(["ds", "mdl", "metric", "baseline"]).ngroups)
out["n_cells_valid"] = int(cells.ngroups)

# min n reaching >= 0.90 top-1 agreement
minn = []
for k, g in cells:
    g = g.sort_values("n")
    ok = g[g.top1 >= 0.90]
    minn.append(dict(cell="/".join(k), min_n=int(ok.n.min()) if len(ok) else None,
                     top1_at_25=float(g[g.n == 25].top1.iloc[0]), top1_at_150=float(g[g.n == 150].top1.iloc[0])))
mn = pd.DataFrame(minn)
out["min_n"] = {
    "10": int((mn.min_n == 10).sum()), "25": int((mn.min_n == 25).sum()), "50": int((mn.min_n == 50).sum()),
    "100": int((mn.min_n == 100).sum()), "150": int((mn.min_n == 150).sum()), "never": int(mn.min_n.isna().sum())}
out["mean_top1_by_n"] = {str(n): round(float(valid[valid.n == n].top1.mean()), 3) for n in NGRID}
out["median_top1_by_n"] = {str(n): round(float(valid[valid.n == n].top1.median()), 3) for n in NGRID}
out["frac_cells_top1_lt90_at_150"] = round(float((mn.top1_at_150 < 0.9).mean()), 3)
out["frac_cells_top1_lt90_at_25"] = round(float((mn.top1_at_25 < 0.9).mean()), 3)
mn.to_csv("min_n.csv", index=False)

# curves by metric (pgfplots)
curves = {}
for met in METRICS:
    v = valid[valid.metric == met]
    curves[met] = [(n, round(float(v[v.n == n].top1.mean()), 3)) for n in NGRID]
for dsn in sorted(valid.ds.unique()):
    v = valid[valid.ds == dsn]
    curves[dsn] = [(n, round(float(v[v.n == n].top1.mean()), 3)) for n in NGRID]
out["curves"] = curves

# winners: baseline and metric dependence (reference = mean over seeds)
ref = df[df.method != "Random"].groupby(["ds", "mdl", "metric", "baseline", "method"]).score.mean().reset_index()
win = ref.loc[ref.groupby(["ds", "mdl", "metric", "baseline"]).score.idxmax()]
win = win.merge(st[st.n == 10][["ds", "mdl", "metric", "baseline", "degenerate"]], on=["ds", "mdl", "metric", "baseline"])
win = win[~win.degenerate]
tab = win.pivot_table(index=["ds", "mdl", "metric"], columns="baseline", values="method", aggfunc="first")
out["winner_table"] = {"/".join(k): {b: tab.loc[k, b] if b in tab.columns and pd.notna(tab.loc[k, b]) else "tie" for b in BASELINES}
                       for k in tab.index}
changes = [len(set(v for v in d.values() if v != "tie")) > 1 for d in out["winner_table"].values()]
out["frac_metric_model_cells_winner_changes_with_baseline"] = [int(sum(changes)), len(changes)]
# metric disagreement: same ds/mdl/baseline, deletion vs faithcorr
w2 = win.pivot_table(index=["ds", "mdl", "baseline"], columns="metric", values="method", aggfunc="first").dropna()
out["metric_disagree"] = [int((w2.Deletion != w2.FaithCorr).sum()), int(len(w2))]

# per-seed winner (full pool) differs from reference winner?
per_seed = df[df.method != "Random"].loc[lambda d: d.groupby(["ds", "mdl", "metric", "baseline", "seed"]).score.idxmax()]
per_seed = per_seed.merge(win[["ds", "mdl", "metric", "baseline", "method"]].rename(columns={"method": "ref"}),
                          on=["ds", "mdl", "metric", "baseline"])
out["seed_winner_differs"] = [int((per_seed.method != per_seed.ref).sum()), int(len(per_seed))]
cells_seed = per_seed.groupby(["ds", "mdl", "metric", "baseline"]).apply(lambda g: (g.method != g.ref).any())
out["cells_where_some_seed_changes_winner"] = [int(cells_seed.sum()), int(len(cells_seed))]

# variance decomposition of scores (real methods, full pool means) per ds/mdl/metric
dec = []
for (ds, mdl, met), g in df[df.method != "Random"].groupby(["ds", "mdl", "metric"]):
    if set(g.baseline.unique()) != set(BASELINES): continue
    y = g.score.values; gm = y.mean(); sst = ((y - gm) ** 2).sum()
    if sst < 1e-12: continue
    def ss(cols):
        m = g.groupby(cols).score.transform("mean"); return ((m - gm) ** 2).sum()
    s_m, s_b, s_s = ss(["method"]), ss(["baseline"]), ss(["seed"])
    s_mb = ss(["method", "baseline"]) - s_m - s_b
    s_ms = ss(["method", "seed"]) - s_m - s_s
    resid = sst - s_m - s_b - s_s - s_mb - s_ms
    dec.append(dict(ds=ds, mdl=mdl, metric=met, method=s_m / sst, baseline=s_b / sst, seed=s_s / sst,
                    method_x_baseline=s_mb / sst, method_x_seed=s_ms / sst, other=resid / sst))
dd = pd.DataFrame(dec); dd.to_csv("variance_decomposition.csv", index=False)
out["variance_share_mean"] = {c: round(float(dd[c].mean()), 3) for c in
                              ["method", "baseline", "seed", "method_x_baseline", "method_x_seed", "other"]}

# within-cell instance sampling: share of cells where 95% bootstrap CI of (best - second best) excludes 0 at n=25/50/150
ci = {}
for n in [25, 50, 150]:
    hits = tot = 0
    for (ds, mdl, met, bl), g in valid[valid.n == n].groupby(["ds", "mdl", "metric", "baseline"]):
        bi = BASELINES.index(bl); mi_ = METRICS.index(met)
        S = data[(ds, mdl, seeds[0])][0][:3, :, bi, mi_]
        if n > S.shape[1]: continue
        idx = rng.choice(S.shape[1], size=n, replace=False); X = S[:, idx]
        order = np.argsort(-X.mean(1)); diff = X[order[0]] - X[order[1]]
        bs = [diff[rng.integers(0, n, n)].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5]); hits += int(lo > 0); tot += 1
    ci[str(n)] = [hits, tot]
out["ci_excludes_zero_top2"] = ci

json.dump(out, open("exp_results.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ("winner_table",)}, indent=1, default=str))
print(pd.Series(out["winner_table"]).to_string())

# seeds in which a setting was completely uninformative (all four methods, incl. Random, equal)
unin = []
for (ds, mdl, met, bl), g in df.groupby(["ds", "mdl", "metric", "baseline"]):
    for s, gs in g.groupby("seed"):
        if gs.score.max() - gs.score.min() < 1e-6: unin.append((ds, mdl, met, bl, s))
u = pd.DataFrame(unin, columns=["ds", "mdl", "metric", "baseline", "seed"])
out["uninformative_seed_cells"] = u.groupby(["ds", "mdl", "metric", "baseline"]).size().to_dict() if len(u) else {}
out["uninformative_seed_cells"] = {"/".join(k): int(v) for k, v in out["uninformative_seed_cells"].items()}
out["n_lt90_at_25"] = int((mn.top1_at_25 < 0.9).sum()); out["n_lt90_at_150"] = int((mn.top1_at_150 < 0.9).sum())
json.dump(out, open("exp_results.json", "w"), indent=1, default=str)
print("uninformative:", out["uninformative_seed_cells"], out["n_lt90_at_25"], out["n_lt90_at_150"])
