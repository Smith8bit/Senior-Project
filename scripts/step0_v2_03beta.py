"""v2.03beta = step 0 of the v2.04 plan (claude/v2-04-plan-synthesis.md): post-hoc changes to v2.03, NO re-training.

Runs on a CPU. The three seeds' SAVED v2.03 weights are only used for forward passes; XGBoost is trained once per seed
on the same fit rows as v2.03 (a few minutes). Settings decided by Siwat on 2026-10-26:
  * call-mix by rank: 1/3 down, 1/3 up, the rest flat on every date;
  * class-share correction (Saerens EM): on each date's stocks only (causal);
  * deep + XGBoost blend: deep weight w on the grid 0, 0.1, ..., 1, fitted per horizon on the inner hold-out;
  * news placebo: 5 shuffles, on validation and on 2026.

What is compared (each fitted on the INNER HOLD-OUT only; nothing is fitted on validation or 2026 labels):
  raw    v2.03 as trained: most likely class.
  ts     temperature scaling (Guo et al. 2017): one temperature per seed and horizon, fitted by log-loss. Never changes
         a call, so macro-F1 is exactly the raw score; only the probabilities (log-loss) change.
  bcts   bias-corrected temperature scaling (Alexandari et al. 2020): temperature + biases for "flat" and "up".
  rank   per date, the calls that maximise the summed log-probability with exactly round(n/3) "down", round(n/3) "up"
         and the rest "flat" (an assignment problem; scipy linear_sum_assignment). Needs no fitting and gives the same
         calls for raw, ts or bcts probabilities.
  em     Saerens, Latinne & Decaestecker (2002): bcts probabilities, then each date's class shares are estimated by EM
         from that date's stocks (no labels) and the probabilities are re-weighted to them; most likely class. The
         starting shares are the inner hold-out's class shares (the shares bcts is calibrated to).
  blend  w * ts(v2.03) + (1 - w) * ts(XGBoost), deep seed s with XGBoost seed s; w per horizon, chosen on the inner
         hold-out by the mean macro-F1 of the three seed pairs (ties: larger w).
  blend_<rank|em>  the blend followed by whichever of rank / em scores higher on the inner hold-out (cross-fitted).
Inner hold-out scores of the fitted methods are CROSS-FITTED: the hold-out dates are split into two halves; each half
is scored with parameters fitted on the other half. Validation and 2026 use parameters fitted on the whole hold-out.

Adopt rule (pre-registered in claude/log-entry-2026-10-26-v2-03beta-design.md, before any result):
  a call-changing method is adopted if its cross-fitted inner-hold-out 3-horizon mean macro-F1 is above raw AND its
  validation macro-F1 (seed mean) is not below raw at any horizon. ts is adopted (for probabilities) if it lowers the
  validation log-loss at every horizon. 2026 is a seen window: reported, never used to adopt.

News placebo: the per-stock-day sentiment vectors [p_pos, p_neg, p_neu] of the stock-days WITH news inside the period
(validation block, or 2026) are shuffled across stocks and days; which stock-days have news is kept; no-news days keep
the fill. Five shuffles (rng seeds 20261026 + k). Also scored: news on, and news blanked (every day of the period gets
the no-news fill). Reading: placebo ~ news on -> what helps is the news pattern (which days have news), not what it
says; placebo ~ blanked -> the content matters.

Stages (each skips work already saved in the output folder):
  python scripts/step0_v2_03beta.py prepare   # 2026 world from the saved v2 forward-test inputs + identity check
  python scripts/step0_v2_03beta.py predict   # v2.03 forward passes (torch, CPU); resumable part by part
  python scripts/step0_v2_03beta.py trees     # XGBoost on the fit rows (xgboost)
  python scripts/step0_v2_03beta.py analyze   # everything else (numpy, scipy); writes step0.json and prints tables
  python scripts/step0_v2_03beta.py all       # the four in order
Defaults: --run-json results/v2xx/A100/v2_03.json, --fwd-inputs results/forward_2026/inputs,
          --news-fwd data/derived/sentiment_scores_fwd2026.csv, --out-dir results/v2_03beta.
VALIDATION ONLY for decisions; the 2025 test block stays sealed (no 2025 row is predicted or scored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import baselines_tabular as BT  # noqa: E402
import features as F  # noqa: E402
import features_proposal as FP  # noqa: E402
import forward_test_v2 as FT  # noqa: E402
import forward_test_v2xx as FV  # noqa: E402
import labels as L  # noqa: E402
import posthoc as PH  # noqa: E402
import sequences as SQ  # noqa: E402

NAME = "v2.03beta"
HS = L.HORIZONS
CLS = L.CLASSES
# --- Siwat's settings, 2026-10-26 -----------------------------------------------------------------------------------
RANK_SHARES = (1 / 3, 1 / 3)          # down, up per date (flat = the rest)
EM_WINDOW = "date"                    # each date's stocks
BLEND_GRID = tuple(round(0.1 * i, 1) for i in range(11))
PLACEBO_SHUFFLES = 5
PLACEBO_PERIODS = ("val", "fwd")
# --- fixed by this design (no tuning) ---------------------------------------------------------------------------------
PLACEBO_RNG0 = 20261026
EM_MAX_ITER, EM_TOL = 1000, 1e-8
EPS = 1e-9
PASS_BAR = FT.PASS_BAR
say = FT.say


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def paths(args):
    out = Path(args.out_dir)
    return {"out": out, "fwd_root": out / "fwd_root", "parts": out / "parts", "trees": out / "trees.csv.gz",
            "manifest": out / "manifest.json", "summary": out / "step0.json"}


# ======================================================================================================== prepare
def prepare(args):
    P = paths(args)
    P["out"].mkdir(parents=True, exist_ok=True)
    if P["manifest"].exists() and (P["fwd_root"] / "forward_config.json").exists() and not args.redo:
        say("prepare: done (pass --redo to rebuild)"); return
    inp = P["out"] / "inputs"
    src = Path(args.fwd_inputs)
    need = ["forward_prices.csv", "forward_events.csv", "inputs_meta.json"]
    miss = [f for f in need if not (src / f).exists()]
    if miss:
        raise SystemExit(f"stop: missing {miss} in {src} (copy them from Drive Senior/results/forward_2026/inputs/)")
    inp.mkdir(parents=True, exist_ok=True)
    for f in need:
        shutil.copy(src / f, inp / f)
    meta = json.loads((inp / "inputs_meta.json").read_text(encoding="utf-8"))
    for f, k in (("forward_prices.csv", "forward_prices_sha256"), ("forward_events.csv", "forward_events_sha256")):
        if sha(inp / f) != meta[k]:
            raise SystemExit(f"stop: {f} differs from the copy the 2026 forward tests used ({sha(inp / f)[:12]} vs {meta[k][:12]})")
    # the same call the v2.03 forward notebooks made (no extra breaks), so the 2026 rows are the same rows
    FT.main(["prepare", "--fwd-root", str(P["fwd_root"]), "--out-dir", str(P["out"]), "--price-source", "yahoo"])


# ======================================================================================================== shared world
def load_run(args) -> dict:
    run = FV.load_run(Path(args.run_json))           # stops if the model code differs from the code v2.03 used
    if run.get("smoke"):
        raise SystemExit("stop: this is a --smoke run of train_v2xx.py")
    return run


def world(args, run, with_news_fwd=True) -> dict:
    """Rows and inputs exactly as v2.03 saw them, in the 2026 world (forward_test_v2xx.build), plus the inner hold-out
    (`es`) and fit rows rebuilt exactly as scripts/train_v2xx.build_data cuts them, checked against the run."""
    P = paths(args)
    cfg = run["config"]
    B = FV.build(cfg, P["fwd_root"], args.news_fwd if with_news_fwd else None)
    tr = B["tr_rows"]
    tdates = np.sort(tr.date.unique()); cut = int(len(tdates) * (1 - cfg["protocol"]["inner_es_share"]))
    fit_end, es_start = tdates[cut - 1 - cfg["protocol"]["purge"]], tdates[cut]
    fit, es = tr[tr.date <= fit_end], tr[tr.date >= es_start]
    ie = run["inner_es"]
    assert (str(pd.Timestamp(fit_end).date()), str(pd.Timestamp(es_start).date()), len(fit), len(es)) == \
        (ie["fit_end"], ie["es_start"], ie["rows_fit"], ie["rows_es"]), f"inner hold-out differs from the run: {ie}"
    assert len(B["va"]) == run["data"]["rows_by_split"]["val"], "validation rows differ from the run"
    B.update(fit=fit, es=es, fit_end=fit_end, es_start=es_start)
    return B


# ======================================================================================================== predict
def _same_state(a, b) -> bool:
    import torch
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class SentimentVariants:
    """z (standardised inputs) with the 3 sentiment channels replaced: news on, blanked, or shuffled in a period.
    Uses the same training-row mean / std as sequences.standardise_full (sentiment channels have no missing values)."""

    def __init__(self, full, channels, train_index, fill, has_news):
        self.full, self.fill = full, fill
        self.z = SQ.standardise_full(full, channels, train_index)
        self.idx = [channels.index(c) for c in FP.SENT_CH]
        self.raw = full[FP.SENT_CH].to_numpy(float)
        self.mu = full.loc[train_index, FP.SENT_CH].mean().to_numpy(float)
        self.sd = full.loc[train_index, FP.SENT_CH].std().replace(0, 1.0).to_numpy(float)
        self.has = np.asarray(has_news, float) == 1.0
        chk = self._z_with(self.raw)
        assert np.array_equal(chk, self.z[:, self.idx]), "sentiment standardisation does not match standardise_full"

    def _z_with(self, vals):
        return np.nan_to_num(((vals - self.mu) / self.sd).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def get(self, kind: str, period_mask=None, k: int = 0) -> np.ndarray:
        if kind == "news":
            return self.z
        vals = self.raw.copy()
        if kind == "blank":
            vals[period_mask] = self.fill
        elif kind == "placebo":
            m = np.flatnonzero(period_mask & self.has)
            perm = np.random.default_rng(PLACEBO_RNG0 + k).permutation(len(m))
            vals[m] = self.raw[m[perm]]
        else:
            raise ValueError(kind)
        z = self.z.copy()
        z[:, self.idx] = self._z_with(vals)
        return z


def part_specs(B) -> list[tuple[str, str, pd.DataFrame, str | None, int]]:
    specs = [("es", "news", B["es"], None, 0), ("val", "news", B["va"], None, 0), ("fwd", "news", B["fwd"], None, 0),
             ("val_blank", "blank", B["va"], "val", 0), ("fwd_blank", "blank", B["fwd"], "fwd", 0)]
    for per in PLACEBO_PERIODS:
        for k in range(1, PLACEBO_SHUFFLES + 1):
            specs.append((f"{per}_placebo{k}", "placebo", B["va"] if per == "val" else B["fwd"], per, k))
    return specs


def predict(args):
    import torch
    import train_v2xx as TV
    P = paths(args)
    P["parts"].mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    run = load_run(args)
    cfg, seeds = run["config"], [r["seed"] for r in run["runs"]]
    t0 = time.time()
    B = world(args, run)
    full, channels = B["full"], B["channels"]
    assert channels == run["channels"], "channels differ from the run"
    assert B["fit_symbols"] == run["stock_vocab"], "stock vocabulary rebuilt from the fit rows differs from the run"
    fill = FV.NO_NEWS[cfg["inputs"]["sentiment_no_news"]]
    V = SentimentVariants(full, channels, B["tr_rows"].index, fill, full["has_news"].to_numpy())
    va0, va1 = (pd.Timestamp(d) for d in FT.OFFICIAL_BLOCKS["val"])
    masks = {"val": ((full.date >= va0) & (full.date <= va1)).to_numpy(), "fwd": (full.date >= pd.Timestamp(FT.FWD_START)).to_numpy()}
    news_share = {per: round(float(V.has[m].mean()), 4) for per, m in masks.items()}
    say(f"{NAME} predict: seeds {seeds}; rows es {len(B['es'])} / val {len(B['va'])} / 2026 {len(B['fwd'])}; "
        f"stock-days with news: {news_share}; built in {time.time() - t0:.0f}s; device {TV.DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else f", {torch.get_num_threads()} CPU threads"))

    sectors = pd.read_csv(P["fwd_root"] / "data/sectors.csv")
    models = {}
    for seed in seeds:                                   # load once; the three horizon states are one shared checkpoint
        wf = Path(args.weights) / f"seed{seed}.pt"
        if not wf.exists():
            raise SystemExit(f"stop: missing {wf} (copy it from Drive Senior/results/v2xx/A100/v2_03_weights/)")
        ck = torch.load(wf, map_location="cpu", weights_only=False)        # the project's own checkpoint file
        assert ck["stock_vocab"] == run["stock_vocab"], "stock vocabulary differs between run and checkpoint"
        r_sd = np.asarray(ck["r_sd"])
        assert np.allclose(r_sd, B["tr_rows"][[f"fwd_{h}" for h in HS]].std().to_numpy(), rtol=0, atol=1e-12), "return scale differs"
        frozen = {s: i + 1 for i, s in enumerate(sorted(pd.read_csv(ROOT / "data/sectors.csv").sector.unique()))}
        assert ck["sector_ids"] == frozen, "sector ids differ from the frozen sector table"
        uniq, which = [], []
        for st in ck["states"]:
            j = next((i for i, u in enumerate(uniq) if _same_state(u, st)), None)
            if j is None:
                uniq.append(st); j = len(uniq) - 1
            which.append(j)
        nets = []
        for st in uniq:
            m = TV.build_model(cfg, len(channels), len(ck["stock_vocab"]), len(ck["sector_ids"]))
            m.load_state_dict(st); m.to(TV.DEVICE).eval(); nets.append(m)
        models[seed] = {"nets": nets, "which": which, "sha256": sha(wf)[:16], "epochs_used": ck["epochs_used"],
                        "sym2s": {s: i + 1 for i, s in enumerate(ck["stock_vocab"])},
                        "sym2g": dict(zip(sectors.symbol, sectors.sector.map(ck["sector_ids"])))}
        say(f"  seed {seed}: {len(uniq)} distinct state(s) for the 3 horizons; epochs used {ck['epochs_used']}")

    lookback, bs = cfg["model"]["lookback"], cfg["training"]["batch_size"]
    specs = part_specs(B)
    if args.core_only:                                   # es / val / 2026 only: enough for every method, no placebo
        specs = [s for s in specs if s[1] == "news"]
    todo = [s for s in specs if not (P["parts"] / f"{s[0]}.csv.gz").exists() or args.redo]
    n_all = sum(min(len(d), args.smoke_rows or len(d)) for _, _, d, _, _ in todo) * len(seeds)
    done, t1 = 0, time.time()
    for name, kind, d, per, k in todo:
        d = d.iloc[: args.smoke_rows] if args.smoke_rows else d
        zz = V.get(kind, masks.get(per), k)
        X = SQ.windows(full, zz, d.index, lookback)
        out = d[["symbol", "date"]].copy()
        out["date"] = out.date.dt.strftime("%Y-%m-%d")
        for h in HS:
            out[f"y_{h}"] = d[f"y_{h}"].to_numpy()
            if name in ("es", "val", "fwd"):
                out[f"r_{h}"], out[f"sigma_{h}"] = d[f"r_{h}"].to_numpy(), d[f"sigma_{h}"].to_numpy()
        for seed in seeds:
            M_ = models[seed]
            S = d.symbol.map(M_["sym2s"]).fillna(0).to_numpy(np.int64)
            G = d.symbol.map(M_["sym2g"]).fillna(0).to_numpy(np.int64)
            outs = [TV.predict(net, X, S, G, bs)[0] for net in M_["nets"]]
            for j, h in enumerate(HS):
                Pj = outs[M_["which"][j]][:, j]
                for c, cls in enumerate(CLS):
                    out[f"v2xx_s{seed}_p{h}_{cls}"] = Pj[:, c].round(6)
            done += len(d)
            el = time.time() - t1
            say(f"  {name:13s} seed {seed}: {len(d)} rows | {done}/{n_all} done, {el / 60:.1f} min so far, "
                f"about {el / max(done, 1) * (n_all - done) / 60:.0f} min left")
        tmp = P["parts"] / f"{name}.tmp.csv.gz"
        out.to_csv(tmp, index=False); tmp.replace(P["parts"] / f"{name}.csv.gz")
        del X, zz
    meta = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": cfg["name"], "seeds": seeds,
            "torch": torch.__version__, "threads": torch.get_num_threads(), "news_share": news_share,
            "device": str(TV.DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": __import__("platform").platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "weights": {str(s): {"sha256": m["sha256"], "epochs_used": m["epochs_used"]} for s, m in models.items()},
            "news_fwd": B["news_stats"], "smoke_rows": args.smoke_rows,
            "placebo": {"shuffles": PLACEBO_SHUFFLES, "rng_seeds": [PLACEBO_RNG0 + k for k in range(1, PLACEBO_SHUFFLES + 1)],
                        "periods": list(PLACEBO_PERIODS)}}
    (P["out"] / "predict_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    say(f"predict: done in {(time.time() - t0) / 60:.1f} min")


# ======================================================================================================== trees
def trees(args):
    from xgboost import XGBClassifier
    import xgboost
    P = paths(args)
    if P["trees"].exists() and not args.redo:
        say("trees: done"); return
    run = load_run(args)
    seeds = [r["seed"] for r in run["runs"]]
    t0 = time.time()
    with FT.forward_world(P["fwd_root"]):
        data, info = BT.build_dataset(root=P["fwd_root"], extra="e03")
    ie = run["inner_es"]
    tr = data[data.split == "train"]
    fit = tr[tr.date <= pd.Timestamp(ie["fit_end"])]
    parts = {"es": tr[tr.date >= pd.Timestamp(ie["es_start"])], "val": data[data.split == "val"],
             "fwd": data[data.date >= pd.Timestamp(FT.FWD_START)]}
    assert (len(fit), len(parts["es"]), len(parts["val"])) == (ie["rows_fit"], ie["rows_es"], run["data"]["rows_by_split"]["val"]), \
        "XGBoost rows differ from v2.03's fit / inner hold-out / validation rows"
    feats = F.FEATURES + BT.EXTRA["e03"]
    params = dict(BT.XGB) | ({"n_estimators": 20} if args.smoke_rows else {})
    frames = []
    for k, d in parts.items():
        f_ = d[["symbol", "date"]].copy(); f_["date"] = f_.date.dt.strftime("%Y-%m-%d"); f_.insert(0, "part", k)
        frames.append(f_.reset_index(drop=True))
    for seed in seeds:
        for h in HS:
            clf = XGBClassifier(objective="multi:softprob", num_class=3, random_state=seed, **params)
            clf.fit(fit[feats], fit[f"y_{h}"].to_numpy(int))
            for f_, d in zip(frames, parts.values()):
                pr = clf.predict_proba(d[feats])
                for c, cls in enumerate(CLS):
                    f_[f"xgb_s{seed}_p{h}_{cls}"] = pr[:, c].round(6)
            say(f"  XGBoost seed {seed} h{h}: trained on {len(fit)} fit rows ({time.time() - t0:.0f}s)")
    pd.concat(frames, ignore_index=True).to_csv(P["trees"], index=False)
    (P["out"] / "trees_meta.json").write_text(json.dumps({"xgboost": xgboost.__version__, "params": params, "features": len(feats),
                                                          "trained_on": f"fit rows only ({len(fit)}; same as v2.03)",
                                                          "seeds": seeds}, indent=2), encoding="utf-8")


# ======================================================================================================== post-hoc methods
def logp(p):
    return np.log(np.clip(p, EPS, 1.0))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.mean(logp(p)[np.arange(len(y)), y]))


A_MIN = 1e-3                          # temperature must stay positive (a = 1/T >= A_MIN keeps every call's order)


def fit_affine(p, y, bias: bool, iters: int = 100) -> np.ndarray:
    """Newton fit of q = softmax(a * log p + b), b = (0, b_flat, b_up) (bias=True) or b = 0 (temperature only, a = 1/T).
    The log-loss is convex in (a, b), so Newton with step halving finds the minimum. a is kept >= A_MIN (a positive
    temperature): when the free minimum has a < A_MIN (probabilities no better than, or worse than, a uniform guess),
    the constrained minimum lies on a = A_MIN and only the biases are fitted. Returns theta = (a, b_flat, b_up)."""
    z = logp(p)
    th = _newton(z, y, np.array([1.0, 0.0, 0.0]), [0, 1, 2] if bias else [0], iters)
    if th[0] < A_MIN:
        th = np.array([A_MIN, 0.0, 0.0])
        if bias:
            th = _newton(z, y, th, [1, 2], iters)
    return th


def _newton(z, y, th, free, iters):
    n = len(y)
    Y = np.eye(3)[y]

    def phi():                                                 # [n, 3 classes, 3 params]
        f = np.zeros((n, 3, 3)); f[:, :, 0] = z; f[:, 1, 1] = 1.0; f[:, 2, 2] = 1.0
        return f[:, :, free]
    Ph = phi()

    def loss(t):
        q = softmax(t[0] * z + np.array([0.0, t[1], t[2]]))
        return -np.mean(np.log(np.clip(q[np.arange(n), y], 1e-300, None))), q
    cur, q = loss(th)
    for _ in range(iters):
        g = np.einsum("ic,ick->k", q - Y, Ph) / n
        m = np.einsum("ic,ick->ik", q, Ph)
        H = (np.einsum("ic,ick,icl->kl", q, Ph, Ph) - np.einsum("ik,il->kl", m, m)) / n
        step = np.linalg.solve(H + 1e-10 * np.eye(len(free)), g)
        s = 1.0
        while s > 1e-6:
            t2 = th.copy(); t2[free] -= s * step
            new, q2 = loss(t2)
            if new <= cur:
                break
            s /= 2
        if s <= 1e-6 or cur - new < 1e-12:
            th, cur, q = (t2, new, q2) if new <= cur else (th, cur, q)
            break
        th, cur, q = t2, new, q2
    return th


def apply_affine(p, th):
    return softmax(th[0] * logp(p) + np.array([0.0, th[1], th[2]]))


def rank_calls(p, dates):
    """Per date: calls maximising sum log p with round(n/3) down, round(n/3) up, rest flat (assignment problem)."""
    from scipy.optimize import linear_sum_assignment
    lp = logp(p)
    out = np.empty(len(p), int)
    for d in np.unique(dates):
        i = np.flatnonzero(dates == d)
        n = len(i)
        kd, ku = int(round(n * RANK_SHARES[0])), int(round(n * RANK_SHARES[1]))
        cols = np.array([0] * kd + [1] * (n - kd - ku) + [2] * ku)
        r, c = linear_sum_assignment(-lp[i][:, cols])
        out[i[r]] = cols[c]
    return out


def em_adjust(p, prior_src, dates):
    """Saerens et al. (2002) per date: estimate the date's class shares from its stocks' probabilities (no labels)
    and re-weight. Returns the adjusted probabilities."""
    out = np.empty_like(p)
    w0 = np.asarray(prior_src, float)
    for d in np.unique(dates):
        i = np.flatnonzero(dates == d)
        q, pi = p[i], w0.copy()
        for _ in range(EM_MAX_ITER):
            a = q * (pi / w0); a /= a.sum(1, keepdims=True)
            new = a.mean(0)
            if np.abs(new - pi).max() < EM_TOL:
                pi = new; break
            pi = new
        a = q * (pi / w0); out[i] = a / a.sum(1, keepdims=True)
    return out


def f1(y, c):
    return PH.macro_f1(np.asarray(y, int), np.asarray(c, int))


# ======================================================================================================== analyze
def load_parts(args, seeds):
    P = paths(args)
    parts = {}
    for f in sorted(P["parts"].glob("*.csv.gz")):
        if ".tmp." in f.name:
            continue
        parts[f.name[:-7]] = pd.read_csv(f)
    need = {s[0] for s in [("es",), ("val",), ("fwd",)]}
    miss = need - set(parts)
    if miss:
        raise SystemExit(f"stop: run `predict` first (missing parts {sorted(miss)})")
    t = pd.read_csv(P["trees"]) if P["trees"].exists() else None
    if t is None:
        raise SystemExit("stop: run `trees` first")
    for k in ("es", "val", "fwd"):
        m = parts[k][["symbol", "date"]].merge(t[t.part == k].drop(columns="part"), on=["symbol", "date"], how="left", validate="one_to_one")
        assert m.filter(like="xgb_s").notna().all().all(), f"XGBoost predictions missing for some {k} rows"
        for c in m.columns:
            if c.startswith("xgb_s"):
                parts[k][c] = m[c].to_numpy()
    return parts


def probs(df, prefix, seed, h):
    return df[[f"{prefix}_s{seed}_p{h}_{c}" for c in CLS]].to_numpy(float)


def method_calls(fit_df, app_df, seeds, h, w_blend, mix):
    """Calls (and probabilities where defined) on app_df, every parameter fitted on fit_df. Returns
    {method: ([S, n] calls, [S, n, 3] probs or None)}."""
    yf = fit_df[f"y_{h}"].to_numpy(int)
    dates = app_df["date"].to_numpy()
    prior = np.bincount(yf, minlength=3) / len(yf)
    res = {k: ([], []) for k in ("raw", "ts", "bcts", "rank", "em", "blend", "blend_mix")}
    for s in seeds:
        pf, pa = probs(fit_df, "v2xx", s, h), probs(app_df, "v2xx", s, h)
        xf, xa = probs(fit_df, "xgb", s, h), probs(app_df, "xgb", s, h)
        th_ts, th_bc = fit_affine(pf, yf, False), fit_affine(pf, yf, True)
        q_ts, q_bc = apply_affine(pa, th_ts), apply_affine(pa, th_bc)
        q_em = em_adjust(q_bc, prior, dates)
        tx = fit_affine(xf, yf, False)
        qb = w_blend * q_ts + (1 - w_blend) * apply_affine(xa, tx)
        for k, (c, q) in {"raw": (pa.argmax(1), pa), "ts": (q_ts.argmax(1), q_ts), "bcts": (q_bc.argmax(1), q_bc),
                          "rank": (rank_calls(pa, dates), None), "em": (q_em.argmax(1), q_em),
                          "blend": (qb.argmax(1), qb)}.items():
            res[k][0].append(c); res[k][1].append(q)
        if mix == "rank":
            res["blend_mix"][0].append(rank_calls(qb, dates)); res["blend_mix"][1].append(None)
        else:
            qbf = w_blend * apply_affine(pf, th_ts) + (1 - w_blend) * apply_affine(xf, tx)
            qe = em_adjust(apply_affine(qb, fit_affine(qbf, yf, True)), prior, dates)
            res["blend_mix"][0].append(qe.argmax(1)); res["blend_mix"][1].append(qe)
    return {k: (np.stack(c), None if q[0] is None else np.stack(q)) for k, (c, q) in res.items()}


def choose_w(fit_df, seeds, h):
    """Deep weight w on BLEND_GRID maximising the seed-pair mean macro-F1 on fit_df (ties: larger w)."""
    y = fit_df[f"y_{h}"].to_numpy(int)
    best, bw = -1.0, None
    for w in BLEND_GRID:
        sc = []
        for s in seeds:
            pf, xf = probs(fit_df, "v2xx", s, h), probs(fit_df, "xgb", s, h)
            q = w * apply_affine(pf, fit_affine(pf, y, False)) + (1 - w) * apply_affine(xf, fit_affine(xf, y, False))
            sc.append(f1(y, q.argmax(1)))
        m = float(np.mean(sc))
        if m >= best - 1e-12:
            best, bw = m, w
    return bw, round(best, 4)


def halves(df):
    ud = np.sort(df.date.unique())
    first = set(ud[: len(ud) // 2])
    a = df.date.isin(first).to_numpy()
    return df[a].reset_index(drop=True), df[~a].reset_index(drop=True)


def summarise(y, dates, calls: dict, probs_: dict, bar=None, base="raw"):
    out = {}
    for k, C in calls.items():
        sc = [f1(y, c) for c in C]
        out[k] = {"macro_f1": round(statistics.mean(sc), 4), "std": round(statistics.stdev(sc), 4) if len(sc) > 1 else 0.0,
                  "per_seed": [round(x, 4) for x in sc],
                  "call_shares": [round(float((C == i).mean()), 4) for i in range(3)]}
        if probs_.get(k) is not None:
            out[k]["log_loss"] = round(statistics.mean(nll(q, y) for q in probs_[k]), 5)
    if dates is not None:
        boot = PH.block_bootstrap(dates, y, calls, pairs=[(k, base) for k in calls if k != base], bar=bar)
        for k in calls:
            out[k]["ci95"] = boot["values"][k]["ci95"]
            if bar is not None:
                out[k]["share_resamples_at_or_above_bar"] = boot["values"][k].get("share_of_resamples_at_or_above_bar")
            if k != base:
                dd = boot["differences"][f"{k} - {base}"]
                out[k]["minus_raw"] = {"point": dd["point"], "ci95": dd["ci95"], "share_above_0": dd["share_above_0"]}
    return out


def analyze(args):
    P = paths(args)
    run = load_run(args)
    seeds = [r["seed"] for r in run["runs"]]
    parts = load_parts(args, seeds)
    es, val, fwd = parts["es"], parts["val"], parts["fwd"]
    kk = {h: run["data"][f"h{h}"]["k"] for h in HS}
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "name": NAME,
           "what": "step 0 of the v2.04 plan: post-hoc changes to v2.03's saved weights; no re-training",
           "run": {"model": run["config"]["name"], "git_commit": run.get("git_commit"), "seeds": seeds},
           "settings": {"rank_shares_down_up": RANK_SHARES, "em_window": EM_WINDOW, "blend_grid": BLEND_GRID,
                        "placebo_shuffles": PLACEBO_SHUFFLES, "placebo_periods": PLACEBO_PERIODS,
                        "placebo_rng_seeds": [PLACEBO_RNG0 + k for k in range(1, PLACEBO_SHUFFLES + 1)]},
           "rows": {k: int(len(v)) for k, v in parts.items()}, "pass_bar": {f"h{h}": PASS_BAR[h] for h in HS},
           "smoke": bool(args.smoke_rows)}

    # 0) reproduction: validation (news on) vs the run's stored predictions, and inner hold-out F1 vs the run's record
    stored = pd.read_csv(Path(args.run_json).with_name(Path(args.run_json).stem + "_preds.csv.gz"))
    stored["date"] = stored.date.astype(str).str[:10]
    m = val.merge(stored, on=["symbol", "date"], suffixes=("", "_ref"), validate="one_to_one")
    rep, ok = {}, len(m) == len(val) == len(stored) or (args.smoke_rows and len(m) == len(val))
    for j, h in enumerate(HS):
        dp = max(float(np.abs(probs(m, "v2xx", s, h) - m[[f"v2xx_s{s}_p{h}_{c}_ref" for c in CLS]].to_numpy(float)).max()) for s in seeds)
        ye = es[f"y_{h}"].to_numpy(int)
        es_here = [round(f1(ye, probs(es, "v2xx", s, h).argmax(1)), 4) for s in seeds]
        es_run = [next(e for e in r["history"] if e["epoch"] == r["best_epoch"])["es_per_horizon"][j] for r in run["runs"]]
        rep[f"h{h}"] = {"val_max_abs_prob_diff": round(dp, 7), "es_f1_here": es_here, "es_f1_run": es_run}
        ok &= dp <= FV.REPRO_PROB_TOL and (args.smoke_rows or max(abs(a - b) for a, b in zip(es_here, es_run)) <= FV.REPRO_F1_TOL)
    res["reproduction"] = rep | {"ok": bool(ok)}
    say("REPRODUCTION (CPU forward passes vs the A100 run): " + "; ".join(
        f"h{h} val |dp| {rep[f'h{h}']['val_max_abs_prob_diff']:.1e}, hold-out F1 {rep[f'h{h}']['es_f1_here']} vs {rep[f'h{h}']['es_f1_run']}" for h in HS))
    if not ok:
        P["summary"].write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        raise SystemExit("stop: the saved weights do not reproduce v2.03 here; nothing else is scored")

    # 1) choose rank vs em on the inner hold-out, cross-fitted; choose the blend weight per horizon
    A, Bh = halves(es)
    cf = {}
    for h in HS:
        y_a, y_b = A[f"y_{h}"].to_numpy(int), Bh[f"y_{h}"].to_numpy(int)
        wa, _ = choose_w(Bh, seeds, h); wb, _ = choose_w(A, seeds, h)
        on_a = method_calls(Bh, A, seeds, h, wa, "rank"); on_b = method_calls(A, Bh, seeds, h, wb, "rank")
        on_a_em = method_calls(Bh, A, seeds, h, wa, "em"); on_b_em = method_calls(A, Bh, seeds, h, wb, "em")
        y = np.concatenate([y_a, y_b])
        cf[h] = {}
        for k in ("raw", "ts", "bcts", "rank", "em", "blend"):
            C = np.concatenate([on_a[k][0], on_b[k][0]], axis=1)
            cf[h][k] = statistics.mean(f1(y, c) for c in C)
        cf[h]["blend_rank"] = statistics.mean(f1(y, c) for c in np.concatenate([on_a["blend_mix"][0], on_b["blend_mix"][0]], axis=1))
        cf[h]["blend_em"] = statistics.mean(f1(y, c) for c in np.concatenate([on_a_em["blend_mix"][0], on_b_em["blend_mix"][0]], axis=1))
    mean3 = {k: statistics.mean(cf[h][k] for h in HS) for k in cf[1]}
    mix = "rank" if mean3["rank"] >= mean3["em"] else "em"
    w = {h: choose_w(es, seeds, h) for h in HS}
    res["inner_holdout_cross_fitted"] = {"per_horizon": {f"h{h}": {k: round(v, 4) for k, v in cf[h].items()} for h in HS},
                                         "three_horizon_mean": {k: round(v, 4) for k, v in mean3.items()},
                                         "note": "raw is optimistic here: v2.03's kept epoch was chosen on this hold-out"}
    res["choices"] = {"call_mix": mix, "rule": "rank vs em: higher cross-fitted 3-horizon mean on the inner hold-out",
                      "blend_w": {f"h{h}": {"w_deep": w[h][0], "inner_holdout_f1": w[h][1]} for h in HS}}
    say(f"choices on the inner hold-out: call-mix = {mix} (rank {mean3['rank']:.4f} vs em {mean3['em']:.4f}); "
        f"blend deep weight w = {[w[h][0] for h in HS]}")

    # 2) validation and 2026: parameters fitted on the whole inner hold-out
    order = ["raw", "ts", "bcts", "rank", "em", "blend", f"blend_{mix}"]
    for part_name, df in (("validation", val), ("forward_2026_seen", fwd)):
        out = {}
        for h in HS:
            mc = method_calls(es, df, seeds, h, w[h][0], mix)
            mc[f"blend_{mix}"] = mc.pop("blend_mix")
            y = df[f"y_{h}"].to_numpy(int)
            past = L.classify(df[f"r_{h}"].to_numpy(), df[f"sigma_{h}"].to_numpy(), kk[h])
            calls = {k: mc[k][0] for k in order}
            xgb = np.stack([probs(df, "xgb", s, h).argmax(1) for s in seeds])
            calls |= {"xgb_fit_rows": xgb, "trend_continues": past[None, :], "trend_reverses": (2 - past)[None, :]}
            out[f"h{h}"] = summarise(y, df["date"].to_numpy(), calls, {k: mc[k][1] for k in order}, bar=PASS_BAR[h])
            out[f"h{h}"]["actual_shares"] = [round(float((y == i).mean()), 4) for i in range(3)]
        res[part_name] = out

    # 3) adopt rule (validation only)
    adopt = {}
    for k in order[1:]:
        if k == "ts":
            better = all(res["validation"][f"h{h}"]["ts"]["log_loss"] < res["validation"][f"h{h}"]["raw"]["log_loss"] for h in HS)
            adopt[k] = {"adopted": better, "rule": "lowers validation log-loss at every horizon (calls unchanged)"}
            continue
        a1 = mean3[k] > mean3["raw"]
        a2 = all(res["validation"][f"h{h}"][k]["macro_f1"] >= res["validation"][f"h{h}"]["raw"]["macro_f1"] for h in HS)
        adopt[k] = {"adopted": bool(a1 and a2), "inner_holdout_above_raw": bool(a1), "validation_not_below_raw_any_horizon": bool(a2)}
    res["adopt"] = adopt

    # 4) news placebo
    plc = {}
    for per, base in (("val", val), ("fwd", fwd)):
        shuf = [parts[f"{per}_placebo{k}"] for k in range(1, PLACEBO_SHUFFLES + 1) if f"{per}_placebo{k}" in parts]
        if f"{per}_blank" not in parts or not shuf:
            continue
        out = {}
        for h in HS:
            y = base[f"y_{h}"].to_numpy(int)
            for d in [parts[f"{per}_blank"]] + shuf:
                assert (d[["symbol", "date"]].to_numpy() == base[["symbol", "date"]].to_numpy()).all()
            on = np.stack([probs(base, "v2xx", s, h).argmax(1) for s in seeds])
            bl = np.stack([probs(parts[f"{per}_blank"], "v2xx", s, h).argmax(1) for s in seeds])
            pl = np.stack([probs(d, "v2xx", s, h).argmax(1) for d in shuf for s in seeds])
            boot = PH.block_bootstrap(base["date"].to_numpy(), y, {"news_on": on, "placebo": pl, "blanked": bl},
                                      pairs=[("news_on", "placebo"), ("placebo", "blanked"), ("news_on", "blanked")])
            out[f"h{h}"] = {k: {"macro_f1": boot["values"][k]["point"], "ci95": boot["values"][k]["ci95"],
                                "call_shares": [round(float((C == i).mean()), 4) for i in range(3)]}
                            for k, C in (("news_on", on), ("placebo", pl), ("blanked", bl))}
            out[f"h{h}"]["placebo_per_shuffle"] = [round(statistics.mean(f1(y, pl[i * len(seeds) + j]) for j in range(len(seeds))), 4)
                                                   for i in range(len(shuf))]
            out[f"h{h}"]["differences"] = boot["differences"]
            out[f"h{h}"]["shuffles_scored"] = len(shuf)
        plc["validation" if per == "val" else "forward_2026_seen"] = out
    res["news_placebo"] = plc

    P["summary"].write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    report(res, order)
    say(f"\nwrote {P['summary']}")


def report(res, order):
    say(f"\n{NAME}: VALIDATION macro-F1 (seed mean; parameters fitted on the inner hold-out only)")
    say(f"{'method':16s}" + "".join(f"{f'{h} d':>22s}" for h in HS) + f"{'adopted':>10s}")
    for k in order + ["xgb_fit_rows", "trend_continues", "trend_reverses"]:
        row = f"{k:16s}"
        for h in HS:
            r = res["validation"][f"h{h}"][k]
            d = r.get("minus_raw")
            row += f"{r['macro_f1']:>9.4f}" + (f" ({d['point']:+.4f}{'*' if d['ci95'][0] > 0 or d['ci95'][1] < 0 else ' '})" if d else " " * 11) + "  "
        a = res["adopt"].get(k, {}).get("adopted")
        say(row + f"{'' if a is None else ('yes' if a else 'no'):>8s}")
    say("(change vs raw; * = 95% date-block interval excludes 0)   pass bar " + " / ".join(str(PASS_BAR[h]) for h in HS))
    say("log-loss on validation (raw -> ts -> bcts): " + "; ".join(
        f"h{h} {res['validation'][f'h{h}']['raw']['log_loss']:.4f} -> {res['validation'][f'h{h}']['ts']['log_loss']:.4f} -> "
        f"{res['validation'][f'h{h}']['bcts']['log_loss']:.4f}" for h in HS))
    say("calls down/flat/up on validation (raw -> chosen call-mix) vs actual: " + "; ".join(
        f"h{h} {res['validation'][f'h{h}']['raw']['call_shares']} -> {res['validation'][f'h{h}'][order[-1]]['call_shares']} vs "
        f"{res['validation'][f'h{h}']['actual_shares']}" for h in HS))
    say(f"\n2026 (SEEN window; reported only): " + "; ".join(
        f"h{h} raw {res['forward_2026_seen'][f'h{h}']['raw']['macro_f1']:.4f}, {order[-1]} {res['forward_2026_seen'][f'h{h}'][order[-1]]['macro_f1']:.4f}"
        for h in HS))
    for per, out in res["news_placebo"].items():
        say(f"\nNEWS PLACEBO, {per}: macro-F1 news on / placebo ({out['h1']['shuffles_scored']} shuffles) / blanked; news on - placebo (95% interval)")
        for h in HS:
            r = out[f"h{h}"]; d = r["differences"]["news_on - placebo"]
            say(f"  h{h}: {r['news_on']['macro_f1']:.4f} / {r['placebo']['macro_f1']:.4f} / {r['blanked']['macro_f1']:.4f}; "
                f"{d['point']:+.4f} {d['ci95']}; calls on {r['news_on']['call_shares']} placebo {r['placebo']['call_shares']} "
                f"blanked {r['blanked']['call_shares']}")


# ======================================================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["prepare", "predict", "trees", "analyze", "all"])
    ap.add_argument("--run-json", default=str(ROOT / "results/v2xx/A100/v2_03.json"))
    ap.add_argument("--weights", default=None, help="default: <run>_weights/ beside the run JSON")
    ap.add_argument("--fwd-inputs", default=str(ROOT / "results/forward_2026/inputs"))
    ap.add_argument("--news-fwd", default=str(ROOT / "data/derived/sentiment_scores_fwd2026.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results/v2_03beta"))
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--core-only", action="store_true", help="predict: inner hold-out, validation and 2026 only (no placebo)")
    ap.add_argument("--smoke-rows", type=int, default=0, help="plumbing test only: this many rows per part")
    args = ap.parse_args(argv)
    rj = Path(args.run_json)
    args.weights = args.weights or str(rj.with_name(rj.stem + "_weights"))
    stages = ["prepare", "predict", "trees", "analyze"] if args.stage == "all" else [args.stage]
    import importlib.util
    need = {"sklearn"} | ({"torch"} if "predict" in stages else set()) | ({"xgboost"} if "trees" in stages else set()) | \
        ({"scipy"} if "analyze" in stages else set())
    miss = sorted(m for m in need if importlib.util.find_spec(m) is None)
    if miss:
        raise SystemExit(f"stop: this Python lacks {miss}. Install with: pip install "
                         + " ".join({"sklearn": "scikit-learn"}.get(m, m) for m in miss))
    for st in stages:
        {"prepare": prepare, "predict": predict, "trees": trees, "analyze": analyze}[st](args)


if __name__ == "__main__":
    main()
