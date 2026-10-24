"""Run config for the v2.xx experiments, main model v3 and the faithful PatchTST baseline (configs/*.json).

Every number is chosen by Siwat. A value of null means "not decided yet": `load()` then refuses to run and lists
every missing field with what it accepts. Values fixed by the proposal, the slides or the original paper are
filled in the config file itself and checked here, so they cannot drift silently.

model.family selects the architecture:
  proposal_msp      : the committee-slide Multi-Scale PatchTST (src/proposal_model.py)       -> v2.01 onwards
  patchtst_faithful : PatchTST as in Nie et al. 2023 and the official code (src/patchtst_faithful.py) -> baseline
"""
from __future__ import annotations

import json
from pathlib import Path

POS_INT, POS_NUM, NONNEG_NUM, PROB, BOOL = "positive integer", "positive number", "number >= 0", "number in [0, 1)", "true or false"
INT_LIST = "non-empty list of distinct positive integers"
OPEN01 = "number strictly between 0 and 1"
FAMILIES = ("proposal_msp", "patchtst_faithful")

# (section, key): (accepts, required_when) ; required_when = None -> always required
COMMON = {
    ("inputs", "ohlc_return_base"): (("own_previous", "previous_close"), None),
    ("inputs", "sma_periods"): (INT_LIST, None),
    ("inputs", "ema_periods"): (INT_LIST, None),
    ("inputs", "rsi_period"): (POS_INT, None),
    ("inputs", "macd_fast"): (POS_INT, None),
    ("inputs", "macd_slow"): (POS_INT, None),
    ("inputs", "bb_period"): (POS_INT, None),
    ("inputs", "bb_width_k"): (POS_NUM, None),
    ("inputs", "adx_period"): (POS_INT, None),
    ("inputs", "sentiment_no_news"): (("zeros", "neutral", "uniform"), None),
    ("model", "lookback"): (POS_INT, None),
    ("model", "d_model"): (POS_INT, None),
    ("model", "n_heads"): ("positive integer dividing d_model", None),
    ("model", "n_layers"): (POS_INT, None),
    ("model", "d_ff"): (POS_INT, None),
    ("model", "dropout"): (PROB, None),
    ("model", "head_dropout"): (PROB, None),
    ("loss", "weighting"): (("kendall", "fixed"), None),
    ("loss", "w_trend"): (POS_NUM, ("weighting", "fixed")),
    ("loss", "w_price"): (POS_NUM, ("weighting", "fixed")),
    ("loss", "price_loss"): (("mse", "huber"), None),
    ("loss", "huber_delta"): (POS_NUM, ("price_loss", "huber")),
    ("training", "lr"): (POS_NUM, None),
    ("training", "weight_decay"): (NONNEG_NUM, None),
    ("training", "batch_size"): (POS_INT, None),
    ("training", "max_epochs"): (POS_INT, None),
    ("training", "grad_clip"): ("positive number, or 0 for none", None),
    ("training", "checkpoint"): (("shared", "per_horizon"), None),
    # v2.02 options (Siwat, 2026-10-25); configs/v2_01.json states the v2.01 behaviour explicitly
    ("training", "early_stopping"): (BOOL, None),
    ("training", "stop_rule"): (("mean", "every"), ("early_stopping", True)),
    ("training", "patience"): (POS_INT, ("early_stopping", True)),
    ("training", "selection"): (("f1_mean", "f1_up_to_ce_min"), None),
    ("training", "lr_schedule"): (("constant", "cosine"), None),
    ("training", "lr_min"): (NONNEG_NUM, ("lr_schedule", "cosine")),
    ("training", "weight_averaging"): (("none", "ema"), None),
    ("training", "ema_decay"): (OPEN01, ("weight_averaging", "ema")),
    ("report", "seed_ensemble"): (BOOL, None),
}

SCHEMA = {
    "proposal_msp": COMMON | {
        ("model", "patch_lengths"): ("list of at least 2 distinct positive integers", None),
        ("model", "patch_strides"): ("list of positive integers, one per patch length", None),
        ("model", "channel_mode"): (("independent", "mixed"), None),
        ("model", "revin"): (BOOL, None),
        ("model", "encoder_norm"): (("layernorm", "batchnorm"), None),
        ("model", "fusion_hidden"): (POS_INT, None),
        ("model", "d_stock"): (POS_INT, None),
        ("model", "d_sector"): (POS_INT, None),
        ("model", "stock_id_dropout"): (PROB, None),
    },
    "patchtst_faithful": COMMON | {
        ("model", "patch_len"): (POS_INT, None),
        ("model", "stride"): (POS_INT, None),
        ("model", "attn_dropout"): (PROB, None),
    },
}

_INPUTS_FIXED = {
    ("inputs", "price_channels"): ["open", "high", "low", "close", "volume"],                   # slide 9
    ("inputs", "indicators"): ["sma", "ema", "rsi", "macd", "bb_pctb", "adx"],                  # proposal scope 2
    ("inputs", "sentiment"): ["p_pos", "p_neg", "p_neu"],                                       # slide 9
    ("loss", "price_target"): "log_return",                                                     # slide 9, hard rule 1
    ("protocol", "horizons"): [1, 5, 10],                                                       # proposal scope 4
}
# Fixed by the signed proposal, the committee slides, the original paper, or earlier pre-registered protocol.
FIXED = {
    "proposal_msp": _INPUTS_FIXED | {
        ("inputs", "stock_identity"): ["stock_id", "sector_id"],                                # slide 9
        ("model", "pooling"): "mean",                                                           # slide 11
        ("model", "scale_fusion"): "attention",                                                 # slides 11, 19
        ("model", "trunk_width"): 128,                                                          # slides 12-13
        ("model", "head_hidden"): [128, 64],                                                    # slides 12-13
    },
    "patchtst_faithful": _INPUTS_FIXED | {                                                      # Nie et al. 2023 + official code
        ("model", "channel_mode"): "independent",
        ("model", "revin"): True,
        ("model", "padding"): "end",
        ("model", "encoder_norm"): "batchnorm",
        ("model", "head"): "last_patch_flatten_linear",
    },
}


def _ok(accepts, v) -> bool:
    if isinstance(accepts, tuple):
        return v in accepts
    if accepts == POS_INT:
        return isinstance(v, int) and not isinstance(v, bool) and v > 0
    if accepts == POS_NUM:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    if accepts == NONNEG_NUM:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
    if accepts == PROB:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < 1
    if accepts == BOOL:
        return isinstance(v, bool)
    if accepts == OPEN01:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < 1
    if accepts == INT_LIST:
        return (isinstance(v, list) and len(v) > 0 and len(set(v)) == len(v)
                and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in v))
    return True                                       # list fields with special rules are checked in problems()


def _blank(v) -> bool:
    return v is None or (isinstance(v, list) and len(v) > 0 and all(x is None for x in v))


def problems(cfg: dict) -> list[str]:
    """Every problem with the config, as readable lines (empty list = ready to run)."""
    fam = cfg.get("model", {}).get("family")
    if fam not in FAMILIES:
        return [f"model.family must be one of {FAMILIES} (got {fam!r})"]
    out = []
    for (sec, key), want in FIXED[fam].items():
        got = cfg.get(sec, {}).get(key)
        if got != want:
            out.append(f"{sec}.{key} is fixed (proposal / slides / paper) at {want!r}; the file has {got!r}")
    for (sec, key), (accepts, when) in SCHEMA[fam].items():
        if when is not None and cfg.get(sec, {}).get(when[0]) != when[1]:
            continue                                  # only needed with another choice
        v = cfg.get(sec, {}).get(key)
        if _blank(v):
            out.append(f"{sec}.{key}: not decided yet (accepts {accepts if isinstance(accepts, str) else ' | '.join(map(str, accepts))})")
        elif not _ok(accepts, v):
            out.append(f"{sec}.{key} = {v!r} is not valid (accepts {accepts})")
    m = cfg.get("model", {})
    lb = m.get("lookback")
    if fam == "proposal_msp":
        pl, st = m.get("patch_lengths"), m.get("patch_strides")
        if not _blank(pl):
            if not (isinstance(pl, list) and len(pl) >= 2 and len(set(pl)) == len(pl)
                    and all(isinstance(p, int) and not isinstance(p, bool) and p > 0 for p in pl)):
                out.append("model.patch_lengths must be at least 2 distinct positive integers (multi-scale, proposal objective 1)")
            elif isinstance(lb, int) and lb < max(pl):
                out.append(f"model.lookback {lb} is shorter than the longest patch ({max(pl)})")
        if not _blank(st) and isinstance(pl, list):
            if not (isinstance(st, list) and len(st) == len(pl) and all(isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in st)):
                out.append(f"model.patch_strides must be {len(pl)} positive integers, one per patch length {pl}")
    else:
        if isinstance(lb, int) and isinstance(m.get("patch_len"), int) and m["patch_len"] > lb:
            out.append(f"model.patch_len {m['patch_len']} is longer than the look-back {lb}")
    if isinstance(m.get("d_model"), int) and isinstance(m.get("n_heads"), int) and m["n_heads"] > 0 and m["d_model"] % m["n_heads"]:
        out.append(f"model.d_model {m['d_model']} is not divisible by model.n_heads {m['n_heads']}")
    i = cfg.get("inputs", {})
    if isinstance(i.get("macd_fast"), int) and isinstance(i.get("macd_slow"), int) and i["macd_fast"] >= i["macd_slow"]:
        out.append("inputs.macd_fast must be smaller than inputs.macd_slow")
    gc = cfg.get("training", {}).get("grad_clip")
    if gc is not None and not (isinstance(gc, (int, float)) and not isinstance(gc, bool) and gc >= 0):
        out.append("training.grad_clip must be a number >= 0 (0 = no clipping)")
    t = cfg.get("training", {})
    if t.get("selection") == "f1_up_to_ce_min":
        if t.get("checkpoint") not in (None, "shared"):
            out.append("training.selection f1_up_to_ce_min works with checkpoint shared only")
        if t.get("early_stopping") is True:
            out.append("training.selection f1_up_to_ce_min is used without early stopping (training.early_stopping false)")
    if t.get("lr_schedule") == "cosine" and t.get("early_stopping") is True:
        out.append("training.lr_schedule cosine runs the full schedule, so training.early_stopping must be false")
    if (t.get("lr_schedule") == "cosine" and isinstance(t.get("lr_min"), (int, float)) and isinstance(t.get("lr"), (int, float))
            and t["lr_min"] >= t["lr"]):
        out.append("training.lr_min must be smaller than training.lr")
    p = cfg.get("protocol", {})
    if len(p.get("seeds") or []) < 3:
        out.append("protocol.seeds: at least 3 seeds (hard rule 5)")
    for key in ("inner_es_share", "purge"):
        if p.get(key) is None:
            out.append(f"protocol.{key} is missing")
    return out


def load(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    bad = problems(cfg)
    if bad:
        raise SystemExit(f"config {path} is not ready to run ({len(bad)} item(s)):\n  - " + "\n  - ".join(bad))
    return cfg
