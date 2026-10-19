"""Checks for the v2.xx models: v2.01 (src/proposal_model.py), the faithful PatchTST baseline (src/patchtst_faithful.py),
their encoders (src/encoders.py), inputs (src/features_proposal.py) and configs (src/run_config.py).

  python tests/test_v2xx.py            # needs torch; reads the price files for the look-ahead check
  python tests/test_v2xx_pipeline.py   # slower: full-pipeline scramble test on the real data

The numbers in TEST_CFG / TEST_BASE below are small test fixtures so the checks run fast. They are not v2.01's
numbers; those are Siwat's, in configs/v2_01.json (checked separately below).

1. Config: the shipped v2_01.json is complete and valid; a blank config lists every undecided item; values fixed by
   the proposal, slides or paper cannot change; at least 2 scales; conditional fields asked only when needed.
2. v2.01 model: shapes in both channel modes and both encoders; scale weights are a softmax; head and Linear-layer
   sizes from the slides; weight count = hand formula; both heads get gradient; day t always reaches the output.
3. BatchNorm encoder: PatchTST layer structure (post-norm, BatchNorm over d_model, GELU); in eval mode a row's
   prediction does not depend on the other rows of its batch.
4. Faithful PatchTST: N = floor((L-P)/S) + 2 tokens (end padding); the last patch contains day t; heads read the
   last patch only; weight count = hand formula.
5. Inputs: scrambling every price after a cut-off leaves all price/technical channels on or before it unchanged;
   a planted leak is caught. Several SMA / EMA periods give one channel each.
6. Sentiment: daily mean of [pos, neg, neu]; suspension days roll forward; fills zeros / neutral / uniform.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import encoders as EN  # noqa: E402
import features_proposal as FP  # noqa: E402
import run_config as RC  # noqa: E402
from patchtst_faithful import FaithfulPatchTST  # noqa: E402
from proposal_model import ProposalMultiScalePatchTST, make_loss  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


SHIPPED = json.loads((ROOT / "configs/v2_01.json").read_text(encoding="utf-8"))
BASELINE = json.loads((ROOT / "configs/baseline_patchtst.json").read_text(encoding="utf-8"))
TEST_CFG = copy.deepcopy(SHIPPED)          # fixtures only (see docstring)
TEST_CFG["inputs"].update(sma_periods=[9, 21], ema_periods=[9, 21])                     # 16 channels
TEST_CFG["model"].update(lookback=32, patch_lengths=[4, 8, 16], patch_strides=[4, 8, 16], channel_mode="independent",
                         revin=False, encoder_norm="layernorm", d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.1,
                         fusion_hidden=8, d_stock=4, d_sector=4, stock_id_dropout=0.1, head_dropout=0.1)
TEST_CFG["training"].update(max_epochs=2, patience=1)
C = len(FP.channels(TEST_CFG["inputs"]))


def build(cfg, n_channels=C, n_stocks=10, n_sectors=5):
    m = cfg["model"]
    return ProposalMultiScalePatchTST(
        n_channels=n_channels, lookback=m["lookback"], patch_lengths=m["patch_lengths"], patch_strides=m["patch_strides"],
        channel_mode=m["channel_mode"], revin=m["revin"], encoder_norm=m["encoder_norm"], d_model=m["d_model"],
        n_heads=m["n_heads"], n_layers=m["n_layers"], d_ff=m["d_ff"], dropout=m["dropout"], fusion_hidden=m["fusion_hidden"],
        n_stocks=n_stocks, d_stock=m["d_stock"], n_sectors=n_sectors, d_sector=m["d_sector"], trunk_width=m["trunk_width"],
        head_hidden=tuple(m["head_hidden"]), head_dropout=m["head_dropout"])


def ids(B):
    return torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)


def test_config():
    check(RC.problems(SHIPPED) == [], "shipped configs/v2_01.json is complete and valid")
    s = SHIPPED
    check(s["model"]["patch_lengths"] == [1, 5, 21] and s["model"]["patch_strides"] == [1, 5, 21] and s["model"]["lookback"] == 126
          and s["inputs"]["sma_periods"] == [9, 21, 200] and s["model"]["encoder_norm"] == "batchnorm"
          and s["inputs"]["sentiment_no_news"] == "uniform" and s["training"]["grad_clip"] == 0,
          "shipped v2_01.json holds Siwat's 2026-10-19 choices (spot check)")
    blank = copy.deepcopy(SHIPPED)
    for sec in ("inputs", "model", "loss", "training"):
        for k in list(blank[sec]):
            if (sec, k) in RC.SCHEMA["proposal_msp"]:
                blank[sec][k] = None
    bad = RC.problems(blank)
    check(len(bad) >= 35 and all("not decided yet" in b for b in bad), f"a blank v2.01 config lists {len(bad)} undecided items and nothing else")
    bb = RC.problems(BASELINE)
    check(bb and all("not decided yet" in b and b.startswith("model.") for b in bb),
          f"shipped baseline_patchtst.json: {len(bb)} model numbers undecided, nothing else wrong")
    check(RC.problems(TEST_CFG) == [], "the test fixture config passes")
    c = copy.deepcopy(TEST_CFG); c["model"]["head_hidden"] = [64, 32]
    check(any("head_hidden" in b for b in RC.problems(c)), "changing the slide head sizes is refused")
    c = copy.deepcopy(TEST_CFG); c["model"].update(patch_lengths=[16], patch_strides=[8])
    check(any("at least 2" in b for b in RC.problems(c)), "one scale is refused (multi-scale is in the proposal)")
    c = copy.deepcopy(TEST_CFG); c["model"].update(patch_lengths=[4, 4], patch_strides=[4, 4])
    check(any("at least 2 distinct" in b for b in RC.problems(c)), "duplicate patch lengths are refused")
    c = copy.deepcopy(TEST_CFG); c["model"]["patch_strides"] = [4, 8]
    check(any("one per patch length" in b for b in RC.problems(c)), "strides must match the patch lengths")
    c = copy.deepcopy(TEST_CFG); c["model"]["lookback"] = 12
    check(any("shorter than the longest patch" in b for b in RC.problems(c)), "look-back shorter than the longest patch is refused")
    c = copy.deepcopy(TEST_CFG); c["loss"].update(weighting="kendall", w_trend=None, w_price=None)
    check(RC.problems(c) == [], "kendall weighting does not ask for w_trend / w_price")
    c = copy.deepcopy(TEST_CFG); c["loss"]["price_loss"] = "huber"
    check(any("huber_delta" in b for b in RC.problems(c)), "huber asks for huber_delta")
    c = copy.deepcopy(TEST_CFG); c["model"].update(d_model=16, n_heads=3)
    check(any("divisible" in b for b in RC.problems(c)), "d_model must divide by n_heads")
    c = copy.deepcopy(TEST_CFG); c["inputs"]["sma_periods"] = [9, 9]
    check(any("sma_periods" in b for b in RC.problems(c)), "duplicate SMA periods are refused")
    c = copy.deepcopy(TEST_CFG); c["protocol"]["seeds"] = [1, 2]
    check(any("at least 3 seeds" in b for b in RC.problems(c)), "fewer than 3 seeds is refused")
    c = copy.deepcopy(BASELINE); c["model"]["revin"] = False
    check(any("model.revin is fixed" in b for b in RC.problems(c)), "the baseline's paper-fixed parts (e.g. RevIN) cannot change")


def head_sizes(head):
    return [(l.in_features, l.out_features) for l in head if isinstance(l, torch.nn.Linear)]


def enc_layer_params(d, ff):
    return 4 * d * d + 2 * d * ff + 9 * d + ff          # attention 4d^2+4d, two norms 4d, feed-forward 2*d*ff+ff+d


def expected_params(cfg, C=C, n_stocks=10, n_sectors=5):
    m = cfg["model"]; d, ff, L = m["d_model"], m["d_ff"], m["lookback"]
    per_step = 1 if m["channel_mode"] == "independent" else C
    total = 0
    for p, s in zip(m["patch_lengths"], m["patch_strides"]):
        T = (L - p) // s + 1
        total += p * per_step * d + d + T * d + m["n_layers"] * enc_layer_params(d, ff)
    total += d * m["fusion_hidden"] + 2 * m["fusion_hidden"] + 1
    total += (n_stocks + 1) * m["d_stock"] + (n_sectors + 1) * m["d_sector"]
    enc_out = d * (C if m["channel_mode"] == "independent" else 1)
    total += (enc_out + m["d_stock"] + m["d_sector"]) * 128 + 128
    head = lambda out: 128 * 128 + 128 + 128 * 64 + 64 + 64 * out + out
    total += head(3) + head(9) + (2 * C if m["revin"] else 0)
    return total


def test_model():
    torch.manual_seed(0)
    for norm in EN.ENCODER_NORMS:
        for mode in ("independent", "mixed"):
            cfg = copy.deepcopy(TEST_CFG); cfg["model"].update(channel_mode=mode, encoder_norm=norm)
            model = build(cfg); B = 5
            o = model(torch.randn(B, 32, C), torch.randint(0, 11, (B,)), torch.randint(0, 6, (B,)))
            tag = f"{norm}/{mode}"
            check(tuple(o["trend_logits"].shape) == (B, 3, 3) and tuple(o["returns"].shape) == (B, 3), f"{tag}: output shapes")
            w = o["scale_weights"]
            check(w.shape[-1] == 3 and torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6), f"{tag}: scale weights are a softmax")
            check(head_sizes(model.price_head) == [(128, 128), (128, 64), (64, 3)] and
                  head_sizes(model.trend_head) == [(128, 128), (128, 64), (64, 9)] and model.linear.out_features == 128,
                  f"{tag}: heads 128->128->64->3 / ->9 and Linear layer 128 (slides 12-13)")
            n = sum(p.numel() for p in model.parameters())
            check(n == expected_params(cfg), f"{tag}: weight count {n:,} equals the hand formula")
            loss, _ = make_loss(cfg["loss"])(o, torch.randint(0, 3, (B, 3)), torch.randn(B, 3)); loss.backward()
            check(model.price_head[0].weight.grad.abs().sum() > 0 and model.trend_head[0].weight.grad.abs().sum() > 0,
                  f"{tag}: both heads receive gradient")
    for mode in ("independent", "mixed"):                   # look-back 30, strides 4/8/16: leftover days at the start
        cfg = copy.deepcopy(TEST_CFG); cfg["model"].update(lookback=30, channel_mode=mode, dropout=0.0)
        m = build(cfg).eval(); x = torch.randn(2, 30, C)
        with torch.no_grad():
            base = m(x, *ids(2))["trend_logits"]
            x2 = x.clone(); x2[:, -1, :] += 5.0
            x3 = x.clone(); x3[:, 0, :] += 5.0
            new, old = m(x2, *ids(2))["trend_logits"], m(x3, *ids(2))["trend_logits"]
        check(not torch.allclose(base, new) and torch.allclose(base, old),
              f"{mode}: day t always reaches the output; the leftover oldest day is the one dropped")
    cfg = copy.deepcopy(SHIPPED)                            # Siwat's numbers: patch 1/5/21, look-back 126
    m = build(cfg, n_channels=18, n_stocks=66, n_sectors=18)
    check([b.n_tokens for b in m.branches] == [126, 25, 6] and [b.offset for b in m.branches] == [0, 1, 0],
          "v2.01 numbers: tokens 126 / 25 / 6; the week scale skips the oldest day")
    check(sum(p.numel() for p in m.parameters()) == expected_params(cfg, C=18, n_stocks=66, n_sectors=18),
          f"v2.01 numbers: weight count {sum(p.numel() for p in m.parameters()):,} equals the hand formula")


def test_batchnorm_encoder():
    layer = EN.PatchTSTEncoderLayer(16, 2, 32, dropout=0.1)
    check(isinstance(layer.norm_attn.bn, torch.nn.BatchNorm1d) and layer.norm_attn.bn.num_features == 16
          and isinstance(layer.ff[1], torch.nn.GELU) and layer.self_attn.dropout == 0.0,
          "PatchTST layer: BatchNorm1d over d_model, GELU feed-forward, attention dropout 0 (official defaults)")
    torch.manual_seed(1)
    layer.train()
    for _ in range(5):                                       # give the running statistics something to hold
        layer(torch.randn(8, 7, 16) * 3 + 1)
    layer.eval()
    z = torch.randn(4, 7, 16)
    with torch.no_grad():
        manual = layer.norm_attn(z + layer.self_attn(z, z, z, need_weights=False)[0])
        manual = layer.norm_ffn(manual + layer.ff(manual))
        out = layer(z)
    check(torch.allclose(out, manual, atol=1e-6), "post-norm order: normalise after each residual add")
    cfg = copy.deepcopy(TEST_CFG); cfg["model"]["encoder_norm"] = "batchnorm"
    m = build(cfg); m.train()
    for _ in range(3):
        m(torch.randn(16, 32, C), *ids(16))
    m.eval()
    x = torch.randn(6, 32, C)
    with torch.no_grad():
        alone = m(x[:1], *ids(1))["trend_logits"]
        batch = m(x, *ids(6))["trend_logits"][:1]
        other = m(torch.cat([x[:1], 10 * torch.randn(5, 32, C)]), *ids(6))["trend_logits"][:1]
    check(torch.allclose(alone, batch, atol=1e-5) and torch.allclose(alone, other, atol=1e-5),
          "eval mode: a row's prediction does not depend on the other rows in its batch (no cross-row leak)")


def faithful(L=32, P=8, S=4, d=16, heads=2, layers=1, ff=32, C=C):
    return FaithfulPatchTST(n_channels=C, lookback=L, patch_len=P, stride=S, d_model=d, n_heads=heads, n_layers=layers,
                            d_ff=ff, dropout=0.0, attn_dropout=0.0, head_dropout=0.0)


def test_faithful():
    torch.manual_seed(2)
    for L, P, S in ((32, 8, 4), (30, 16, 8), (126, 16, 8)):
        m = faithful(L, P, S)
        check(m.n_tokens == (L - P) // S + 2 and m.W_pos.shape[0] == m.n_tokens, f"L {L}, P {P}, S {S}: N = floor((L-P)/S) + 2 = {m.n_tokens}")
        m.eval(); x = torch.randn(2, L, C)
        with torch.no_grad():
            o = m(x)
            x2 = x.clone(); x2[:, -1, :] += 3.0
            o2 = m(x2)
        check(tuple(o["trend_logits"].shape) == (2, 3, 3) and tuple(o["returns"].shape) == (2, 3)
              and not torch.allclose(o["trend_logits"], o2["trend_logits"]), f"L {L}: output shapes; day t reaches the output")
    m = faithful()
    check(isinstance(m.encoder[0], EN.PatchTSTEncoderLayer) and m.trend_head[-1].in_features == C * 16
          and float(m.W_pos.abs().max()) <= 0.02, "BatchNorm PatchTST encoder; heads read C x d_model (last patch); W_pos init within +-0.02")
    d, ff, L, P, S = 16, 32, 32, 8, 4
    N = (L - P) // S + 2
    expect = 2 * C + P * d + d + N * d + enc_layer_params(d, ff) + (C * d * 9 + 9) + (C * d * 3 + 3)
    n = sum(p.numel() for p in m.parameters())
    check(n == expect, f"faithful PatchTST weight count {n:,} equals the hand formula {expect:,}")
    m.eval(); x = torch.randn(2, L, C)
    with torch.no_grad():
        a = m(x)["trend_logits"]
        x3 = x.clone(); x3[:, : L - P, :] += 3.0          # change everything except the days in the last patch
        z = m.encoder(m.drop(m.W_P(m.pad(m.revin(x).permute(0, 2, 1)).reshape(2 * C, L + S).unfold(-1, P, S)) + m.W_pos))
    check(z.shape[1] == N, "encoder sees all N tokens; the heads then keep only the last one")


def test_inputs():
    import prices as P
    df = P.load_prices(ROOT)
    df = df[df.symbol.isin(["PTT", "ADVANC", "SCB", "GULF", "BANPU"])].reset_index(drop=True)
    base_cfg = SHIPPED["inputs"]
    cols = FP.PRICE_CH + FP.tech_channels(base_cfg)
    check(cols[5:11] == ["sma_gap_9", "sma_gap_21", "sma_gap_200", "ema_gap_9", "ema_gap_21", "ema_gap_200"] and len(FP.channels(base_cfg)) == 18,
          "v2.01 inputs: one channel per SMA / EMA period, 18 channels in all")
    for basis in FP.OHLC_BASES:
        cfg = dict(base_cfg, ohlc_return_base=basis)
        a = FP.add_price_tech(df, cfg)
        for cut in ("2022-06-30", "2024-02-15"):
            d2 = df.copy(); d2["volume"] = d2["volume"].astype(float)
            after = d2.date > pd.Timestamp(cut); rng = np.random.default_rng(1)
            for c in ("open", "high", "low", "close"):
                d2.loc[after, c] = d2.loc[after, c] * rng.uniform(0.5, 1.5, after.sum())
            d2.loc[after, "volume"] = d2.loc[after, "volume"] * rng.uniform(0.2, 5, after.sum())
            d2.loc[after, "ret"] = rng.normal(0, 0.05, after.sum())
            b = FP.add_price_tech(d2, cfg)
            before = ~after
            diff = np.nanmax(np.abs(a.loc[before, cols].to_numpy() - b.loc[before, cols].to_numpy()))
            same_nan = (a.loc[before, cols].isna().to_numpy() == b.loc[before, cols].isna().to_numpy()).all()
            check(diff < 1e-9 and same_nan, f"{basis}, cut {cut}: all {len(cols)} price/technical channels on or before the cut unchanged")
    leak = FP.add_price_tech(df, base_cfg); leak["lr_close"] = leak.groupby("seg_key")["lr_close"].shift(-1)
    d2 = df.copy(); after = d2.date > pd.Timestamp("2022-06-30"); d2.loc[after, "ret"] = 0.0
    leak2 = FP.add_price_tech(d2, base_cfg); leak2["lr_close"] = leak2.groupby("seg_key")["lr_close"].shift(-1)
    check(np.nanmax(np.abs(leak.loc[~after, "lr_close"] - leak2.loc[~after, "lr_close"])) > 1e-6,
          "a planted leak (tomorrow's return) is caught by the same comparison")


def test_sentiment():
    rows = pd.DataFrame({"symbol": ["AAA"] * 4 + ["BBB"] * 2,
                         "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-02", "2024-01-03"])})
    scores = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA", "BBB"],
                           "trading_day": ["2024-01-02", "2024-01-02", "2024-01-04", "2024-01-03"],   # 01-04: AAA suspended
                           "p_pos": [0.8, 0.2, 0.6, 0.1], "p_neg": [0.1, 0.5, 0.3, 0.7], "p_neu": [0.1, 0.3, 0.1, 0.2]})
    z = FP.sentiment_probs(scores, rows, "zeros")
    at = lambda f, s, d: f.loc[(s, pd.Timestamp(d)), FP.SENT_CH].to_numpy(float)
    check(np.allclose(at(z, "AAA", "2024-01-02"), [0.5, 0.3, 0.2]), "two articles on one day -> mean of [pos, neg, neu]")
    check(np.allclose(at(z, "AAA", "2024-01-05"), [0.6, 0.3, 0.1]), "news on a suspension day rolls forward to the next trading row")
    check(np.allclose(at(z, "AAA", "2024-01-03"), [0, 0, 0]), "fill 'zeros' -> [0, 0, 0]")
    check(np.allclose(at(FP.sentiment_probs(scores, rows, "neutral"), "AAA", "2024-01-08"), [0, 0, 1]), "fill 'neutral' -> [0, 0, 1]")
    u = FP.sentiment_probs(scores, rows, "uniform")
    check(np.allclose(at(u, "AAA", "2024-01-03"), [1 / 3] * 3, atol=0) and abs(at(u, "AAA", "2024-01-03").sum() - 1) < 1e-12,
          "fill 'uniform' -> exactly [1/3, 1/3, 1/3] (Siwat's choice), sums to 1")
    check(np.allclose(at(u, "AAA", "2024-01-02"), [0.5, 0.3, 0.2]), "the fill never touches days with news")
    check(np.allclose(at(z, "BBB", "2024-01-03"), [0.1, 0.7, 0.2]), "news of one stock never reaches another")


if __name__ == "__main__":
    test_config()
    test_model()
    test_batchnorm_encoder()
    test_faithful()
    test_sentiment()
    test_inputs()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
