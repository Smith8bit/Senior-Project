"""Checks for the batch 5-7 training options in scripts/baselines_deep.py (needs torch; run on Colab or any torch install).

  python tests/test_train_options.py

1. Defaults build the E21 parent unchanged (472,572 weights with 32 inputs and 18 sectors).
2. E07 / E08 / E09 / E15 patch scales and E27 look-back 30 give the designed token counts; E17 shrinks only the
   input-mixing layer (trunk width stays 128).
3. E28 channel masking hides whole inputs (every day of the window) at about the set rate; E06 context masking zeroes
   only the oldest days and keeps at least context_min days; both are off by default.
4. E13 weighted loss equals the plain loss when all weights are 1, and down-weighting changes it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import baselines_deep as BD  # noqa: E402

FAILS = []
DEFAULT_OPT = dict(BD.OPT)


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def n(m):
    return sum(p.numel() for p in m.parameters())


def build(**opt):
    BD.OPT.clear(); BD.OPT.update(DEFAULT_OPT); BD.OPT.update(revin=False, cls_only=True, stop_rule="every", **opt)
    return BD.make_model("multiscale_sector", 32, 18, 0.2).eval()


def main():
    parent = build()
    check(n(parent) == 472_572, f"defaults rebuild the E21 parent: {n(parent):,} weights")
    toks = lambda m: [b.n_tokens for b in m.branches]   # noqa: E731
    check(toks(parent) == [12, 5], f"parent tokens per scale {toks(parent)}")
    for label, patches, want in [("E07", ((2, 1), (20, 10)), [59, 5]), ("E08", ((5, 5), (20, 5)), [12, 9]),
                                 ("E09", ((5, 5), (10, 5)), [12, 11]), ("E15", ((16, 8),), [6])]:
        m = build(patches=patches)
        check(toks(m) == want, f"{label} patches {patches}: tokens {toks(m)} (designed {want}); {n(m):,} weights")
    BD.LOOKBACK = 30
    m = build()
    check(toks(m) == [6, 2], f"E27 look-back 30: tokens {toks(m)}; {n(m):,} weights")
    BD.LOOKBACK = 60
    e17 = build(d_channel=16, trunk_width=128)
    check(e17.trunk[0].in_features == 32 * 16 + 16 and e17.trunk[0].out_features == 128,
          f"E17: input-mixing layer {e17.trunk[0].in_features} -> {e17.trunk[0].out_features}; {n(e17):,} weights")
    x, g = torch.randn(4, 60, 32), torch.randint(0, 19, (4,))
    with torch.no_grad():
        check(e17(x, g)["trend_logits"].shape == (4, 3, 3), "E17 forward pass works")

    # 3) masking
    gen = torch.Generator().manual_seed(0)
    ctx = np.random.default_rng(1)
    xb = torch.ones(2000, 60, 32)
    build()
    check(BD.augment(xb, gen, ctx).equal(xb), "masking is off by default")
    build(chan_mask=0.15)
    out = BD.augment(xb, gen, ctx)
    per_input = out.mean(1)                     # 1 = kept, 0 = hidden for the whole window
    check(bool(((per_input == 0) | (per_input == 1)).all()), "E28 hides whole inputs (every day of the window)")
    rate = float((per_input == 0).float().mean())
    check(0.13 < rate < 0.17, f"E28 hidden share {rate:.3f} (set 0.15)")
    build(context_min=20)
    kept = []
    for _ in range(200):
        o = BD.augment(xb[:2], gen, ctx)
        day_on = o[0, :, 0] != 0
        first = int(day_on.float().argmax())
        check_ok = bool(day_on[first:].all()) and 60 - first >= 20
        kept.append(60 - first)
        if not check_ok:
            break
    check(check_ok, "E06 zeroes only the oldest days and keeps the last m >= 20 days")
    check(min(kept) <= 25 and max(kept) == 60, f"E06 kept lengths span {min(kept)}..{max(kept)}")
    check(xb.sum() == xb.numel(), "masking never changes the caller's tensor")

    # 4) weighted loss
    build()
    loss = BD.ClsOnlyLoss()
    out = {"trend_logits": torch.randn(8, 3, 3)}
    y = torch.randint(0, 3, (8, 3))
    plain = loss(out, y, None)[0]
    ones = loss(out, y, None, torch.ones(8))[0]
    check(torch.allclose(plain, ones, atol=1e-6), f"E13 weights of 1 give the plain loss ({float(plain):.5f} vs {float(ones):.5f})")
    w = torch.tensor([0.5] * 4 + [1.0] * 4)
    check(not torch.allclose(plain, loss(out, y, None, w)[0]), "E13 down-weighting changes the loss")
    BD.OPT.clear(); BD.OPT.update(DEFAULT_OPT)
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
