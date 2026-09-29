"""
Lightweight self-supervised spatial-refinement network.

The harmonic/phenology model (`phenology.py`) already gives a physically
sensible *expected* value for every cloudy pixel, but it's fit independently
per pixel through time, so it has no notion of local spatial texture (field
boundaries, roads, individual tree crowns) -- reconstructed patches can look
slightly flat/blurred compared to their genuinely-observed surroundings.

This module adds a small (~0.4M-param) Partial-Convolution U-Net (Liu et al.
2018, "Image Inpainting for Irregular Holes Using Partial Convolutions") that
takes [masked reflectance, validity mask, phenology-curve guide] and predicts
a spatially-textured reflectance image. Partial convolutions renormalize each
conv by how much *valid* input actually fell under the kernel, so holes are
filled from genuine spatial context, not corrupted by the zeros we pad them
with, and the network correctly propagates validity inward layer by layer.

Training is self-supervised and per-scene (test-time training): we never
have ground truth for the actual clouds, so instead we repeatedly sample
patches that are already (mostly) cloud-free, synthetically punch random
cloud-shaped holes into them, and train the network to reconstruct the
hidden pixels from context + the phenology guide. No external labeled
dataset or pretrained weights are used -- everything the network learns
comes from this one AOI's own clear observations, across all 12 months.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Conv2d):
    def forward(self, x, mask):
        # mask: (B,1,H,W) in {0,1}
        with torch.no_grad():
            ones_weight = torch.ones(1, mask.shape[1], *self.kernel_size, device=x.device)
            mask_sum = F.conv2d(mask, ones_weight, stride=self.stride, padding=self.padding)
            window_size = ones_weight.numel()
            mask_ratio = window_size / mask_sum.clamp(min=1e-6)
            new_mask = (mask_sum > 0).float()

        raw_out = F.conv2d(x * mask, self.weight, self.bias, self.stride, self.padding)
        if self.bias is not None:
            bias = self.bias.view(1, -1, 1, 1)
            out = (raw_out - bias) * mask_ratio + bias
        else:
            out = raw_out * mask_ratio
        out = out * new_mask
        return out, new_mask


class PCBlock(nn.Module):
    """Partial-conv + GroupNorm (not BatchNorm: with holes covering anywhere
    from 0-90% of a tile, per-batch activation statistics swing wildly, and
    BatchNorm's running stats made inference on largely-invalid tiles produce
    the garbage seen in the first attempt at this. GroupNorm normalizes each
    sample independently, so it stays well-behaved regardless of hole size.
    """

    def __init__(self, cin, cout, down=False):
        super().__init__()
        stride = 2 if down else 1
        self.pconv = PartialConv2d(cin, cout, kernel_size=3, stride=stride, padding=1)
        self.gn = nn.GroupNorm(min(8, cout), cout)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.act(self.gn(x))
        return x, mask


class PCUNet(nn.Module):
    """Small 3-level partial-conv U-Net. ~0.4M params.

    Predicts a *bounded residual* around the phenology-curve guide rather
    than raw reflectance: out = guide + tanh(delta) * residual_scale. This
    is the key robustness fix over a naive from-scratch predictor -- even an
    undertrained network, or a tile where almost nothing is spatially valid,
    can only ever nudge the output slightly away from the already-sensible
    temporal estimate, never produce an unbounded/degenerate color.
    """

    def __init__(self, in_ch=9, out_ch=4, base=24, residual_scale=0.15):
        super().__init__()
        self.residual_scale = residual_scale
        self.enc1 = PCBlock(in_ch, base, down=False)
        self.enc2 = PCBlock(base, base * 2, down=True)
        self.enc3 = PCBlock(base * 2, base * 4, down=True)
        self.bott = PCBlock(base * 4, base * 4, down=False)
        self.dec2 = PCBlock(base * 4 + base * 2, base * 2, down=False)
        self.dec1 = PCBlock(base * 2 + base, base, down=False)
        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, mask, guide):
        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        b, mb = self.bott(e3, m3)

        b_up = F.interpolate(b, size=e2.shape[-2:], mode="nearest")
        mb_up = F.interpolate(mb, size=e2.shape[-2:], mode="nearest")
        d2, md2 = self.dec2(torch.cat([b_up, e2], dim=1), torch.cat([mb_up, m2], dim=1)[:, :1])

        d2_up = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        md2_up = F.interpolate(md2, size=e1.shape[-2:], mode="nearest")
        d1, _ = self.dec1(torch.cat([d2_up, e1], dim=1), torch.cat([md2_up, m1], dim=1)[:, :1])

        delta = torch.tanh(self.out_conv(d1)) * self.residual_scale
        return guide + delta


def _random_blob_mask(h, w, n_blobs_range=(1, 3), size_frac_range=(0.15, 0.9), rng=None):
    """1 = hole (occluded), 0 = kept, mimicking irregular cloud shapes."""
    rng = rng or np.random.default_rng()
    mask = np.zeros((h, w), dtype=np.float32)
    n_blobs = rng.integers(*n_blobs_range, endpoint=True)
    for _ in range(n_blobs):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry = int(h * rng.uniform(*size_frac_range) / 2)
        rx = int(w * rng.uniform(*size_frac_range) / 2)
        yy, xx = np.ogrid[:h, :w]
        # random-ish blob via superellipse + noise on radius
        theta = rng.uniform(0, 2 * np.pi)
        wob = 1 + 0.35 * np.sin((np.arctan2(yy - cy, xx - cx) - theta) * rng.integers(2, 5))
        ellipse = ((xx - cx) / (rx * wob + 1e-3)) ** 2 + ((yy - cy) / (ry * wob + 1e-3)) ** 2
        mask[ellipse <= 1] = 1.0
    return mask


def train_refiner(
    refl, weight, recon_curve, n_iters=400, patch=192, batch=4, base=24,
    lr=2e-3, device="cpu", seed=0, log_every=100,
):
    """refl,weight,recon_curve: (T,4,H,W)/(T,H,W)/(T,4,H,W) numpy.
    Self-supervised training entirely on this AOI's own clear pixels.
    Returns the trained model.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    T, C, H, W = refl.shape
    valid = (weight >= 0.5).astype(np.float32)  # (T,H,W) confidently-clear mask

    model = PCUNet(in_ch=2 * C + 1, out_ch=C, base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Precompute per-month coverage so we can bias sampling toward months/
    # regions that actually have enough clear context to learn from.
    def sample_patch():
        for _ in range(20):
            t = rng.integers(0, T)
            y0 = rng.integers(0, H - patch)
            x0 = rng.integers(0, W - patch)
            v = valid[t, y0:y0 + patch, x0:x0 + patch]
            if v.mean() > 0.85:
                return t, y0, x0
        return t, y0, x0  # fall back to whatever the last try was

    model.train()
    losses = []
    for it in range(n_iters):
        xb, mb, yb = [], [], []
        for _ in range(batch):
            t, y0, x0 = sample_patch()
            r = refl[t, :, y0:y0 + patch, x0:x0 + patch]
            v = valid[t, y0:y0 + patch, x0:x0 + patch]
            g = recon_curve[t, :, y0:y0 + patch, x0:x0 + patch]
            hole = _random_blob_mask(patch, patch, rng=rng)
            keep_mask = v * (1 - hole)  # what the network is allowed to see
            loss_mask = v * hole        # where we can score it (real+hidden)

            x_in = np.concatenate([r * keep_mask, keep_mask[None], g], axis=0)
            xb.append(x_in)
            mb.append(keep_mask[None])
            yb.append(np.concatenate([r, loss_mask[None]], axis=0))

        x_in = torch.from_numpy(np.stack(xb)).float().to(device)
        mask_in = torch.from_numpy(np.stack(mb)).float().to(device)
        mask_in_full = mask_in.repeat(1, x_in.shape[1], 1, 1)
        target = torch.from_numpy(np.stack(yb)).float().to(device)
        y_true, loss_mask_t = target[:, :C], target[:, C:C + 1]
        guide_t = x_in[:, -C:]  # the recon_curve channels we concatenated into x_in

        pred = model(x_in, mask_in_full, guide_t)
        diff = (pred - y_true).abs() * loss_mask_t
        loss = diff.sum() / loss_mask_t.sum().clamp(min=1.0)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (it + 1) % log_every == 0:
            print(f"    iter {it+1}/{n_iters}  loss={np.mean(losses[-log_every:]):.5f}")

    model.eval()
    return model


@torch.no_grad()
def apply_refiner(model, refl, weight, recon_curve, device="cpu", tile=512, overlap=64):
    """Run the trained refiner over the full AOI, tiled to bound memory.
    Returns refined: (T,4,H,W) numpy, valid everywhere (used only where the
    real observation is not confidently clear -- composition happens in
    compose.py).
    """
    T, C, H, W = refl.shape
    valid = (weight >= 0.5).astype(np.float32)
    refined = np.zeros_like(refl)

    step = tile - overlap
    for t in range(T):
        acc = np.zeros((C, H, W), dtype=np.float32)
        wsum = np.zeros((H, W), dtype=np.float32)
        for y0 in range(0, H, step):
            y1 = min(y0 + tile, H)
            for x0 in range(0, W, step):
                x1 = min(x0 + tile, W)
                r = refl[t, :, y0:y1, x0:x1]
                v = valid[t, y0:y1, x0:x1]
                g = recon_curve[t, :, y0:y1, x0:x1]
                x_in = np.concatenate([r * v, v[None], g], axis=0)[None]
                m_in = np.repeat(v[None, None], x_in.shape[1], axis=1)
                x_in_t = torch.from_numpy(x_in).float().to(device)
                m_in_t = torch.from_numpy(m_in).float().to(device)
                guide_t = torch.from_numpy(g[None]).float().to(device)
                pred = model(x_in_t, m_in_t, guide_t)[0].cpu().numpy()

                h, w = y1 - y0, x1 - x0
                wy = np.ones(h, dtype=np.float32)
                wx = np.ones(w, dtype=np.float32)
                # simple cosine feather on overlap edges to blend tiles
                fe = min(overlap, h // 2, w // 2)
                if fe > 0:
                    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fe))
                    wy[:fe] = np.minimum(wy[:fe], ramp)
                    wy[-fe:] = np.minimum(wy[-fe:], ramp[::-1])
                    wx[:fe] = np.minimum(wx[:fe], ramp)
                    wx[-fe:] = np.minimum(wx[-fe:], ramp[::-1])
                blend = np.outer(wy, wx).astype(np.float32)

                acc[:, y0:y1, x0:x1] += pred * blend[None]
                wsum[y0:y1, x0:x1] += blend
        refined[t] = acc / np.clip(wsum, 1e-6, None)[None]

    return refined