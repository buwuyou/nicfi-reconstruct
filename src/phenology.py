"""
Phenology-aware temporal reconstruction.

Model: per-pixel, per-band harmonic (Fourier) regression across the 12
months of 2025,

    y(t) = a0 + a1*cos(2*pi*t/12) + b1*sin(2*pi*t/12)
              + a2*cos(4*pi*t/12) + b2*sin(4*pi*t/12)

fit by observation-weighted least squares, where the weight comes from the
cloud/shadow/haze quality map (clear=1, thin/haze=partial, thick cloud &
shadow=0, `cloud_mask.quality_weight`). Evaluating the fitted curve at any
month gives the "expected clear-sky" value for that month -- this is what
"follow the supposed phenology of the geography" means operationally here.

Handling pixels with too little clear data of their own ("nearby cloud-free
place" idea): rather than importing a curve from a separate geographic
location (which would need its own registration/BRDF/atmospheric matching),
we borrow strength from *spectrally-and-temporally-similar pixels elsewhere
in the same scene* -- i.e. the nearest land-cover analogs, which by
construction share the same climate/phenology drivers and the same sensor,
date, and processing. Pixels are clustered by their weighted-mean spectral
signature (KMeans), a cluster-level harmonic curve is fit from the pooled
observations of *all* member pixels (so a cluster's curve stays well
constrained even where individual pixels are cloudy), and each pixel's own
fit is shrunk toward its cluster's curve in proportion to how little
confident data that pixel has (Bayesian-ish ridge shrinkage). A pixel that's
cloudy in 10 of 12 months still gets a plausible, land-cover-appropriate
seasonal estimate instead of a degenerate fit to 2 noisy points.
"""
import numpy as np

N_BANDS = 4
HARMONIC_ORDER = 2
N_COEF = 1 + 2 * HARMONIC_ORDER  # intercept + (cos,sin) per harmonic = 5


def design_matrix(n_months: int = 12, order: int = HARMONIC_ORDER) -> np.ndarray:
    t = np.arange(n_months, dtype=np.float64)
    cols = [np.ones(n_months)]
    for h in range(1, order + 1):
        cols.append(np.cos(2 * np.pi * h * t / 12))
        cols.append(np.sin(2 * np.pi * h * t / 12))
    return np.stack(cols, axis=1)  # (T, k)


def _batched_normal_equations(X: np.ndarray, w: np.ndarray, y: np.ndarray):
    """X:(T,k) w:(T,N) y:(T,N) -> A:(N,k,k), b:(N,k)  [A^T W X and X^T W y]"""
    T, k = X.shape
    XXt = np.einsum("ti,tj->tij", X, X)          # (T,k,k), tiny
    A = np.einsum("tn,tij->nij", w, XXt)          # (N,k,k)
    wy = w * y                                    # (T,N)
    b = (X.T @ wy).T                              # (N,k)
    return A, b


def fit_cluster_priors(refl, weight, n_clusters=10, min_weight_for_feature=0.5, random_state=0):
    """refl:(T,4,H,W) weight:(T,H,W) -> (cluster_labels (H,W) int32, cluster_beta (K,4,k))

    Clusters pixels by their weighted-mean reflectance signature (a cheap,
    unsupervised stand-in for land-cover type), then fits one harmonic curve
    per cluster per band from *all* member-pixel observations pooled
    together.
    """
    from sklearn.cluster import KMeans

    T, C, H, W = refl.shape
    N = H * W
    w_flat = weight.reshape(T, N)
    refl_flat = refl.reshape(T, C, N)

    # Feature = confidence-weighted mean reflectance per band (fallback to
    # plain mean over whatever's available if nothing is confidently clear).
    conf = np.clip(w_flat, 0, None)
    wsum = conf.sum(axis=0)  # (N,)
    safe_wsum = np.where(wsum > 1e-3, wsum, 1.0)
    feat = np.einsum("tn,tcn->cn", conf, refl_flat) / safe_wsum  # (C,N)
    feat = feat.T  # (N,C)

    rng = np.random.default_rng(random_state)
    has_data = wsum > 1e-3
    sample_idx = np.where(has_data)[0]
    if len(sample_idx) > 200_000:
        sample_idx = rng.choice(sample_idx, size=200_000, replace=False)

    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=random_state)
    km.fit(feat[sample_idx])
    labels_flat = km.predict(feat).astype(np.int32)  # every pixel gets a cluster, even low-data ones
    cluster_labels = labels_flat.reshape(H, W)

    X = design_matrix(T)
    cluster_beta = np.zeros((n_clusters, C, N_COEF), dtype=np.float32)
    for k in range(n_clusters):
        members = labels_flat == k
        if members.sum() == 0:
            continue
        w_k = w_flat[:, members]          # (T, Nk)
        y_k = refl_flat[:, :, members]    # (T, C, Nk)
        # pool: sum normal equations over all member pixels -> one robust fit
        XXt = np.einsum("ti,tj->tij", X, X)
        A = np.einsum("tn,tij->ij", w_k, XXt) + 1e-4 * np.eye(N_COEF)
        A_inv = np.linalg.inv(A)
        for c in range(C):
            wy = (w_k * y_k[:, c, :])          # (T,Nk)
            b = X.T @ wy.sum(axis=1)           # (k,)
            cluster_beta[k, c] = A_inv @ b

    return cluster_labels, cluster_beta


def reconstruct(refl, weight, cluster_labels, cluster_beta, target_weight=10.0, lambda_stab=1e-2):
    """Fit the per-pixel harmonic model, shrunk toward its cluster prior.

    Returns:
        recon_curve: (T,4,H,W) float32 -- smooth harmonic estimate for every
            month, usable to fill any cloudy/hazy pixel.
        confidence:  (4,H,W) float32 in [0,1] -- how much of `recon_curve`
            is driven by this pixel's own data vs. the cluster prior
            (mostly for diagnostics/visualization).
    """
    T, C, H, W = refl.shape
    N = H * W
    X = design_matrix(T)
    w_flat = weight.reshape(T, N)
    refl_flat = refl.reshape(T, C, N)
    labels_flat = cluster_labels.reshape(N)

    A, _ = _batched_normal_equations(X, w_flat, refl_flat[:, 0, :])
    A_reg = A + lambda_stab * np.eye(X.shape[1])[None]
    A_inv = np.linalg.inv(A_reg)  # (N,k,k), computed once, reused across bands

    total_w = w_flat.sum(axis=0)  # (N,)
    conf_scalar = np.clip(total_w / target_weight, 0, 1)  # (N,)

    recon_curve = np.zeros((T, C, H, W), dtype=np.float32)
    confidence = np.zeros((C, H, W), dtype=np.float32)
    prior_beta_px = cluster_beta[labels_flat]  # (N, C, k)

    for c in range(C):
        _, b = _batched_normal_equations(X, w_flat, refl_flat[:, c, :])
        beta_local = np.einsum("nij,nj->ni", A_inv, b)  # (N,k)
        beta_final = (
            conf_scalar[:, None] * beta_local
            + (1 - conf_scalar[:, None]) * prior_beta_px[:, c, :]
        )
        curve = X @ beta_final.T  # (T,N)
        recon_curve[:, c, :, :] = curve.reshape(T, H, W)
        confidence[c] = conf_scalar.reshape(H, W)

    return recon_curve, confidence