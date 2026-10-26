###############################################################################
# ██████╗  █████╗ ████████╗ ██████╗██╗  ██╗ ██████╗ ██████╗ ██████╗ ███████╗
# ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝
# ██████╔╝███████║   ██║   ██║     ███████║██║     ██║   ██║██████╔╝█████╗
# ██╔═══╝ ██╔══██║   ██║   ██║     ██╔══██║██║     ██║   ██║██╔══██╗██╔══╝
# ██║     ██║  ██║   ██║   ╚██████╗██║  ██║╚██████╗╚██████╔╝██║  ██║███████╗
# ╚═╝     ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝╚═╝  ╚═╝╚══════╝
#
#  DefectVision — Enhanced PatchCore Pipeline  (Target: F1 ≥ 0.99)
#
#  HOW TO USE IN GOOGLE COLAB:
#  ─────────────────────────────
#  1. Upload these files to Colab (or to Google Drive):
#       • train.csv
#       • test.csv
#       • sample_submission.csv
#       • clips/  folder  (all 1245 images inside)
#
#  2. Copy-paste each CELL below into a separate Colab cell, run in order.
#
#  3. At the end, download the zip with all candidate submissions.
#
###############################################################################


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 ▸ Setup & Imports
# ═══════════════════════════════════════════════════════════════════════════════

import os, gc, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models import (
    wide_resnet50_2, Wide_ResNet50_2_Weights,
    resnet50, ResNet50_Weights,
)
from PIL import Image, ImageFilter
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 ▸ Upload / Mount Data
# ═══════════════════════════════════════════════════════════════════════════════
# ── OPTION A: Upload a zip ──────────────────────────────────────────────────
# Zip your competition folder locally:
#   PowerShell:  Compress-Archive -Path train.csv, test.csv, sample_submission.csv, clips -DestinationPath data.zip
# Then upload data.zip to Colab and run this cell.

# from google.colab import files
# uploaded = files.upload()  # pick data.zip
# !unzip -qo data.zip -d /content/data

# ── OPTION B: Google Drive ──────────────────────────────────────────────────
# Put your data folder in Drive, mount, and set DATA_ROOT below.
# from google.colab import drive
# drive.mount('/content/drive')

# Auto-detect data location
if os.path.exists("clips.zip") and not os.path.exists("clips"):
    print("Unzipping clips.zip ...")
    os.system("unzip -qo clips.zip")

DATA_ROOT = "/content" if os.path.exists("/content/train.csv") else ("/content/data" if os.path.exists("/content/data/train.csv") else ".")
print(f"Using DATA_ROOT: {DATA_ROOT}")

# Load metadata
train = pd.read_csv(f"{DATA_ROOT}/train.csv")
test  = pd.read_csv(f"{DATA_ROOT}/test.csv")
sub_path = f"{DATA_ROOT}/sample_submission.csv" if os.path.exists(f"{DATA_ROOT}/sample_submission.csv") else f"{DATA_ROOT}/submission.csv"
sample_sub = pd.read_csv(sub_path) if os.path.exists(sub_path) else None

# Build image-path lookup
CLIPS = f"{DATA_ROOT}/clips" if os.path.exists(f"{DATA_ROOT}/clips") else "/content/clips"
img_map = {}
for _, row in pd.concat([train, test], ignore_index=True).iterrows():
    img_map[row["path"]] = os.path.join(CLIPS, row["path"])

categories = sorted(train.category.unique())
print(f"Train : {len(train)}  |  Test : {len(test)}  |  Categories : {categories}")
print(f"Clips found : {len([f for f in os.listdir(CLIPS) if f.endswith(('.jpg','.png'))]) if os.path.exists(CLIPS) else 0}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 ▸ Hyper-parameters  (tune these if needed)
# ═══════════════════════════════════════════════════════════════════════════════

SIZES       = [256, 320]      # multi-scale extraction (average scores across both)
D           = 384             # random projection dim  (proven @ 0.92)
K_NN        = 5               # k-nearest-neighbour  (friend used k=1, k=5 is more robust)
TOP_K       = 10              # top-K most anomalous patches → image score
BANK_MAX    = 150_000         # memory-bank cap per category
N_FOLDS     = 5               # OOF cross-validation folds
BS_EXTRACT  = 16              # batch size for feature extraction
BS_SCORE    = 2               # batch size for scoring  (reduce if OOM)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 ▸ Core Functions
# ═══════════════════════════════════════════════════════════════════════════════

# ── image loading ────────────────────────────────────────────────────────────
def load_img(fname, size=256, flip=False):
    img = Image.open(img_map[fname]).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(0.6))      # reduce JPEG artifacts
    img = img.resize((size, size), Image.BILINEAR)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


# ── backbone 1: WideResNet50_2 ──────────────────────────────────────────────
print("Loading WideResNet50_2 …")
_wrn = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device).eval()
for p in _wrn.parameters():
    p.requires_grad_(False)

_g1 = torch.Generator().manual_seed(0)
PROJ_WRN = (torch.randn(1536, D, generator=_g1) / np.sqrt(D)).to(device)

@torch.no_grad()
def _extract_wrn(batch):
    x = (batch.to(device) - MEAN) / STD
    m = _wrn
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    l1 = m.layer1(x); l2 = m.layer2(l1); l3 = m.layer3(l2)
    l2 = F.avg_pool2d(l2, 3, 1, 1)
    l3 = F.avg_pool2d(l3, 3, 1, 1)
    l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([l2, l3], 1)                           # (B,1536,H,W)
    B, C, H, W = f.shape
    patches = f.permute(0, 2, 3, 1).reshape(B, H * W, C)
    return (patches @ PROJ_WRN).half().cpu()              # (B, HW, D)


# ── backbone 2: ResNet50 v2 ─────────────────────────────────────────────────
print("Loading ResNet50 v2 …")
_r50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
for p in _r50.parameters():
    p.requires_grad_(False)

_g2 = torch.Generator().manual_seed(42)                  # different seed → different projection
PROJ_R50 = (torch.randn(1536, D, generator=_g2) / np.sqrt(D)).to(device)

@torch.no_grad()
def _extract_r50(batch):
    x = (batch.to(device) - MEAN) / STD
    m = _r50
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    l1 = m.layer1(x); l2 = m.layer2(l1); l3 = m.layer3(l2)
    l2 = F.avg_pool2d(l2, 3, 1, 1)
    l3 = F.avg_pool2d(l3, 3, 1, 1)
    l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([l2, l3], 1)
    B, C, H, W = f.shape
    patches = f.permute(0, 2, 3, 1).reshape(B, H * W, C)
    return (patches @ PROJ_R50).half().cpu()


# ── generic patch extraction ─────────────────────────────────────────────────
def get_patches(fnames, extract_fn, size=256, bs=BS_EXTRACT, flip=False):
    out = []
    for i in tqdm(range(0, len(fnames), bs), leave=False):
        b = torch.stack([load_img(f, size=size, flip=flip) for f in fnames[i:i+bs]])
        out.append(extract_fn(b))
    return torch.cat(out)


# ── memory bank ──────────────────────────────────────────────────────────────
def build_bank(patches, max_size=BANK_MAX, seed=0):
    bank = patches.reshape(-1, D)
    if len(bank) > max_size:
        idx = torch.randperm(len(bank), generator=torch.Generator().manual_seed(seed))[:max_size]
        bank = bank[idx]
    return bank.to(device).float()


# ── k-NN scoring ─────────────────────────────────────────────────────────────
@torch.no_grad()
def score_images(patches, bank, k=K_NN, top_k=TOP_K, bs=BS_SCORE):
    """
    For each patch  → distance to k nearest neighbours in bank (averaged)
    For each image  → mean of top_k most-anomalous patch distances
    """
    scores = []
    for i in range(0, len(patches), bs):
        q = patches[i:i+bs].to(device).float()
        b_size = q.shape[0]
        q_flat = q.reshape(-1, D)                         # (bs*patches, D)

        # chunk cdist to avoid OOM on large banks
        dists = torch.cdist(q_flat, bank)                 # (bs*patches, bank)

        if k > 1:
            d_patch = dists.topk(k, dim=1, largest=False).values.mean(1)
        else:
            d_patch = dists.min(1).values

        d_patch = d_patch.reshape(b_size, -1)              # (bs, n_patches)
        img_score = d_patch.topk(top_k, dim=1).values.mean(1)
        scores.append(img_score.cpu())
    return torch.cat(scores).numpy()


print("✅  All functions defined")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 ▸ Run Per-Category Pipeline  (WRN50 + R50, multi-scale, TTA)
# ═══════════════════════════════════════════════════════════════════════════════

t0 = time.time()
results = {}

for cat in categories:
    print(f"\n{'━'*55}")
    print(f"  Category: {cat}")
    print(f"{'━'*55}")

    tr_paths = train[train.category == cat].path.tolist()
    te_mask  = (test.category == cat)
    te_paths = test[te_mask].path.tolist()
    te_idx   = np.where(te_mask)[0]
    print(f"  Train: {len(tr_paths)}   Test: {len(te_paths)}")

    cat_oof  = {}   # backbone → oof scores (averaged across scales)
    cat_test = {}   # backbone → test scores (averaged across scales)

    for bname, extract_fn in [("wrn50", _extract_wrn), ("r50", _extract_r50)]:
        oof_per_scale  = []
        test_per_scale = []

        for size in SIZES:
            tag = f"  [{bname} @{size}]"
            print(f"{tag} extracting train patches …")
            P      = get_patches(tr_paths, extract_fn, size=size)
            P_flip = get_patches(tr_paths, extract_fn, size=size, flip=True)

            # ── 5-fold OOF  (score normal images out-of-fold) ────────────
            print(f"{tag} OOF scoring …")
            oof = np.zeros(len(tr_paths))
            kf  = KFold(N_FOLDS, shuffle=True, random_state=42)
            for tr_i, va_i in kf.split(tr_paths):
                bank = build_bank(torch.cat([P[tr_i], P_flip[tr_i]]))
                oof[va_i] = score_images(P[va_i], bank)
                del bank; torch.cuda.empty_cache()
            oof_per_scale.append(oof)

            # ── full bank → test scoring  (original + flip TTA) ──────────
            print(f"{tag} test scoring + TTA …")
            bank = build_bank(torch.cat([P, P_flip]))

            P_te      = get_patches(te_paths, extract_fn, size=size)
            ts_orig   = score_images(P_te, bank)

            P_te_flip = get_patches(te_paths, extract_fn, size=size, flip=True)
            ts_flip   = score_images(P_te_flip, bank)

            ts = (ts_orig + ts_flip) / 2.0                # TTA average
            test_per_scale.append(ts)

            del bank, P, P_flip, P_te, P_te_flip
            gc.collect(); torch.cuda.empty_cache()

        # average across scales
        cat_oof[bname]  = np.mean(oof_per_scale, axis=0)
        cat_test[bname] = np.mean(test_per_scale, axis=0)

        print(f"  [{bname}] OOF normal:  mean={cat_oof[bname].mean():.4f}  "
              f"std={cat_oof[bname].std():.4f}  max={cat_oof[bname].max():.4f}")

    results[cat] = {
        "te_idx":    te_idx,
        "oof_wrn":   cat_oof["wrn50"],
        "oof_r50":   cat_oof["r50"],
        "test_wrn":  cat_test["wrn50"],
        "test_r50":  cat_test["r50"],
    }

elapsed = time.time() - t0
print(f"\n✅  Pipeline finished in {elapsed/60:.1f} minutes")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 ▸ Ensemble Scores & Rank Normalisation
# ═══════════════════════════════════════════════════════════════════════════════

def rank_pct(arr):
    """Convert array to percentile ranks in [0, 1]."""
    order = arr.argsort().argsort().astype(float)
    return (order + 1) / len(order)

# dataframe with all test scores
df = test[["id", "category"]].copy()
df["score_wrn"]  = 0.0
df["score_r50"]  = 0.0

for cat, r in results.items():
    idx = r["te_idx"]
    df.loc[idx, "score_wrn"] = r["test_wrn"]
    df.loc[idx, "score_r50"] = r["test_r50"]

# per-category percentile ranks
df["rank_wrn"] = 0.0
df["rank_r50"] = 0.0
for cat in categories:
    m = df.category == cat
    df.loc[m, "rank_wrn"] = rank_pct(df.loc[m, "score_wrn"].values)
    df.loc[m, "rank_r50"] = rank_pct(df.loc[m, "score_r50"].values)

# ensemble rank  (simple average, works great for diverse backbones)
df["rank_ens"] = (df["rank_wrn"] + df["rank_r50"]) / 2.0

# re-rank the ensemble per category so we can threshold cleanly
for cat in categories:
    m = df.category == cat
    df.loc[m, "rank_ens_final"] = rank_pct(df.loc[m, "rank_ens"].values)

# raw ensemble score  (for GMM thresholding)
df["score_ens"] = (df["score_wrn"] + df["score_r50"]) / 2.0

print("Ensemble scores computed ✅")
print(df[["id", "category", "rank_wrn", "rank_r50", "rank_ens_final"]].head(10))


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 ▸ Generate ALL Candidate Submissions
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs("/content/submissions", exist_ok=True)

def make_submission(labels_series, name):
    """Build, validate, and save a submission csv."""
    sub = sample_sub[["id"]].copy()
    sub = sub.merge(
        pd.DataFrame({"id": df["id"], "label": labels_series.values}),
        on="id", how="left"
    )
    assert len(sub) == len(sample_sub), "row count mismatch"
    assert sub.label.isna().sum() == 0, "missing labels"
    sub["label"] = sub["label"].astype(int)
    path = f"/content/submissions/{name}.csv"
    sub.to_csv(path, index=False)
    n = int(sub.label.sum())
    print(f"  {name}.csv  →  {n}/{len(sub)} defective ({100*n/len(sub):.1f}%)")
    return path

print("─── Method 1: Fixed defect-rate (rank-based) ───")
for rate in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    labels = pd.Series(0, index=df.index, dtype=int)
    for cat in categories:
        m = df.category == cat
        cat_df = df[m]
        n_def  = round(len(cat_df) * rate)
        top    = cat_df.nlargest(n_def, "rank_ens_final").index
        labels.loc[top] = 1
    make_submission(labels, f"ens_rank_top{int(rate*100)}")

print("\n─── Method 2: OOF-calibrated (mean + k·σ of normal) ───")
for k_mult in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    labels = pd.Series(0, index=df.index, dtype=int)
    for cat, r in results.items():
        m = df.category == cat
        oof_ens = (r["oof_wrn"] + r["oof_r50"]) / 2.0    # normal-only OOF
        threshold = oof_ens.mean() + k_mult * oof_ens.std()
        test_ens  = df.loc[m, "score_ens"].values
        labels.loc[df[m].index] = (test_ens > threshold).astype(int)
    make_submission(labels, f"ens_oof_{k_mult:.1f}std")

print("\n─── Method 3: GMM auto-threshold (2-component per category) ───")
labels_gmm = pd.Series(0, index=df.index, dtype=int)
for cat in categories:
    m = df.category == cat
    scores = df.loc[m, "score_ens"].values.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=42, n_init=5)
    gm.fit(scores)
    means = gm.means_.flatten()
    # component with LOWER mean = normal
    normal_comp = 0 if means[0] < means[1] else 1
    proba_defective = 1.0 - gm.predict_proba(scores)[:, normal_comp]
    labels_gmm.loc[df[m].index] = (proba_defective > 0.5).astype(int)
    nd = int((proba_defective > 0.5).sum())
    print(f"  {cat}: GMM split → {nd}/{m.sum()} defective, "
          f"means=({means[0]:.4f}, {means[1]:.4f})")
make_submission(labels_gmm, "ens_gmm_auto")

# GMM with different probability thresholds
for p_thresh in [0.3, 0.4, 0.6, 0.7]:
    labels_gp = pd.Series(0, index=df.index, dtype=int)
    for cat in categories:
        m = df.category == cat
        scores = df.loc[m, "score_ens"].values.reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=42, n_init=5)
        gm.fit(scores)
        means = gm.means_.flatten()
        normal_comp = 0 if means[0] < means[1] else 1
        proba_defective = 1.0 - gm.predict_proba(scores)[:, normal_comp]
        labels_gp.loc[df[m].index] = (proba_defective > p_thresh).astype(int)
    make_submission(labels_gp, f"ens_gmm_p{int(p_thresh*100)}")

print("\n─── Method 4: WRN50-only (in case ensemble hurts a category) ───")
for rate in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    labels = pd.Series(0, index=df.index, dtype=int)
    for cat in categories:
        m = df.category == cat
        cat_df = df[m]
        n_def  = round(len(cat_df) * rate)
        top    = cat_df.nlargest(n_def, "rank_wrn").index
        labels.loc[top] = 1
    make_submission(labels, f"wrn50_rank_top{int(rate*100)}")

print("\n─── Method 5: Fine sweep around GMM-estimated defect rate ───")
# Use GMM to estimate the most likely defect rate, then sweep around it
estimated_rates = {}
for cat in categories:
    m = df.category == cat
    scores = df.loc[m, "score_ens"].values.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=42, n_init=5)
    gm.fit(scores)
    means = gm.means_.flatten()
    normal_comp = 0 if means[0] < means[1] else 1
    proba_defective = 1.0 - gm.predict_proba(scores)[:, normal_comp]
    estimated_rates[cat] = float((proba_defective > 0.5).mean())
    print(f"  {cat}: estimated defect rate = {estimated_rates[cat]:.2%}")

# Fine sweep: GMM rate ± offsets (per-category)
for offset in [-0.10, -0.05, -0.02, 0.0, +0.02, +0.05, +0.10]:
    labels = pd.Series(0, index=df.index, dtype=int)
    for cat in categories:
        m = df.category == cat
        cat_df = df[m]
        rate = np.clip(estimated_rates[cat] + offset, 0.05, 0.95)
        n_def = round(len(cat_df) * rate)
        top   = cat_df.nlargest(n_def, "rank_ens_final").index
        labels.loc[top] = 1
    sign = "p" if offset >= 0 else "m"
    make_submission(labels, f"ens_gmm_fine_{sign}{abs(int(offset*100)):02d}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 ▸ Summary & Recommendations
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  📊  SUBMISSION SUMMARY")
print("═" * 60)

files = sorted(os.listdir("/content/submissions"))
print(f"\nTotal candidates generated: {len(files)}")

print(f"""
🏆  RECOMMENDED SUBMISSION ORDER  (try these first):

  1. ens_gmm_auto.csv          — GMM auto-threshold (best guess)
  2. ens_gmm_fine_p00.csv      — Fine-tuned around GMM estimate
  3. ens_gmm_fine_m02.csv      — Slightly fewer defective
  4. ens_gmm_fine_p02.csv      — Slightly more defective
  5. ens_oof_2.5std.csv        — OOF-calibrated threshold

  If those don't hit 0.99, try:
  6. ens_rank_top65.csv        — 65% defect rate
  7. ens_rank_top70.csv        — 70% defect rate
  8. ens_rank_top60.csv        — 60% defect rate
""")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 ▸ Download All Submissions
# ═══════════════════════════════════════════════════════════════════════════════

import shutil
shutil.make_archive("/content/all_submissions", "zip", "/content/submissions")
print("📦  /content/all_submissions.zip ready!")

# Uncomment to auto-download:
# from google.colab import files
# files.download("/content/all_submissions.zip")
