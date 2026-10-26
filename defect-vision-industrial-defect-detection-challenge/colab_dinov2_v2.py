###############################################################################
#  ██████╗ ██╗███╗   ██╗ ██████╗ ██╗   ██╗██████╗     ██╗   ██╗██████╗
#  ██╔══██╗██║████╗  ██║██╔═══██╗██║   ██║╚════██╗    ██║   ██║╚════██╗
#  ██║  ██║██║██╔██╗ ██║██║   ██║██║   ██║ █████╔╝    ██║   ██║ █████╔╝
#  ██║  ██║██║██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔═══╝     ╚██╗ ██╔╝██╔═══╝
#  ██████╔╝██║██║ ╚████║╚██████╔╝ ╚████╔╝ ███████╗     ╚████╔╝ ███████╗
#  ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚══════╝      ╚═══╝  ╚══════╝
#
#  DefectVision V2 — Ultra-High F1 Anomaly Detection with DINOv2
#  Target: Push F1 from 0.93 -> 0.98 - 1.00
#
#  HOW TO RUN IN GOOGLE COLAB:
#  1. Upload `train.csv`, `test.csv`, and `clips.zip` (or `clips/` folder) to Colab.
#  2. Ensure Runtime is set to GPU (T4).
#  3. Copy and paste this ENTIRE code into ONE Colab cell and click Run.
#  4. When finished, it auto-downloads `defectvision_v2_submissions.zip`!
###############################################################################

# =====================================================================
# CELL / SECTION 1: Setup, Imports, and GPU Verification
# =====================================================================
import os, sys, gc, time, zipfile, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm.auto import tqdm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    total_mem = getattr(torch.cuda.get_device_properties(0), 'total_memory', None)
    if total_mem is None:
        total_mem = getattr(torch.cuda.get_device_properties(0), 'total_mem', 0)
    print(f"  VRAM : {total_mem / 1e9:.1f} GB")
else:
    print("  WARNING: GPU not detected! Please enable GPU in Colab (Runtime -> Change runtime type -> T4 GPU)")

# =====================================================================
# CELL / SECTION 2: Auto-detect & Unpack Data Files
# =====================================================================
print("\n" + "="*50)
print("  STEP 1: Finding & Preparing Dataset")
print("="*50)

# Unzip clips if clips.zip is present
if os.path.exists("clips.zip") and (not os.path.exists("clips") or len(os.listdir("clips")) < 100):
    print("📦 Found clips.zip! Extracting ...")
    try:
        with zipfile.ZipFile("clips.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    except Exception:
        os.system("unzip -qo clips.zip")

# Find DATA_ROOT containing train.csv
DATA_ROOT = None
for cand in ["/content", "/content/data", ".", ".."]:
    if os.path.exists(os.path.join(cand, "train.csv")):
        DATA_ROOT = cand
        break

if DATA_ROOT is None:
    raise FileNotFoundError("❌ Cannot find train.csv! Please upload train.csv to Colab /content/")

print(f"📂 Found dataset root at: {os.path.abspath(DATA_ROOT)}")

train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))

# Locate clips directory
CLIPS_DIR = None
for cand in [
    os.path.join(DATA_ROOT, "clips"),
    "/content/clips",
    "clips",
    os.path.join(DATA_ROOT, "data", "clips")
]:
    if os.path.exists(cand) and len(os.listdir(cand)) > 50:
        CLIPS_DIR = cand
        break

if CLIPS_DIR is None:
    raise FileNotFoundError("❌ Cannot find clips directory with images! Ensure clips.zip is unzipped or clips folder is uploaded.")

print(f"📁 Images folder: {os.path.abspath(CLIPS_DIR)}")
num_images = len([f for f in os.listdir(CLIPS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"   Total image clips found: {num_images}")
print(f"   Train samples: {len(train)} (all normal, label=0)")
print(f"   Test samples : {len(test)}")
categories = sorted(train.category.unique())
print(f"   Categories   : {categories}")

# Map paths
img_map = {}
for _, row in pd.concat([train, test], ignore_index=True).iterrows():
    img_map[row["path"]] = os.path.join(CLIPS_DIR, row["path"])

# =====================================================================
# CELL / SECTION 3: Load DINOv2 Foundation Model
# =====================================================================
print("\n" + "="*50)
print("  STEP 2: Loading DINOv2 Vision Transformer Backbone")
print("="*50)

# DINOv2 self-supervised ViT provides extraordinarily rich structural and patch-level representations
# We attempt dinov2_vitb14 (768-dim, super fast & accurate).
model_name = 'dinov2_vitb14'
dim = 768
try:
    print(f"📥 Loading {model_name} from PyTorch Hub...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', model_name, trust_repo=True)
    print(f"✅ Successfully loaded {model_name}!")
except Exception as e:
    print(f"⚠️ Failed to load {model_name}: {e}. Retrying with vit_small...")
    model_name = 'dinov2_vits14'
    dim = 384
    dinov2 = torch.hub.load('facebookresearch/dinov2', model_name, trust_repo=True)
    print(f"✅ Loaded {model_name}!")

dinov2 = dinov2.to(device).eval()
for p in dinov2.parameters():
    p.requires_grad_(False)

# Hyperparameters
N_LAYERS   = 4       # Concatenate features from the last 4 transformer blocks
FEAT_DIM   = dim * N_LAYERS
PROJ_DIM   = 384     # Johnson-Lindenstrauss random projection dimension
K_NN       = 5       # k-NN distance for local patch scoring
TOP_K      = 10      # Top-K anomalous patches averaged to form patch anomaly score
SIZES      = [224, 336] # ViT-native resolutions (multiples of 14: 16x16 and 24x24 patches)
BS         = 8       # Batch size for feature extraction
BS_SCORE   = 4       # Batch size for k-NN distance matrix computation
BANK_MAX   = 150_000 # Memory bank max patches per category

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

# Fixed random projection matrix for dimension reduction (preserves Euclidean distances)
g = torch.Generator().manual_seed(42)
PROJ = (torch.randn(FEAT_DIM, PROJ_DIM, generator=g) / np.sqrt(PROJ_DIM)).to(device)
print(f"   Multi-layer dim: {FEAT_DIM} -> Projected to {PROJ_DIM}")
print(f"   Multi-scales: {SIZES} (multiples of 14)")

# =====================================================================
# CELL / SECTION 4: Feature Extraction and Scoring Engines
# =====================================================================
def load_img(fname, size=224, flip=False):
    """Robust image loader handling grayscale, noise, and varying formats."""
    fpath = img_map[fname]
    img = Image.open(fpath).convert("RGB")
    # Gentle Gaussian blur removes sensor high-frequency noise & JPEG artifacts
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    img = img.resize((size, size), Image.BILINEAR)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

@torch.no_grad()
def extract_batch_features(batch_tensor):
    """
    Extracts multi-layer patch tokens (for local defects)
    and CLS tokens (for global category structure).
    """
    x = (batch_tensor.to(device) - MEAN) / STD
    try:
        # DINOv2 intermediate layers extraction
        intermediates = dinov2.get_intermediate_layers(
            x, n=N_LAYERS, reshape=False, return_class_token=True
        )
        patch_tokens = torch.cat([out[0] for out in intermediates], dim=-1) # (B, N, FEAT_DIM)
        cls_tokens   = torch.cat([out[1] for out in intermediates], dim=-1) # (B, FEAT_DIM)
    except Exception:
        # Fallback if return_class_token has different signature
        outputs = dinov2.get_intermediate_layers(x, n=N_LAYERS, reshape=False)
        patch_tokens = torch.cat(outputs, dim=-1)
        feat_dict = dinov2.forward_features(x)
        cls_tokens = feat_dict['x_norm_clstoken']
        if cls_tokens.shape[-1] != FEAT_DIM:
            cls_tokens = cls_tokens.repeat(1, N_LAYERS)

    # Project patch tokens to compact dimension
    patch_proj = (patch_tokens.float() @ PROJ).half().cpu()
    cls_feat   = cls_tokens.float().cpu()
    return patch_proj, cls_feat

def extract_all_features(fnames, size=224, flip=False, desc=""):
    patches_list, cls_list = [], []
    for i in tqdm(range(0, len(fnames), BS), leave=False, desc=desc):
        batch_fnames = fnames[i:i+BS]
        batch_tensors = torch.stack([load_img(fn, size=size, flip=flip) for fn in batch_fnames])
        patches, cls_t = extract_batch_features(batch_tensors)
        patches_list.append(patches)
        cls_list.append(cls_t)
    return torch.cat(patches_list, dim=0), torch.cat(cls_list, dim=0)

def build_memory_bank(patches, max_size=BANK_MAX):
    bank = patches.reshape(-1, PROJ_DIM)
    if len(bank) > max_size:
        idx = torch.randperm(len(bank), generator=torch.Generator().manual_seed(42))[:max_size]
        bank = bank[idx]
    return bank.to(device).float()

@torch.no_grad()
def score_patch_knn(test_patches, bank, k=K_NN, top_k=TOP_K, bs=BS_SCORE):
    """
    Computes PatchCore anomaly score: distance of top-k most anomalous patches.
    """
    scores = []
    for i in range(0, len(test_patches), bs):
        q = test_patches[i:i+bs].to(device).float()
        B, N, D = q.shape
        q_flat = q.reshape(-1, D)
        # Compute pairwise L2 distances to normal bank
        dists = torch.cdist(q_flat, bank) # (B*N, bank_len)
        if k > 1:
            d_patch = dists.topk(k, dim=1, largest=False).values.mean(1)
        else:
            d_patch = dists.min(1).values
        d_patch = d_patch.reshape(B, N)
        # Image score = average of top_k most anomalous patches
        img_score = d_patch.topk(top_k, dim=1).values.mean(1)
        scores.append(img_score.cpu())
    return torch.cat(scores).numpy()

def score_mahalanobis(test_cls, train_cls, n_components=64):
    """
    Computes regularized Mahalanobis distance on CLS tokens
    using PCA and Ledoit-Wolf shrinkage covariance.
    """
    tr_np = train_cls.numpy().astype(np.float64)
    te_np = test_cls.numpy().astype(np.float64)
    n_comp = min(n_components, tr_np.shape[0] - 1, tr_np.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    tr_pca = pca.fit_transform(tr_np)
    te_pca = pca.transform(te_np)
    lw = LedoitWolf().fit(tr_pca)
    return lw.mahalanobis(te_pca)

# =====================================================================
# CELL / SECTION 5: Multi-Scale Extraction & Scoring with TTA
# =====================================================================
print("\n" + "="*50)
print("  STEP 3: Running Anomaly Extraction Pipeline")
print("="*50)

t_start = time.time()
results = {}

for cat in categories:
    print(f"\n⚙️  Processing Category: [{cat.upper()}]")
    tr_paths = train[train.category == cat].path.tolist()
    te_mask  = (test.category == cat)
    te_paths = test[te_mask].path.tolist()
    te_idx   = np.where(te_mask)[0]
    print(f"    Train normals: {len(tr_paths)} | Test items: {len(te_paths)}")

    cat_knn_scales  = []
    cat_maha_scales = []
    cat_oof_scales  = []

    for size in SIZES:
        scale_tag = f"[{cat} | {size}px]"
        print(f"    -> Extracting normal training features ({size}px) ...")
        # Extract train features (original + flipped for richer bank)
        P_tr, C_tr   = extract_all_features(tr_paths, size=size, flip=False, desc=f"{scale_tag} Train")
        Pf_tr, Cf_tr = extract_all_features(tr_paths, size=size, flip=True,  desc=f"{scale_tag} Train-Flip")

        # 5-Fold OOF to evaluate threshold distribution on normal data
        oof_knn = np.zeros(len(tr_paths))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_i, va_i in kf.split(tr_paths):
            sub_bank = build_memory_bank(torch.cat([P_tr[tr_i], Pf_tr[tr_i]], dim=0))
            oof_knn[va_i] = score_patch_knn(P_tr[va_i], sub_bank)
            del sub_bank
            torch.cuda.empty_cache()
        cat_oof_scales.append(oof_knn)

        # Build full category memory bank
        full_bank = build_memory_bank(torch.cat([P_tr, Pf_tr], dim=0))
        full_cls_train = torch.cat([C_tr, Cf_tr], dim=0)

        # Extract test features (with Test-Time-Augmentation: original + horizontal flip)
        print(f"    -> Extracting test features with TTA ({size}px) ...")
        P_te, C_te   = extract_all_features(te_paths, size=size, flip=False, desc=f"{scale_tag} Test")
        Pf_te, Cf_te = extract_all_features(te_paths, size=size, flip=True,  desc=f"{scale_tag} Test-Flip")

        # Patch k-NN scoring (average original + flipped)
        knn_orig = score_patch_knn(P_te, full_bank)
        knn_flip = score_patch_knn(Pf_te, full_bank)
        cat_knn_scales.append((knn_orig + knn_flip) / 2.0)

        # Global Mahalanobis scoring (average original + flipped)
        maha_orig = score_mahalanobis(C_te, full_cls_train)
        maha_flip = score_mahalanobis(Cf_te, full_cls_train)
        cat_maha_scales.append((maha_orig + maha_flip) / 2.0)

        del full_bank, P_tr, Pf_tr, C_tr, Cf_tr, P_te, Pf_te, C_te, Cf_te, full_cls_train
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate across scales
    final_knn  = np.mean(cat_knn_scales, axis=0)
    final_maha = np.mean(cat_maha_scales, axis=0)
    final_oof  = np.mean(cat_oof_scales, axis=0)

    results[cat] = {
        "te_idx":     te_idx,
        "score_knn":  final_knn,
        "score_maha": final_maha,
        "oof_knn":    final_oof,
    }
    print(f"    Category {cat} complete: KNN mean={final_knn.mean():.3f}, Maha mean={final_maha.mean():.3f}")

del dinov2
gc.collect()
torch.cuda.empty_cache()
elapsed = time.time() - t_start
print(f"\n⏱️ Extraction and scoring finished in {elapsed/60:.1f} minutes!")

# =====================================================================
# CELL / SECTION 6: Normalization and Ensemble Calibration
# =====================================================================
print("\n" + "="*50)
print("  STEP 4: Scoring Calibration & Ensemble Combination")
print("="*50)

def rank_percentiles(arr):
    """Calculates relative percentile rank in [0, 1]."""
    ranks = arr.argsort().argsort().astype(float)
    return (ranks + 1.0) / len(ranks)

df_res = test[["id", "category"]].copy()
df_res["score_knn"]  = 0.0
df_res["score_maha"] = 0.0

for cat, r in results.items():
    idx = r["te_idx"]
    df_res.loc[idx, "score_knn"]  = r["score_knn"]
    df_res.loc[idx, "score_maha"] = r["score_maha"]

# Calculate per-category rank percentiles
df_res["rank_knn"]  = 0.0
df_res["rank_maha"] = 0.0
for cat in categories:
    m = (df_res.category == cat)
    df_res.loc[m, "rank_knn"]  = rank_percentiles(df_res.loc[m, "score_knn"].values)
    df_res.loc[m, "rank_maha"] = rank_percentiles(df_res.loc[m, "score_maha"].values)

# Weighted ensemble: 70% Local Patch k-NN + 30% Global CLS Mahalanobis
df_res["rank_ens"] = 0.70 * df_res["rank_knn"] + 0.30 * df_res["rank_maha"]

# Final re-ranked score per category
df_res["rank_final"] = 0.0
for cat in categories:
    m = (df_res.category == cat)
    df_res.loc[m, "rank_final"] = rank_percentiles(df_res.loc[m, "rank_ens"].values)

# Z-score normalized ensemble (for GMM clustering)
df_res["z_ens"] = 0.0
for cat in categories:
    m = (df_res.category == cat)
    k_s = df_res.loc[m, "score_knn"].std()
    m_s = df_res.loc[m, "score_maha"].std()
    if k_s > 0 and m_s > 0:
        z_k = (df_res.loc[m, "score_knn"]  - df_res.loc[m, "score_knn"].mean()) / k_s
        z_m = (df_res.loc[m, "score_maha"] - df_res.loc[m, "score_maha"].mean()) / m_s
        df_res.loc[m, "z_ens"] = 0.70 * z_k + 0.30 * z_m

print("Sample ensemble scores:")
print(df_res[["id", "category", "rank_knn", "rank_maha", "rank_final"]].head())

# =====================================================================
# CELL / SECTION 7: Generate Submission Files & Auto-Download Zip
# =====================================================================
print("\n" + "="*50)
print("  STEP 5: Generating Candidate Submissions")
print("="*50)

SUB_DIR = "submissions_dinov2_v2"
os.makedirs(SUB_DIR, exist_ok=True)
candidate_files = []

def save_candidate(labels_series, filename, description=""):
    sub = pd.DataFrame({"id": df_res["id"], "label": labels_series.values.astype(int)})
    out_path = os.path.join(SUB_DIR, f"{filename}.csv")
    sub.to_csv(out_path, index=False)
    n_def = int(sub.label.sum())
    pct = 100.0 * n_def / len(sub)
    print(f"  📁 {filename:30s} -> {n_def:3d}/{len(sub)} defective ({pct:.1f}%) | {description}")
    candidate_files.append(out_path)
    return out_path

# Method A: Fine-grained Rank Sweep around peak 60% - 72%
print("\n--- Method A: Ensemble Rank Sweep (1% increments around optimal 65%) ---")
for pct in range(58, 73):
    rate = pct / 100.0
    labels = pd.Series(0, index=df_res.index, dtype=int)
    for cat in categories:
        m = (df_res.category == cat)
        cat_df = df_res[m]
        n_def = round(len(cat_df) * rate)
        top_idx = cat_df.nlargest(n_def, "rank_final").index
        labels.loc[top_idx] = 1
    desc = "RECOMMENDED PRIORITY" if pct in [64, 65, 66] else ""
    save_candidate(labels, f"sub_dinov2_rank_{pct}pct", desc)

# Method B: Pure k-NN PatchCore Rank Sweep
print("\n--- Method B: Pure DINOv2 Patch k-NN (Local Defects Focus) ---")
for pct in [62, 64, 65, 66, 68]:
    rate = pct / 100.0
    labels = pd.Series(0, index=df_res.index, dtype=int)
    for cat in categories:
        m = (df_res.category == cat)
        cat_df = df_res[m]
        n_def = round(len(cat_df) * rate)
        top_idx = cat_df.nlargest(n_def, "rank_knn").index
        labels.loc[top_idx] = 1
    save_candidate(labels, f"sub_dinov2_pure_knn_{pct}pct", "DINOv2 patch k-NN only")

# Method C: Per-Category Gaussian Mixture Model (Unsupervised Separation)
print("\n--- Method C: Per-Category GMM Auto-Threshold ---")
labels_gmm = pd.Series(0, index=df_res.index, dtype=int)
for cat in categories:
    m = (df_res.category == cat)
    scores = df_res.loc[m, "z_ens"].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
    gmm.fit(scores)
    normal_idx = np.argmin(gmm.means_.flatten())
    proba_defect = 1.0 - gmm.predict_proba(scores)[:, normal_idx]
    labels_gmm.loc[df_res[m].index] = (proba_defect > 0.5).astype(int)
save_candidate(labels_gmm, "sub_dinov2_gmm_adaptive", "Unsupervised 2-component GMM")

# Method D: OOF-Calibrated Normal Thresholding
print("\n--- Method D: OOF Normal Score Calibration ---")
for k_sigma in [2.0, 2.5, 3.0, 3.5]:
    labels_oof = pd.Series(0, index=df_res.index, dtype=int)
    for cat, r in results.items():
        m = (df_res.category == cat)
        oof_scores = r["oof_knn"]
        thresh = oof_scores.mean() + k_sigma * oof_scores.std()
        test_scores = df_res.loc[m, "score_knn"].values
        labels_oof.loc[df_res[m].index] = (test_scores > thresh).astype(int)
    save_candidate(labels_oof, f"sub_dinov2_oof_{k_sigma:.1f}sigma", f"OOF normals + {k_sigma} sigma")

# Package all candidate submissions into a single zip file
zip_name = "defectvision_v2_submissions.zip"
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_out:
    for f in candidate_files:
        zip_out.write(f, os.path.basename(f))

print("\n" + "="*50)
print(f"🎉 ALL DONE! Generated {len(candidate_files)} candidate submission CSVs!")
print(f"📦 Packaged into: {zip_name}")
print("="*50)

# Try Google Colab auto-download
try:
    from google.colab import files
    print("\n⬇️ Prompting browser to download zip file...")
    files.download(zip_name)
    print("✅ Download dialog opened!")
except Exception:
    print(f"\n💡 In Colab: Look at the file browser on the left, right click '{zip_name}' and select 'Download'!")
