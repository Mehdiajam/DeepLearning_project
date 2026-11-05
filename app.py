import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm

from models import BaselineUNet, build_transfer_unet, ConvAutoencoder
from losses import compute_metrics

# =========================================================================
# Page Configuration & Custom Styling
# =========================================================================
st.set_page_config(
    page_title="RoadAnomaly | Deep Learning Validation WebApp",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        color: #f8fafc !important;
        margin: 0;
        font-size: 1.9rem;
        font-weight: 800;
    }
    .main-header p {
        color: #cbd5e1 !important;
        margin: 0.35rem 0 0 0;
        font-size: 1rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .safety-banner-crit {
        background-color: #fef2f2;
        border: 1px solid #f87171;
        border-left: 6px solid #dc2626;
        color: #991b1b;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .safety-banner-warn {
        background-color: #fffbeb;
        border: 1px solid #fbbf24;
        border-left: 6px solid #d97706;
        color: #92400e;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .safety-banner-clear {
        background-color: #f0fdf4;
        border: 1px solid #4ade80;
        border-left: 6px solid #16a34a;
        color: #166534;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .legend-box {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 3px;
        margin-right: 6px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(ROOT_DIR, "frames")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
EDA_DIR = os.path.join(ROOT_DIR, "eda_outputs")
EVAL_DIR = os.path.join(ROOT_DIR, "evaluation_results")
IMG_SIZE = (384, 512)


# =========================================================================
# Caching: Models, Metadata, and Precomputed Split Predictions
# =========================================================================
@st.cache_resource(show_spinner="Loading trained PyTorch models & checkpoints...")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_net = BaselineUNet().to(device)
    transfer_net = build_transfer_unet(encoder_name="resnet34", encoder_weights=None).to(device)
    autoencoder = ConvAutoencoder().to(device)

    b_ckpt = os.path.join(CKPT_DIR, "baseline_unet_best.pth")
    tl_ckpt = os.path.join(CKPT_DIR, "transfer_resnet34_unet_best.pth")
    cae_ckpt = os.path.join(CKPT_DIR, "convautoencoder_best.pth")

    if os.path.exists(b_ckpt):
        baseline_net.load_state_dict(torch.load(b_ckpt, map_location=device, weights_only=True))
    if os.path.exists(tl_ckpt):
        transfer_net.load_state_dict(torch.load(tl_ckpt, map_location=device, weights_only=True))
    if os.path.exists(cae_ckpt):
        autoencoder.load_state_dict(torch.load(cae_ckpt, map_location=device, weights_only=True))

    baseline_net.eval()
    transfer_net.eval()
    autoencoder.eval()

    return baseline_net, transfer_net, autoencoder, device


@st.cache_data(show_spinner="Loading RoadAnomaly dataset metadata & splits...")
def load_dataset_metadata():
    splits_path = os.path.join(ROOT_DIR, "dataset_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    split_lookup = {}
    for s_name in ["train", "val", "test"]:
        for fname in splits[s_name]:
            split_lookup[fname] = s_name.upper()

    csv_path = os.path.join(EDA_DIR, "dataset_summary.csv")
    df = pd.read_csv(csv_path)
    df["split"] = df["filename"].map(split_lookup)
    return df, splits


def preprocess_pil_image(pil_img, img_size=IMG_SIZE):
    img_np = np.array(pil_img.convert("RGB"))
    tf = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    resized_rgb = A.Resize(img_size[0], img_size[1])(image=img_np)["image"]
    tensor_img = tf(image=img_np)["image"].unsqueeze(0)
    return img_np, resized_rgb, tensor_img


def run_single_inference(tensor_img, baseline_net, transfer_net, autoencoder, device, hybrid_w=0.85):
    x = tensor_img.to(device)
    with torch.no_grad():
        # 1. Baseline U-Net
        b_prob = torch.sigmoid(baseline_net(x))[0, 0].cpu().numpy()

        # 2. Transfer ResNet-34 U-Net
        tl_prob_t = torch.sigmoid(transfer_net(x))
        tl_prob = tl_prob_t[0, 0].cpu().numpy()

        # 3. Transfer ResNet-34 + Horizontal Flip TTA
        x_flip = torch.flip(x, dims=[3])
        tl_flip_t = torch.flip(torch.sigmoid(transfer_net(x_flip)), dims=[3])
        tta_prob_t = 0.5 * (tl_prob_t + tl_flip_t)
        tta_prob = tta_prob_t[0, 0].cpu().numpy()

        # 4. Conv Autoencoder Reconstruction & Residual Maps
        recon_t = autoencoder(x)
        recon_np = recon_t[0].permute(1, 2, 0).cpu().numpy()
        recon_np = np.clip(recon_np, 0.0, 1.0)

        cae_raw_t = autoencoder.compute_anomaly_map(x, calibrate_unit=False)[0, 0]
        cae_norm = ((cae_raw_t - cae_raw_t.min()) / (cae_raw_t.max() - cae_raw_t.min() + 1e-7)).cpu().numpy()

        cae_unit_t = autoencoder.compute_anomaly_map(x, calibrate_unit=True)[0, 0]
        cae_unit_norm = ((cae_unit_t - cae_unit_t.min()) / (cae_unit_t.max() - cae_unit_t.min() + 1e-7)).cpu().numpy()

        # 5. Hybrid Ensemble (Supervised TTA + Unsupervised Residual)
        hybrid_prob = hybrid_w * tta_prob + (1.0 - hybrid_w) * cae_unit_norm

    return {
        "baseline": b_prob,
        "resnet34": tl_prob,
        "resnet34_tta": tta_prob,
        "cae": cae_norm,
        "cae_unit": cae_unit_norm,
        "recon_rgb": recon_np,
        "hybrid": hybrid_prob
    }


def apply_postprocessing(prob_map, threshold=0.5, morph_clean=False):
    pred_bin = (prob_map >= threshold).astype(np.float32)
    if morph_clean:
        pred_bin = binary_closing(pred_bin, structure=np.ones((3, 3))).astype(np.float32)
        pred_bin = binary_opening(pred_bin, structure=np.ones((3, 3))).astype(np.float32)
    return pred_bin


def compute_curve_stats(scores_flat, targets_flat):
    fpr, tpr, _ = roc_curve(targets_flat, scores_flat)
    roc_auc = float(auc(fpr, tpr))
    prec, rec, _ = precision_recall_curve(targets_flat, scores_flat)
    ap = float(average_precision_score(targets_flat, scores_flat))
    idx_95 = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[idx_95[0]]) if len(idx_95) > 0 else 1.0
    # Subsample curve points for fast interactive Plotly rendering
    step_roc = max(1, len(fpr) // 350)
    step_pr = max(1, len(prec) // 350)
    return {
        "fpr": fpr[::step_roc],
        "tpr": tpr[::step_roc],
        "auroc": roc_auc,
        "prec": prec[::step_pr],
        "rec": rec[::step_pr],
        "auprc": ap,
        "fpr95": fpr95
    }


@st.cache_data(show_spinner="Evaluating validation & test sets across all model paradigms...")
def precompute_validation_and_test_suite():
    baseline_net, transfer_net, autoencoder, device = load_models()
    df, splits = load_dataset_metadata()

    def process_split(filenames):
        records = []
        for fname in filenames:
            base_name = os.path.splitext(fname)[0]
            img_path = os.path.join(FRAMES_DIR, fname)
            mask_path = os.path.join(FRAMES_DIR, f"{base_name}.labels", "labels_semantic.png")

            pil_img = Image.open(img_path).convert("RGB")
            raw_mask = np.array(Image.open(mask_path))
            bin_mask = (raw_mask == 2).astype(np.float32)
            gt_resized = A.Resize(IMG_SIZE[0], IMG_SIZE[1])(image=np.array(pil_img), mask=bin_mask)["mask"]

            _, resized_rgb, tensor_img = preprocess_pil_image(pil_img, IMG_SIZE)
            preds = run_single_inference(tensor_img, baseline_net, transfer_net, autoencoder, device)

            meta_row = df[df["filename"] == fname].iloc[0]
            records.append({
                "filename": fname,
                "category": meta_row["category"],
                "anomaly_ratio_pct": float(meta_row["anomaly_ratio_pct"]),
                "mask": gt_resized,
                "resized_rgb": resized_rgb,
                **preds
            })
        return records

    val_records = process_split(splits["val"])
    test_records = process_split(splits["test"])

    # Validation Threshold Calibration Sweep
    thresholds = np.linspace(0.05, 0.95, 37)
    model_keys = [
        ("baseline", "Baseline U-Net"),
        ("resnet34", "Transfer ResNet-34 U-Net"),
        ("resnet34_tta", "ResNet-34 + TTA"),
        ("cae", "Conv Autoencoder"),
        ("hybrid", "Hybrid Ensemble (ResNet34+CAE)")
    ]

    optimal_thresholds = {}
    calibration_curves = {k: [] for k, _ in model_keys}

    for key, _ in model_keys:
        best_t, best_d = 0.50, -1.0
        for t in thresholds:
            dices = [
                compute_metrics(torch.from_numpy(item[key]), torch.from_numpy(item["mask"]), threshold=float(t))["dice"]
                for item in val_records
            ]
            mean_d = float(np.mean(dices))
            calibration_curves[key].append(mean_d)
            if mean_d > best_d:
                best_d = mean_d
                best_t = float(t)
        optimal_thresholds[key] = (best_t, best_d)

    # Pixel-level ROC and PR curves on Unseen Test Set
    all_targets_flat = np.concatenate([item["mask"].reshape(-1) for item in test_records])
    curve_results = {}
    for key, _ in model_keys:
        scores_flat = np.concatenate([item[key].reshape(-1) for item in test_records])
        curve_results[key] = compute_curve_stats(scores_flat, all_targets_flat)

    # Evaluate all 6 variants on Unseen Test Set
    def eval_variant(key, threshold, morph_clean=False, percentile_90=False):
        m_list = []
        for item in test_records:
            prob = item[key]
            if percentile_90:
                t_val = float(np.percentile(prob, 90))
                pred_bin = (prob >= t_val).astype(np.float32)
            else:
                pred_bin = apply_postprocessing(prob, threshold=threshold, morph_clean=morph_clean)
            m = compute_metrics(torch.from_numpy(pred_bin), torch.from_numpy(item["mask"]), threshold=0.5)
            m_list.append(m)
        return {k: float(np.mean([m[k] for m in m_list])) for k in m_list[0].keys()}

    tl_cal_t = optimal_thresholds["resnet34"][0]
    tta_cal_t = optimal_thresholds["resnet34_tta"][0]
    hyb_cal_t = optimal_thresholds["hybrid"][0]

    b_m = eval_variant("baseline", 0.50)
    cae_m = eval_variant("cae", 0.50, percentile_90=True)
    tl_m = eval_variant("resnet34", 0.50)
    tl_cal_m = eval_variant("resnet34", tl_cal_t)
    tta_m = eval_variant("resnet34_tta", tta_cal_t, morph_clean=True)
    hyb_m = eval_variant("hybrid", hyb_cal_t, morph_clean=True)

    benchmark_rows = [
        {
            "Variant": "1. Baseline U-Net (Scratch, τ=0.50)",
            "ShortName": "Baseline U-Net",
            "Paradigm": "Supervised (From Scratch)",
            "Threshold": 0.50,
            "Test IoU": round(b_m["iou"], 4),
            "Test Dice (F1)": round(b_m["dice"], 4),
            "Precision": round(b_m["precision"], 4),
            "Recall": round(b_m["recall"], 4),
            "Pixel Acc (%)": round(b_m["accuracy"] * 100, 2),
            "AUROC": round(curve_results["baseline"]["auroc"], 4),
            "AUPRC (AP)": round(curve_results["baseline"]["auprc"], 4),
            "FPR@95%TPR": round(curve_results["baseline"]["fpr95"], 4)
        },
        {
            "Variant": "2. Conv Autoencoder (90th Pct Residual)",
            "ShortName": "Conv Autoencoder",
            "Paradigm": "Unsupervised (L1 Recon)",
            "Threshold": 0.90,
            "Test IoU": round(cae_m["iou"], 4),
            "Test Dice (F1)": round(cae_m["dice"], 4),
            "Precision": round(cae_m["precision"], 4),
            "Recall": round(cae_m["recall"], 4),
            "Pixel Acc (%)": round(cae_m["accuracy"] * 100, 2),
            "AUROC": round(curve_results["cae"]["auroc"], 4),
            "AUPRC (AP)": round(curve_results["cae"]["auprc"], 4),
            "FPR@95%TPR": round(curve_results["cae"]["fpr95"], 4)
        },
        {
            "Variant": "3. Transfer ResNet-34 U-Net (Default τ=0.50)",
            "ShortName": "ResNet-34 (τ=0.50)",
            "Paradigm": "Supervised (ImageNet Pretrained)",
            "Threshold": 0.50,
            "Test IoU": round(tl_m["iou"], 4),
            "Test Dice (F1)": round(tl_m["dice"], 4),
            "Precision": round(tl_m["precision"], 4),
            "Recall": round(tl_m["recall"], 4),
            "Pixel Acc (%)": round(tl_m["accuracy"] * 100, 2),
            "AUROC": round(curve_results["resnet34"]["auroc"], 4),
            "AUPRC (AP)": round(curve_results["resnet34"]["auprc"], 4),
            "FPR@95%TPR": round(curve_results["resnet34"]["fpr95"], 4)
        },
        {
            "Variant": f"4. Transfer ResNet-34 U-Net (Val-Calibrated τ*={tl_cal_t:.2f})",
            "ShortName": f"ResNet-34 (τ*={tl_cal_t:.2f})",
            "Paradigm": "Supervised + Val Calibration",
            "Threshold": round(tl_cal_t, 2),
            "Test IoU": round(tl_cal_m["iou"], 4),
            "Test Dice (F1)": round(tl_cal_m["dice"], 4),
            "Precision": round(tl_cal_m["precision"], 4),
            "Recall": round(tl_cal_m["recall"], 4),
            "Pixel Acc (%)": round(tl_cal_m["accuracy"] * 100, 2),
            "AUROC": round(curve_results["resnet34"]["auroc"], 4),
            "AUPRC (AP)": round(curve_results["resnet34"]["auprc"], 4),
            "FPR@95%TPR": round(curve_results["resnet34"]["fpr95"], 4)
        },
        {
            "Variant": f"5. ResNet-34 + TTA + Morph Clean (τ*={tta_cal_t:.2f})",
            "ShortName": "ResNet-34 + TTA + Morph",
            "Paradigm": "Supervised + TTA + PostProc",
            "Threshold": round(tta_cal_t, 2),
            "Test IoU": round(tta_m["iou"], 4),
            "Test Dice (F1)": round(tta_m["dice"], 4),
            "Precision": round(tta_m["precision"], 4),
            "Recall": round(tta_m["recall"], 4),
            "Pixel Acc (%)": round(tta_m["accuracy"] * 100, 2),
            "AUROC": round(curve_results["resnet34_tta"]["auroc"], 4),
            "AUPRC (AP)": round(curve_results["resnet34_tta"]["auprc"], 4),
            "FPR@95%TPR": round(curve_results["resnet34_tta"]["fpr95"], 4)
        },
        {
            "Variant": f"6. Hybrid Ensemble (ResNet-34 TTA + CAE, τ*={hyb_cal_t:.2f})",
            "ShortName": "Hybrid Ensemble",
            "Paradigm": "Hybrid (Supervised + Unsup)",
            "Threshold": round(hyb_cal_t, 2),
            "Test IoU": round(hyb_m["iou"], 4),
            "Test Dice (F1)": round(hyb_m["dice"], 4),
            "Precision": round(hyb_m["precision"], 4),
            "Recall": round(hyb_m["recall"], 4),
            "Pixel Acc (%)": round(hyb_m["accuracy"] * 100, 2),
            "AUROC": round(curve_results["hybrid"]["auroc"], 4),
            "AUPRC (AP)": round(curve_results["hybrid"]["auprc"], 4),
            "FPR@95%TPR": round(curve_results["hybrid"]["fpr95"], 4)
        }
    ]
    benchmark_df = pd.DataFrame(benchmark_rows)

    # Per-image test breakdown
    per_image_rows = []
    for item in test_records:
        m_base = compute_metrics(torch.from_numpy(item["baseline"]), torch.from_numpy(item["mask"]), threshold=0.5)
        m_tl = compute_metrics(torch.from_numpy(item["resnet34"]), torch.from_numpy(item["mask"]), threshold=0.5)
        pred_tta = apply_postprocessing(item["resnet34_tta"], threshold=tta_cal_t, morph_clean=True)
        m_tta = compute_metrics(torch.from_numpy(pred_tta), torch.from_numpy(item["mask"]), threshold=0.5)
        pred_hyb = apply_postprocessing(item["hybrid"], threshold=hyb_cal_t, morph_clean=True)
        m_hyb = compute_metrics(torch.from_numpy(pred_hyb), torch.from_numpy(item["mask"]), threshold=0.5)

        norm_cae = item["cae"]
        pred_cae = (norm_cae >= np.percentile(norm_cae, 90)).astype(np.float32)
        m_cae = compute_metrics(torch.from_numpy(pred_cae), torch.from_numpy(item["mask"]), threshold=0.5)

        per_image_rows.append({
            "Filename": item["filename"],
            "Category": item["category"],
            "Anomaly Area (%)": item["anomaly_ratio_pct"],
            "Baseline IoU": round(m_base["iou"], 4),
            "Baseline Dice": round(m_base["dice"], 4),
            "Autoencoder Dice": round(m_cae["dice"], 4),
            "ResNet34 IoU": round(m_tl["iou"], 4),
            "ResNet34 Dice": round(m_tl["dice"], 4),
            "ResNet34+TTA IoU": round(m_tta["iou"], 4),
            "ResNet34+TTA Dice": round(m_tta["dice"], 4),
            "Hybrid Dice": round(m_hyb["dice"], 4),
            "Precision": round(m_tta["precision"], 4),
            "Recall": round(m_tta["recall"], 4)
        })
    per_image_df = pd.DataFrame(per_image_rows).sort_values(by="ResNet34+TTA Dice", ascending=False)

    return {
        "val_records": val_records,
        "test_records": test_records,
        "thresholds": thresholds,
        "optimal_thresholds": optimal_thresholds,
        "calibration_curves": calibration_curves,
        "curve_results": curve_results,
        "benchmark_df": benchmark_df,
        "per_image_df": per_image_df,
        "pos_ratio": float(np.mean(all_targets_flat))
    }


# =========================================================================
# Visualization Helpers
# =========================================================================
def make_overlay_image(rgb_img, mask_bin, color=(255, 40, 40), alpha=0.50):
    overlay = rgb_img.astype(np.float32).copy()
    fg = mask_bin > 0.5
    for c in range(3):
        overlay[:, :, c] = np.where(
            fg,
            (1.0 - alpha) * overlay[:, :, c] + alpha * color[c],
            overlay[:, :, c]
        )
    # Draw crisp contour around mask
    from scipy.ndimage import binary_dilation
    contour = binary_dilation(fg, iterations=2) ^ fg
    overlay[contour] = [255, 255, 0]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_heatmap_rgb(prob_map, cmap_name="inferno"):
    colormap = cm.get_cmap(cmap_name)
    rgba = colormap(np.clip(prob_map, 0.0, 1.0))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def make_error_decomposition_rgb(rgb_img, pred_bin, gt_bin, bg_dim=0.35):
    """
    Creates an intuitive error decomposition map blended over dimmed input image:
    - Green: True Positive (Correctly detected hazard)
    - Blue/Cyan: False Negative (Missed hazard pixel - safety critical!)
    - Red: False Positive (False alarm on normal road/background)
    """
    base = (rgb_img.astype(np.float32) * bg_dim).copy()
    tp = (gt_bin == 1) & (pred_bin == 1)
    fn = (gt_bin == 1) & (pred_bin == 0)
    fp = (gt_bin == 0) & (pred_bin == 1)

    base[tp] = [16, 235, 80]    # Vibrant Green: True Positive
    base[fn] = [30, 144, 255]   # Dodger Blue: False Negative (Missed)
    base[fp] = [255, 50, 50]    # Bright Red: False Positive

    tp_cnt = int(np.sum(tp))
    fn_cnt = int(np.sum(fn))
    fp_cnt = int(np.sum(fp))
    tn_cnt = int(np.sum((gt_bin == 0) & (pred_bin == 0)))
    return np.clip(base, 0, 255).astype(np.uint8), {"TP": tp_cnt, "FN": fn_cnt, "FP": fp_cnt, "TN": tn_cnt}


# =========================================================================
# Load Core Resources
# =========================================================================
baseline_net, transfer_net, autoencoder, device = load_models()
df_meta, splits_data = load_dataset_metadata()
suite = precompute_validation_and_test_suite()

# =========================================================================
# Sidebar Navigation & Global Controls
# =========================================================================
with st.sidebar:
    st.markdown("## 🚗 RoadAnomaly Validator")
    st.caption("Deep Learning Module Assessment — Interactive Validation & Visualization WebApp")

    st.markdown("---")
    nav_page = st.radio(
        "Navigation Module",
        [
            "📊 1. Executive Dashboard & Benchmarks",
            "🔍 2. Interactive Model Inspector (60 Frames)",
            "📈 3. Deep Validation Suite (ROC / PR / Scale)",
            "🗂️ 4. Dataset Explorer & Class Imbalance",
            "⚡ 5. Live Custom Image Inference & Profiler"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Global Inference Controls")
    use_calibrated_thresh = st.toggle("Use Validation-Calibrated Threshold (τ*)", value=True, help="Automatically uses the optimal threshold τ* tuned on the Validation Set.")
    custom_threshold = st.slider(
        "Manual Decision Threshold (τ)",
        min_value=0.05,
        max_value=0.95,
        value=float(suite["optimal_thresholds"]["resnet34_tta"][0]),
        step=0.05,
        disabled=use_calibrated_thresh
    )
    enable_tta = st.toggle("Horizontal-Flip TTA", value=True, help="Enables Test-Time Augmentation by averaging original and horizontally flipped predictions.")
    enable_morph = st.toggle("Morphological Cleanup (3×3)", value=True, help="Applies 3x3 binary closing & opening to suppress isolated pixel noise.")
    overlay_alpha = st.slider("Mask Overlay Opacity", 0.15, 0.85, 0.50, 0.05)

    st.markdown("---")
    hw_label = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU Mode"
    st.info(f"**Compute Engine:** `{hw_label}`\n\n**PyTorch:** `{torch.__version__}`\n\n**Input Resolution:** `384 × 512` (from `1280 × 720` HD)")


# =========================================================================
# Header Banner
# =========================================================================
st.markdown("""
<div class="main-header">
    <h1>🚗 Pixel-Level Road Anomaly Detection for Autonomous Driving</h1>
    <p>Interactive Validation & Visualization WebApp • RoadAnomaly Benchmark (Lis et al., ICCV 2019) • Supervised U-Net vs. Pre-trained ResNet-34 vs. Unsupervised Autoencoder</p>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# MODULE 1: EXECUTIVE DASHBOARD & BENCHMARKS
# =========================================================================
if nav_page.startswith("📊 1."):
    # Top KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    best_row = suite["benchmark_df"].iloc[4]  # ResNet-34 + TTA + Morph
    base_row = suite["benchmark_df"].iloc[0]  # Baseline U-Net

    col1.metric(
        "Best Test Dice (F1)",
        f"{best_row['Test Dice (F1)']:.4f}",
        f"+{((best_row['Test Dice (F1)'] - base_row['Test Dice (F1)']) / base_row['Test Dice (F1)'] * 100):.0f}% vs Scratch"
    )
    col2.metric(
        "Best Test IoU (Jaccard)",
        f"{best_row['Test IoU']:.4f}",
        f"+{((best_row['Test IoU'] - base_row['Test IoU']) / base_row['Test IoU'] * 100):.0f}% vs Scratch"
    )
    col3.metric(
        "Test AUROC (Separability)",
        f"{suite['benchmark_df']['AUROC'].max():.4f}",
        f"Baseline: {base_row['AUROC']:.4f}"
    )
    col4.metric(
        "Test AUPRC (Avg Precision)",
        f"{suite['benchmark_df']['AUPRC (AP)'].max():.4f}",
        f"Baseline: {base_row['AUPRC (AP)']:.4f}"
    )
    col5.metric(
        "Test Pixel Accuracy",
        f"{best_row['Pixel Acc (%)']:.2f}%",
        f"+{best_row['Pixel Acc (%)'] - base_row['Pixel Acc (%)']:.2f}% vs Scratch"
    )

    st.markdown("### 🏆 Comprehensive Quantitative Benchmark & Ablation Study (Unseen Test Set)")
    st.markdown(
        "Comparison across our **three core deep learning paradigms** and **validation ablation enhancements** "
        "(Validation Threshold Calibration $\\tau^*$, Horizontal-Flip TTA, Morphological Post-Processing, and Hybrid Supervised+Unsupervised Fusion):"
    )

    styled_df = (
        suite["benchmark_df"]
        .drop(columns=["ShortName"])
        .style
        .highlight_max(subset=["Test IoU", "Test Dice (F1)", "Precision", "Recall", "Pixel Acc (%)", "AUROC", "AUPRC (AP)"], color="#bbf7d0")
        .highlight_min(subset=["FPR@95%TPR"], color="#bbf7d0")
        .format({
            "Threshold": "{:.2f}",
            "Test IoU": "{:.4f}",
            "Test Dice (F1)": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "Pixel Acc (%)": "{:.2f}%",
            "AUROC": "{:.4f}",
            "AUPRC (AP)": "{:.4f}",
            "FPR@95%TPR": "{:.4f}"
        })
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Interactive Charts Row
    c_left, c_right = st.columns([1.15, 0.85])

    with c_left:
        melt_df = suite["benchmark_df"].melt(
            id_vars=["ShortName", "Paradigm"],
            value_vars=["Test IoU", "Test Dice (F1)", "Precision", "Recall", "AUROC", "AUPRC (AP)"],
            var_name="Metric",
            value_name="Score"
        )
        fig_bar = px.bar(
            melt_df,
            x="Metric",
            y="Score",
            color="ShortName",
            barmode="group",
            color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82f6", "#0ea5e9", "#10b981", "#8b5cf6"],
            title="Multi-Metric Comparison Across Model Architectures & Ablations",
            height=430
        )
        fig_bar.update_layout(
            yaxis_range=[0, 1.05],
            legend_title_text="Model Variant",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        radar_categories = ["Test IoU", "Test Dice (F1)", "Precision", "Recall", "AUROC", "AUPRC (AP)"]
        fig_radar = go.Figure()
        radar_models = [
            (0, "Baseline U-Net (Scratch)", "#ef4444"),
            (1, "Conv Autoencoder (Unsup)", "#f59e0b"),
            (2, "Transfer ResNet-34 (τ=0.50)", "#3b82f6"),
            (4, "ResNet-34 + TTA + Morph", "#10b981"),
            (5, "Hybrid Ensemble", "#8b5cf6")
        ]
        for idx_row, name_label, col_hex in radar_models:
            row_vals = [float(suite["benchmark_df"].iloc[idx_row][m]) for m in radar_categories]
            row_vals.append(row_vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=row_vals,
                theta=radar_categories + [radar_categories[0]],
                fill="toself",
                name=name_label,
                line=dict(color=col_hex, width=2)
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
            title="Holistic Performance Radar Profile",
            height=430,
            margin=dict(l=40, r=40, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    # Methodology & Architecture Summary Cards
    st.markdown("### 🧠 Methodology, Architecture & Loss Engineering Summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.info(
            "**1. Baseline U-Net (From Scratch)**\n\n"
            "- 4-stage encoder-decoder (`32 → 64 → 128 → 256 → 512`)\n"
            "- Trained from random initialization\n"
            "- Demonstrates why 40 training frames are insufficient without pre-trained visual priors (**0.2083 Dice**)."
        )
    with m2:
        st.success(
            "**2. Transfer ResNet-34 U-Net**\n\n"
            "- ImageNet pre-trained **ResNet-34** encoder (`24.4M` params)\n"
            "- Reuses low/mid-level edge, texture & shape filters\n"
            "- Boosts Test IoU by **+315%** (`0.1163 → 0.4833`) and Dice by **+210%** (`0.2083 → 0.6449`)."
        )
    with m3:
        st.warning(
            "**3. Conv Autoencoder (Unsupervised)**\n\n"
            "- Trained strictly on L1 image reconstruction ($\\|x - \\hat{x}\\|_1$) with **zero mask labels**\n"
            "- Detects anomalies via per-pixel reconstruction error residuals\n"
            "- Outperforms scratch U-Net (**0.1554 IoU vs 0.1163**)."
        )
    with m4:
        st.error(
            "**4. Combo Loss & Hybrid Fusion**\n\n"
            "- Combats **90.15% : 9.85%** class imbalance via:\n"
            "  $\\mathcal{L} = 0.4\\,\\text{BCE} + 1.0\\,\\text{Dice} + 0.6\\,\\text{Focal}$\n"
            "- Hybrid Ensemble fuses supervised TTA probabilities ($0.85$) with calibrated Autoencoder residuals ($0.15$)."
        )


# =========================================================================
# MODULE 2: INTERACTIVE MODEL INSPECTOR (60 FRAMES)
# =========================================================================
elif nav_page.startswith("🔍 2."):
    st.markdown("### 🔍 Interactive Frame-by-Frame Model Inspector & Error Decomposition")
    st.markdown(
        "Select any frame from the **Test**, **Validation**, or **Training** split to inspect live neural network predictions, "
        "probability heatmaps, autoencoder reconstructions, and pixel-level **True Positive / False Negative / False Positive** decompositions."
    )

    f_col1, f_col2, f_col3 = st.columns([1.0, 1.0, 1.8])
    with f_col1:
        split_filter = st.selectbox("Filter by Dataset Split", ["TEST (Unseen 10 Frames)", "VAL (10 Frames)", "TRAIN (40 Frames)", "ALL (60 Frames)"], index=0)
    with f_col2:
        cat_options = ["All Categories"] + sorted(df_meta["category"].unique().tolist())
        cat_filter = st.selectbox("Filter by Hazard Category", cat_options, index=0)

    filtered_df = df_meta.copy()
    if not split_filter.startswith("ALL"):
        s_code = split_filter.split(" ")[0]
        filtered_df = filtered_df[filtered_df["split"] == s_code]
    if cat_filter != "All Categories":
        filtered_df = filtered_df[filtered_df["category"] == cat_filter]

    with f_col3:
        frame_options = [
            f"{row['filename']}  [{row['split']} | {row['category']} | {row['anomaly_ratio_pct']}% area]"
            for _, row in filtered_df.iterrows()
        ]
        selected_label = st.selectbox("Select Road Scene Frame", frame_options, index=0)
        selected_fname = selected_label.split("  [")[0]

    meta_row = df_meta[df_meta["filename"] == selected_fname].iloc[0]
    base_name = meta_row["base_name"]
    img_path = os.path.join(FRAMES_DIR, selected_fname)
    mask_path = os.path.join(FRAMES_DIR, f"{base_name}.labels", "labels_semantic.png")
    inst_path = os.path.join(FRAMES_DIR, f"{base_name}.labels", "labels_instance.png")

    pil_img = Image.open(img_path).convert("RGB")
    raw_mask = np.array(Image.open(mask_path))
    bin_mask = (raw_mask == 2).astype(np.float32)

    orig_rgb, resized_rgb, tensor_img = preprocess_pil_image(pil_img, IMG_SIZE)
    gt_resized = A.Resize(IMG_SIZE[0], IMG_SIZE[1])(image=orig_rgb, mask=bin_mask)["mask"]

    # Run live inference
    t0 = time.perf_counter()
    preds = run_single_inference(tensor_img, baseline_net, transfer_net, autoencoder, device)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    # Active primary model selection based on TTA toggle
    active_key = "resnet34_tta" if enable_tta else "resnet34"
    active_thresh = float(suite["optimal_thresholds"][active_key][0]) if use_calibrated_thresh else float(custom_threshold)

    pred_primary_bin = apply_postprocessing(preds[active_key], threshold=active_thresh, morph_clean=enable_morph)
    pred_base_bin = apply_postprocessing(preds["baseline"], threshold=0.50, morph_clean=False)
    cae_thresh = float(np.percentile(preds["cae"], 90))
    pred_cae_bin = (preds["cae"] >= cae_thresh).astype(np.float32)
    hyb_thresh = float(suite["optimal_thresholds"]["hybrid"][0]) if use_calibrated_thresh else float(custom_threshold)
    pred_hyb_bin = apply_postprocessing(preds["hybrid"], threshold=hyb_thresh, morph_clean=enable_morph)

    # Compute live metrics for this frame
    m_prim = compute_metrics(torch.from_numpy(pred_primary_bin), torch.from_numpy(gt_resized), threshold=0.5)
    m_base = compute_metrics(torch.from_numpy(pred_base_bin), torch.from_numpy(gt_resized), threshold=0.5)
    m_cae = compute_metrics(torch.from_numpy(pred_cae_bin), torch.from_numpy(gt_resized), threshold=0.5)
    m_hyb = compute_metrics(torch.from_numpy(pred_hyb_bin), torch.from_numpy(gt_resized), threshold=0.5)

    # Autonomous Emergency Braking / Safety Alert Banner
    pred_area_pct = float(np.mean(pred_primary_bin) * 100.0)
    max_conf = float(np.max(preds[active_key]) * 100.0)
    if pred_area_pct >= 2.0:
        st.markdown(
            f'<div class="safety-banner-crit">🚨 <b>AUTONOMOUS PERCEPTION ALERT — CRITICAL ROAD HAZARD DETECTED!</b> '
            f'Predicted Hazard Area: <b>{pred_area_pct:.2f}%</b> of scene (GT: {meta_row["anomaly_ratio_pct"]:.2f}%) | '
            f'Peak Confidence: <b>{max_conf:.1f}%</b> | Category: <b>{meta_row["category"]}</b> | '
            f'Live Inference Time: <b>{infer_ms:.1f} ms</b></div>',
            unsafe_allow_html=True
        )
    elif pred_area_pct >= 0.25:
        st.markdown(
            f'<div class="safety-banner-warn">⚠️ <b>AUTONOMOUS PERCEPTION CAUTION — SMALL / DISTANT OBSTACLE DETECTED!</b> '
            f'Predicted Hazard Area: <b>{pred_area_pct:.2f}%</b> of scene (GT: {meta_row["anomaly_ratio_pct"]:.2f}%) | '
            f'Peak Confidence: <b>{max_conf:.1f}%</b> | Category: <b>{meta_row["category"]}</b> | '
            f'Live Inference Time: <b>{infer_ms:.1f} ms</b></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="safety-banner-clear">ℹ️ <b>MINIMAL FOREGROUND ACTIVATION</b> — '
            f'Predicted Area: <b>{pred_area_pct:.2f}%</b> (GT: {meta_row["anomaly_ratio_pct"]:.2f}%) | '
            f'Try lowering the decision threshold τ in the sidebar to inspect faint hazard activations.</div>',
            unsafe_allow_html=True
        )

    # Frame KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("ResNet-34 Dice (F1)", f"{m_prim['dice']:.4f}", f"{m_prim['dice'] - m_base['dice']:+.4f} vs Baseline")
    k2.metric("ResNet-34 IoU", f"{m_prim['iou']:.4f}", f"{m_prim['iou'] - m_base['iou']:+.4f} vs Baseline")
    k3.metric("Precision", f"{m_prim['precision']:.4f}")
    k4.metric("Recall (Sensitivity)", f"{m_prim['recall']:.4f}")
    k5.metric("Pixel Accuracy", f"{m_prim['accuracy']*100:.2f}%")
    k6.metric("Active Threshold (τ)", f"{active_thresh:.2f}", "TTA+Morph ON" if (enable_tta and enable_morph) else "Custom")

    # Primary 4-Panel Diagnostic Row
    st.markdown("#### 🎯 Primary Diagnostic View: Input Scene, Ground Truth, ResNet-34 Overlay & Pixel Error Map")
    gt_overlay_img = make_overlay_image(resized_rgb, gt_resized, color=(34, 197, 94), alpha=overlay_alpha)
    pred_overlay_img = make_overlay_image(resized_rgb, pred_primary_bin, color=(239, 68, 68), alpha=overlay_alpha)
    err_decomp_img, px_counts = make_error_decomposition_rgb(resized_rgb, pred_primary_bin, gt_resized)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.image(resized_rgb, caption=f"1. Input Frame ({selected_fname})", use_container_width=True)
    with p2:
        st.image(gt_overlay_img, caption=f"2. Ground Truth Hazard Overlay (Green, {meta_row['anomaly_ratio_pct']}% px)", use_container_width=True)
    with p3:
        st.image(pred_overlay_img, caption=f"3. ResNet-34 Pred Overlay (Red, Dice={m_prim['dice']:.3f})", use_container_width=True)
    with p4:
        st.image(err_decomp_img, caption="4. Error Decomposition (Green=TP, Blue=FN, Red=FP)", use_container_width=True)

    total_eval_px = IMG_SIZE[0] * IMG_SIZE[1]
    st.markdown(
        f"**Pixel-Level Confusion Breakdown (`{IMG_SIZE[1]}×{IMG_SIZE[0]}` = `{total_eval_px:,}` px):** &nbsp; "
        f"<span class='legend-box' style='background:#10eb50;'></span>**True Positives (Hit):** `{px_counts['TP']:,}` px &nbsp;|&nbsp; "
        f"<span class='legend-box' style='background:#1e90ff;'></span>**False Negatives (Missed Hazard):** `{px_counts['FN']:,}` px &nbsp;|&nbsp; "
        f"<span class='legend-box' style='background:#ff3232;'></span>**False Positives (False Alarm):** `{px_counts['FP']:,}` px &nbsp;|&nbsp; "
        f"**True Negatives (Road/BG):** `{px_counts['TN']:,}` px",
        unsafe_allow_html=True
    )

    st.markdown("---")
    # Side-by-Side All-Paradigms Comparison
    st.markdown("#### 🔬 Side-by-Side Paradigm Comparison: Probability Heatmaps & Binary Predictions")
    view_mode = st.radio(
        "Select Visualization Mode across All Models:",
        ["Probability / Anomaly Heatmaps (Continuous)", "Thresholded Binary Masks (0/1)", "Blended Camera Overlays"],
        horizontal=True
    )

    m_cols = st.columns(5)
    paradigm_specs = [
        ("1. Ground Truth", gt_resized, None, "GT Mask"),
        (f"2. Baseline U-Net\n(Dice: {m_base['dice']:.3f} | IoU: {m_base['iou']:.3f})", preds["baseline"], pred_base_bin, "inferno"),
        (f"3. Transfer ResNet-34\n(Dice: {m_prim['dice']:.3f} | IoU: {m_prim['iou']:.3f})", preds[active_key], pred_primary_bin, "inferno"),
        (f"4. Conv Autoencoder\n(Dice: {m_cae['dice']:.3f} | IoU: {m_cae['iou']:.3f})", preds["cae_unit"], pred_cae_bin, "magma"),
        (f"5. Hybrid Ensemble\n(Dice: {m_hyb['dice']:.3f} | IoU: {m_hyb['iou']:.3f})", preds["hybrid"], pred_hyb_bin, "viridis")
    ]

    for col_ui, (title_str, prob_arr, bin_arr, cmap_str) in zip(m_cols, paradigm_specs):
        with col_ui:
            if bin_arr is None:
                # Ground truth column
                if view_mode.startswith("Blended"):
                    st.image(gt_overlay_img, caption=title_str, use_container_width=True)
                else:
                    st.image((prob_arr * 255).astype(np.uint8), caption=title_str, use_container_width=True)
            else:
                if view_mode.startswith("Probability"):
                    st.image(make_heatmap_rgb(prob_arr, cmap_str), caption=title_str, use_container_width=True)
                elif view_mode.startswith("Thresholded"):
                    st.image((bin_arr * 255).astype(np.uint8), caption=title_str, use_container_width=True)
                else:
                    st.image(make_overlay_image(resized_rgb, bin_arr, color=(255, 55, 55), alpha=overlay_alpha), caption=title_str, use_container_width=True)

    # Expandable Autoencoder Reconstruction Inspector
    with st.expander("🔍 Inspect Unsupervised Convolutional Autoencoder Reconstruction ($x$ vs. $\\hat{x}$ vs. $|x - \\hat{x}|$)", expanded=False):
        ae1, ae2, ae3 = st.columns(3)
        with ae1:
            st.image(resized_rgb, caption="Original Input Road Frame (x)", use_container_width=True)
        with ae2:
            st.image((preds["recon_rgb"] * 255).astype(np.uint8), caption="Autoencoder Reconstructed Scene (x̂)", use_container_width=True)
        with ae3:
            st.image(make_heatmap_rgb(preds["cae_unit"], "magma"), caption="Per-Pixel L1 Reconstruction Error Heatmap |x - x̂|", use_container_width=True)
        st.caption(
            "Because the Convolutional Autoencoder has a compressed spatial bottleneck and was trained with L1 loss, "
            "it reconstructs low-frequency road surfaces smoothly while failing to reconstruct sharp, unexpected obstacles—creating anomaly spikes in the residual map."
        )


# =========================================================================
# MODULE 3: DEEP VALIDATION SUITE (ROC / PR / CALIBRATION / SCALE)
# =========================================================================
elif nav_page.startswith("📈 3."):
    st.markdown("### 📈 Deep Validation Suite: Official RoadAnomaly Benchmark Protocol")

    val_tabs = st.tabs([
        "📉 1. ROC & Precision-Recall Curves",
        "🎯 2. Validation Threshold Calibration (τ*)",
        "📦 3. Per-Category & Object Scale Analysis",
        "🖼️ 4. Static Matplotlib Publication Figures"
    ])

    # Tab 1: ROC & PR Curves
    with val_tabs[0]:
        st.markdown(
            "Following **Lis et al. (ICCV 2019)**, pixel-level road anomaly segmentation under severe class imbalance ($9.85\\%$ foreground) "
            "is evaluated across all test pixels using threshold-independent curves: **AUROC**, **AUPRC (Average Precision)**, and **FPR@95%TPR**."
        )
        rc1, rc2 = st.columns(2)
        palette_map = [
            ("baseline", "Baseline U-Net (Scratch)", "#ef4444"),
            ("cae", "Conv Autoencoder (Unsupervised)", "#f59e0b"),
            ("resnet34", "Transfer ResNet-34 U-Net", "#3b82f6"),
            ("resnet34_tta", "ResNet-34 + TTA", "#10b981"),
            ("hybrid", "Hybrid Ensemble (ResNet34+CAE)", "#8b5cf6")
        ]

        with rc1:
            fig_roc = go.Figure()
            for k, label, col_hex in palette_map:
                res = suite["curve_results"][k]
                fig_roc.add_trace(go.Scatter(
                    x=res["fpr"],
                    y=res["tpr"],
                    mode="lines",
                    name=f"{label} (AUROC={res['auroc']:.4f})",
                    line=dict(color=col_hex, width=2.5)
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline (0.5000)",
                line=dict(color="gray", dash="dash")
            ))
            fig_roc.update_layout(
                title="Pixel-Level ROC Curve (Unseen Test Set)",
                xaxis_title="False Positive Rate (FPR)",
                yaxis_title="True Positive Rate (TPR / Recall)",
                height=460,
                legend=dict(x=0.35, y=0.08)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with rc2:
            fig_pr = go.Figure()
            for k, label, col_hex in palette_map:
                res = suite["curve_results"][k]
                fig_pr.add_trace(go.Scatter(
                    x=res["rec"],
                    y=res["prec"],
                    mode="lines",
                    name=f"{label} (AUPRC={res['auprc']:.4f})",
                    line=dict(color=col_hex, width=2.5)
                ))
            fig_pr.add_hline(
                y=suite["pos_ratio"],
                line_dash="dash",
                line_color="gray",
                annotation_text=f"No-Skill Baseline ({suite['pos_ratio']:.4f})"
            )
            fig_pr.update_layout(
                title="Pixel-Level Precision-Recall Curve (Unseen Test Set)",
                xaxis_title="Recall (Sensitivity)",
                yaxis_title="Precision (Positive Predictive Value)",
                height=460,
                legend=dict(x=0.32, y=0.98)
            )
            st.plotly_chart(fig_pr, use_container_width=True)

    # Tab 2: Validation Threshold Calibration
    with val_tabs[1]:
        st.markdown(
            "Evaluating imbalanced anomaly detectors solely at an arbitrary threshold $\\tau=0.50$ introduces threshold bias. "
            "Below is the **Validation Set Threshold Sweep** ($\\tau \\in [0.05, 0.95]$) used to lock in the optimal decision threshold $\\tau^*$ prior to testing."
        )
        fig_cal = go.Figure()
        for k, label, col_hex in palette_map:
            best_t, best_d = suite["optimal_thresholds"][k]
            fig_cal.add_trace(go.Scatter(
                x=suite["thresholds"],
                y=suite["calibration_curves"][k],
                mode="lines",
                name=f"{label} (τ*={best_t:.2f}, Val Dice={best_d:.4f})",
                line=dict(color=col_hex, width=2.5)
            ))
            fig_cal.add_trace(go.Scatter(
                x=[best_t],
                y=[best_d],
                mode="markers",
                marker=dict(color=col_hex, size=11, symbol="diamond"),
                showlegend=False
            ))
        fig_cal.add_vline(x=0.50, line_dash="dash", line_color="gray", annotation_text="Default τ = 0.50")
        fig_cal.update_layout(
            title="Validation Set Mean Dice Score vs. Decision Threshold (τ)",
            xaxis_title="Decision Threshold (τ)",
            yaxis_title="Validation Mean Dice (F1)",
            height=460
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    # Tab 3: Per-Category & Object Scale Sensitivity
    with val_tabs[2]:
        st.markdown("#### Granular Breakdown Across All 10 Unseen Test Frames & 5 Hazard Categories")
        cat_perf = suite["per_image_df"].groupby("Category").agg(
            Test_Frames=("Filename", "count"),
            Mean_Area_Pct=("Anomaly Area (%)", "mean"),
            Baseline_Dice=("Baseline Dice", "mean"),
            Autoencoder_Dice=("Autoencoder Dice", "mean"),
            ResNet34_Dice=("ResNet34 Dice", "mean"),
            ResNet34_TTA_Dice=("ResNet34+TTA Dice", "mean"),
            Hybrid_Dice=("Hybrid Dice", "mean")
        ).reset_index().sort_values(by="ResNet34_TTA_Dice", ascending=False)

        sc1, sc2 = st.columns(2)
        with sc1:
            cat_melt = cat_perf.melt(
                id_vars=["Category"],
                value_vars=["Baseline_Dice", "Autoencoder_Dice", "ResNet34_Dice", "ResNet34_TTA_Dice", "Hybrid_Dice"],
                var_name="Model",
                value_name="Mean Test Dice"
            )
            fig_cat = px.bar(
                cat_melt,
                x="Category",
                y="Mean Test Dice",
                color="Model",
                barmode="group",
                color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82f6", "#10b981", "#8b5cf6"],
                title="Test Dice (F1) Breakdown by Hazard Category",
                height=420
            )
            fig_cat.update_layout(yaxis_range=[0, 1.05])
            st.plotly_chart(fig_cat, use_container_width=True)

        with sc2:
            fig_scale = px.scatter(
                suite["per_image_df"],
                x="Anomaly Area (%)",
                y="ResNet34+TTA Dice",
                color="Category",
                size="Anomaly Area (%)",
                hover_data=["Filename", "Baseline Dice", "ResNet34 IoU", "Precision", "Recall"],
                title="Obstacle Scale (% Pixel Area) vs. Segmentation Dice Score",
                height=420
            )
            fig_scale.update_layout(yaxis_range=[0, 1.05])
            st.plotly_chart(fig_scale, use_container_width=True)

        st.markdown("##### 📋 Frame-by-Frame Unseen Test Set Metrics Table")
        st.dataframe(
            suite["per_image_df"].style.highlight_max(
                subset=["Baseline Dice", "Autoencoder Dice", "ResNet34 Dice", "ResNet34+TTA Dice", "Hybrid Dice"],
                color="#bbf7d0"
            ),
            use_container_width=True,
            hide_index=True
        )

    # Tab 4: Static Publication Figures
    with val_tabs[3]:
        st.markdown("High-resolution diagnostic figures generated and saved in `evaluation_results/`:")
        fig_files = [
            ("Training Loss & Validation Dice Convergence Curves", "training_convergence_curves.png"),
            ("Validation Set Threshold Calibration Sweep", "validation_threshold_calibration.png"),
            ("Pixel-Level ROC & Precision-Recall Benchmark Curves", "roc_pr_curves.png"),
            ("Per-Category & Object Scale Sensitivity Analysis", "per_category_and_scale_validation.png"),
            ("Best vs. Hardest Failure Case Pixel Error Decomposition", "failure_mode_decomposition.png"),
            ("Qualitative Side-by-Side Model Comparison Grid", "qualitative_model_comparison.png")
        ]
        for title_fig, fname_fig in fig_files:
            fpath = os.path.join(EVAL_DIR, fname_fig)
            if os.path.exists(fpath):
                st.markdown(f"##### {title_fig}")
                st.image(Image.open(fpath), use_container_width=True)
                st.markdown("---")


# =========================================================================
# MODULE 4: DATASET EXPLORER & CLASS IMBALANCE (EDA)
# =========================================================================
elif nav_page.startswith("🗂️ 4."):
    st.markdown("### 🗂️ Exploratory Data Analysis (EDA) & Stratified Dataset Taxonomy")

    total_px = int(df_meta["total_pixels"].sum())
    anom_px = int(df_meta["anomaly_pixels"].sum())
    bg_px = int(df_meta["bg_pixels"].sum())

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total Dataset Frames", "60 HD Frames", "1280 × 720 (.webp)")
    e2.metric("Total Pixels Audited", f"{total_px:,}", "55.3 Million Pixels")
    e3.metric("Anomaly Foreground Pixels", f"{anom_px:,}", f"{anom_px/total_px*100:.2f}% (Severe Imbalance)")
    e4.metric("Normal Road/BG Pixels", f"{bg_px:,}", f"{bg_px/total_px*100:.2f}% of Dataset")

    c1, c2, c3 = st.columns([1.1, 1.1, 0.9])
    with c1:
        cat_counts = df_meta.groupby(["category", "split"]).size().reset_index(name="Frames")
        fig_split = px.bar(
            cat_counts,
            x="category",
            y="Frames",
            color="split",
            barmode="stack",
            color_discrete_map={"TRAIN": "#2563eb", "VAL": "#f59e0b", "TEST": "#10b981"},
            title="Stratified Split Across 5 Hazard Categories (40 Train / 10 Val / 10 Test)",
            height=380
        )
        st.plotly_chart(fig_split, use_container_width=True)

    with c2:
        fig_box = px.box(
            df_meta,
            x="category",
            y="anomaly_ratio_pct",
            color="category",
            points="all",
            hover_data=["filename", "split"],
            title="Anomaly Pixel Area (%) Distribution by Category [0.37% to 42.33%]",
            height=380
        )
        fig_box.update_layout(showlegend=False, yaxis_title="Anomaly Area (%)")
        st.plotly_chart(fig_box, use_container_width=True)

    with c3:
        fig_pie = px.pie(
            names=["Background / Normal Road (0)", "Anomalous Hazard (2 → 1)"],
            values=[bg_px, anom_px],
            hole=0.48,
            color_discrete_sequence=["#334155", "#ef4444"],
            title="Global Pixel Class Imbalance (90.15% vs 9.85%)",
            height=380
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### 🖼️ Multi-Frame Category Gallery Browser")
    gal_cat = st.selectbox("Select Category to Browse Ground Truth Overlays:", sorted(df_meta["category"].unique().tolist()))
    gal_subset = df_meta[df_meta["category"] == gal_cat].head(8)

    cols_gal = st.columns(4)
    for idx_g, (_, r_gal) in enumerate(gal_subset.iterrows()):
        with cols_gal[idx_g % 4]:
            p_img = Image.open(os.path.join(FRAMES_DIR, r_gal["filename"])).convert("RGB")
            m_img = np.array(Image.open(os.path.join(FRAMES_DIR, f"{r_gal['base_name']}.labels", "labels_semantic.png")))
            _, r_rgb, _ = preprocess_pil_image(p_img, IMG_SIZE)
            m_res = A.Resize(IMG_SIZE[0], IMG_SIZE[1])(image=r_rgb, mask=(m_img == 2).astype(np.float32))["mask"]
            ov_g = make_overlay_image(r_rgb, m_res, color=(239, 68, 68), alpha=0.45)
            st.image(ov_g, caption=f"{r_gal['filename'][:24]} ({r_gal['split']} | {r_gal['anomaly_ratio_pct']}%)", use_container_width=True)

    with st.expander("📋 View Full 60-Frame Catalog Table (`dataset_summary.csv`)", expanded=False):
        st.dataframe(df_meta, use_container_width=True, hide_index=True)


# =========================================================================
# MODULE 5: LIVE CUSTOM IMAGE INFERENCE & REAL-TIME LATENCY PROFILER
# =========================================================================
elif nav_page.startswith("⚡ 5."):
    st.markdown("### ⚡ Live Custom Road Image Inference & Real-Time Deployment Profiler")
    st.markdown(
        "Upload **any road driving image** (`.jpg`, `.png`, `.webp`) to test our trained neural networks on out-of-distribution road scenes in real time, "
        "or run a live hardware latency benchmark."
    )

    uploaded_file = st.file_uploader("Upload a Road Scene Image (.webp, .png, .jpg, .jpeg)", type=["webp", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        custom_pil = Image.open(uploaded_file).convert("RGB")
        orig_w, orig_h = custom_pil.size
        _, custom_rgb, custom_tensor = preprocess_pil_image(custom_pil, IMG_SIZE)

        t0 = time.perf_counter()
        c_preds = run_single_inference(custom_tensor, baseline_net, transfer_net, autoencoder, device)
        c_ms = (time.perf_counter() - t0) * 1000.0

        active_key = "resnet34_tta" if enable_tta else "resnet34"
        active_thresh = float(suite["optimal_thresholds"][active_key][0]) if use_calibrated_thresh else float(custom_threshold)
        c_bin = apply_postprocessing(c_preds[active_key], threshold=active_thresh, morph_clean=enable_morph)
        c_ov = make_overlay_image(custom_rgb, c_bin, color=(239, 68, 68), alpha=overlay_alpha)

        anom_pct = float(np.mean(c_bin) * 100.0)
        st.success(f"✅ Live Inference Complete in **{c_ms:.1f} ms** | Original Size: `{orig_w}×{orig_h}` | Predicted Hazard Coverage: **{anom_pct:.2f}%** (at τ={active_thresh:.2f})")

        u1, u2, u3, u4 = st.columns(4)
        with u1:
            st.image(custom_rgb, caption="1. Uploaded Road Scene", use_container_width=True)
        with u2:
            st.image(c_ov, caption=f"2. ResNet-34 Anomaly Overlay ({anom_pct:.2f}% area)", use_container_width=True)
        with u3:
            st.image(make_heatmap_rgb(c_preds[active_key], "inferno"), caption="3. ResNet-34 Probability Heatmap", use_container_width=True)
        with u4:
            st.image(make_heatmap_rgb(c_preds["hybrid"], "viridis"), caption="4. Hybrid Ensemble Heatmap (Sup+Unsup)", use_container_width=True)
    else:
        st.info("👆 Upload an image above to run live out-of-distribution anomaly segmentation, or use **Module 2 (Interactive Model Inspector)** to test all 60 dataset frames with ground-truth masks.")

    st.markdown("---")
    st.markdown("#### ⏱️ Onboard Autonomous Driving Latency & Complexity Benchmark")
    st.markdown(
        "For deployment on an autonomous vehicle, perception models must operate in real time ($\\ge 30\\text{ FPS}$). "
        "Click the button below to benchmark single-frame inference latency and throughput on the active compute device."
    )

    if st.button("🚀 Run Live Hardware Latency & FPS Benchmark (30 Runs)", type="primary"):
        with st.spinner("Profiling Baseline U-Net, Transfer ResNet-34 U-Net, and Conv Autoencoder..."):
            prof_rows = []
            specs_prof = [
                (baseline_net, "1. Baseline U-Net (From Scratch)", os.path.join(CKPT_DIR, "baseline_unet_best.pth")),
                (transfer_net, "2. Transfer Learning U-Net (ResNet-34)", os.path.join(CKPT_DIR, "transfer_resnet34_unet_best.pth")),
                (autoencoder, "3. Conv Autoencoder (Unsupervised)", os.path.join(CKPT_DIR, "convautoencoder_best.pth"))
            ]
            dummy_in = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1], device=device)
            for mod, m_title, ckpt_p in specs_prof:
                params_m = sum(p.numel() for p in mod.parameters()) / 1e6
                size_mb = os.path.getsize(ckpt_p) / (1024 ** 2) if os.path.exists(ckpt_p) else (params_m * 4)
                with torch.no_grad():
                    for _ in range(5):
                        _ = mod(dummy_in)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_start = time.perf_counter()
                    runs = 30
                    for _ in range(runs):
                        _ = mod(dummy_in)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dt = (time.perf_counter() - t_start) / runs
                lat_ms = dt * 1000.0
                fps_val = 1.0 / dt
                prof_rows.append({
                    "Model Architecture": m_title,
                    "Parameters (M)": round(params_m, 2),
                    "Checkpoint Size (MB)": round(size_mb, 2),
                    "Latency (ms/frame)": round(lat_ms, 2),
                    "Throughput (FPS)": round(fps_val, 1),
                    "Real-Time Ready (>=30 FPS)": "✅ Yes" if fps_val >= 30 else "⚠️ CPU/Bound"
                })
            st.dataframe(pd.DataFrame(prof_rows), use_container_width=True, hide_index=True)

