import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from losses import ComboLoss, compute_metrics
from models import BaselineUNet, build_transfer_unet, ConvAutoencoder

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate_model(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_metrics = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            metrics = compute_metrics(probs, masks, threshold=threshold)
            all_metrics.append(metrics)

    avg_loss = running_loss / len(loader.dataset)
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    avg_metrics["loss"] = avg_loss
    return avg_metrics


def train_segmentation_model(model, model_name, train_loader, val_loader, device, epochs=25, lr=3e-4):
    print(f"\n{'='*55}\nStarting Training: {model_name} ({epochs} epochs, lr={lr})\n{'='*55}")
    criterion = ComboLoss(alpha=0.4, beta=1.0, gamma=0.6)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_dice = 0.0
    best_weights_path = os.path.join("checkpoints", f"{model_name.lower().replace(' ', '_')}_best.pth")
    os.makedirs("checkpoints", exist_ok=True)
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"]
        })

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), best_weights_path)
            improved = "*"
        else:
            improved = ""

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f} | Val Dice: {val_metrics['dice']:.4f} {improved}")

    print(f"Training completed for {model_name}! Best Val Dice: {best_val_dice:.4f}")
    # Load best weights
    model.load_state_dict(torch.load(best_weights_path))
    return model, history


def train_autoencoder(model, train_loader, val_loader, device, epochs=20, lr=5e-4):
    print(f"\n{'='*55}\nStarting Training: ConvAutoencoder ({epochs} epochs)\n{'='*55}")
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_weights_path = os.path.join("checkpoints", "convautoencoder_best.pth")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                recon = model(images)
                loss = criterion(recon, images)
                running_val_loss += loss.item() * images.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)
            improved = "*"
        else:
            improved = ""

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Recon Train Loss: {train_loss:.4f} | Recon Val Loss: {val_loss:.4f} {improved}")

    model.load_state_dict(torch.load(best_weights_path))
    print(f"Autoencoder training finished! Best Val Recon Loss: {best_val_loss:.4f}")
    return model


def evaluate_autoencoder_test(model, test_loader, device, percentile=90):
    """
    Evaluates autoencoder on test set by thresholding the reconstruction error map.
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            error_map = model.compute_anomaly_map(images) # [B, 1, H, W]
            
            # Normalize error map to [0, 1] per sample
            B = error_map.size(0)
            for i in range(B):
                em = error_map[i, 0]
                em_min, em_max = em.min(), em.max()
                norm_em = (em - em_min) / (em_max - em_min + 1e-7)
                
                # Threshold at the top percentile of errors
                thresh_val = np.percentile(norm_em.cpu().numpy(), percentile)
                pred_mask = (norm_em >= thresh_val).float().unsqueeze(0).unsqueeze(0)
                
                m = compute_metrics(pred_mask, masks[i:i+1])
                all_metrics.append(m)
                
    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(img_size=(384, 512), batch_size=4)

    # 1. Train Baseline U-Net (from scratch)
    baseline_model = BaselineUNet().to(device)
    baseline_model, b_hist = train_segmentation_model(
        baseline_model, "Baseline UNet", train_loader, val_loader, device, epochs=25, lr=3e-4
    )

    # 2. Train Transfer Learning U-Net (ResNet-34)
    tl_model = build_transfer_unet(encoder_name="resnet34").to(device)
    tl_model, tl_hist = train_segmentation_model(
        tl_model, "Transfer ResNet34 UNet", train_loader, val_loader, device, epochs=25, lr=3e-4
    )

    # 3. Train Convolutional Autoencoder
    cae_model = ConvAutoencoder().to(device)
    cae_model = train_autoencoder(cae_model, train_loader, val_loader, device, epochs=20, lr=5e-4)

    # 4. Comprehensive Evaluation on Test Set
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SET EVALUATION (Unseen 10 Frames)")
    print("="*60)

    criterion = ComboLoss()
    b_test = evaluate_model(baseline_model, test_loader, criterion, device)
    tl_test = evaluate_model(tl_model, test_loader, criterion, device)
    cae_test = evaluate_autoencoder_test(cae_model, test_loader, device)

    results_table = pd.DataFrame([
        {
            "Model Approach": "Baseline U-Net (From Scratch)",
            "Test IoU": round(b_test["iou"], 4),
            "Test Dice (F1)": round(b_test["dice"], 4),
            "Test Precision": round(b_test["precision"], 4),
            "Test Recall": round(b_test["recall"], 4),
            "Pixel Accuracy": round(b_test["accuracy"], 4)
        },
        {
            "Model Approach": "Transfer Learning U-Net (ResNet-34)",
            "Test IoU": round(tl_test["iou"], 4),
            "Test Dice (F1)": round(tl_test["dice"], 4),
            "Test Precision": round(tl_test["precision"], 4),
            "Test Recall": round(tl_test["recall"], 4),
            "Pixel Accuracy": round(tl_test["accuracy"], 4)
        },
        {
            "Model Approach": "Conv Autoencoder (Reconstruction Error)",
            "Test IoU": round(cae_test["iou"], 4),
            "Test Dice (F1)": round(cae_test["dice"], 4),
            "Test Precision": round(cae_test["precision"], 4),
            "Test Recall": round(cae_test["recall"], 4),
            "Pixel Accuracy": round(cae_test["accuracy"], 4)
        }
    ])

    os.makedirs("evaluation_results", exist_ok=True)
    results_csv = os.path.join("evaluation_results", "model_comparison_metrics.csv")
    results_table.to_csv(results_csv, index=False)
    print(results_table.to_string(index=False))
    print(f"\nResults metrics saved to: {results_csv}")

    # 5. Visual Comparison on Test Images
    print("\nGenerating qualitative side-by-side test visualizations...")
    baseline_model.eval()
    tl_model.eval()
    cae_model.eval()

    sample_test_batch = next(iter(test_loader))
    test_imgs = sample_test_batch["image"].to(device)
    test_masks = sample_test_batch["mask"].to(device)
    test_names = sample_test_batch["filename"]

    with torch.no_grad():
        b_logits = baseline_model(test_imgs)
        b_preds = (torch.sigmoid(b_logits) >= 0.5).float().cpu().numpy()

        tl_logits = tl_model(test_imgs)
        tl_preds = (torch.sigmoid(tl_logits) >= 0.5).float().cpu().numpy()

        cae_errs = cae_model.compute_anomaly_map(test_imgs).cpu().numpy()

    # Denormalize image for plotting
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    num_samples = min(4, test_imgs.size(0))
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 3.8 * num_samples))

    for idx in range(num_samples):
        # Denormalize image
        img_np = test_imgs[idx].cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip((img_np * std + mean), 0, 1)

        gt_mask = test_masks[idx, 0].cpu().numpy()
        b_pred = b_preds[idx, 0]
        tl_pred = tl_preds[idx, 0]
        cae_err = cae_errs[idx, 0]

        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f"Test Input ({test_names[idx][:18]}...)", fontsize=10)
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(gt_mask, cmap="gray")
        axes[idx, 1].set_title("Ground Truth Mask", fontsize=10)
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(b_pred, cmap="inferno")
        axes[idx, 2].set_title("Baseline U-Net Pred", fontsize=10)
        axes[idx, 2].axis("off")

        axes[idx, 3].imshow(tl_pred, cmap="inferno")
        axes[idx, 3].set_title("Transfer ResNet-34 Pred", fontsize=10)
        axes[idx, 3].axis("off")

        axes[idx, 4].imshow(cae_err, cmap="magma")
        axes[idx, 4].set_title("Autoencoder Error Map", fontsize=10)
        axes[idx, 4].axis("off")

    plt.tight_layout()
    comp_plot_path = os.path.join("evaluation_results", "qualitative_model_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Qualitative visualization plot saved to: {comp_plot_path}")

    # Plot Training Convergence Curves
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot([h["epoch"] for h in b_hist], [h["train_loss"] for h in b_hist], label="Baseline Train Loss", linestyle="--")
    plt.plot([h["epoch"] for h in b_hist], [h["val_loss"] for h in b_hist], label="Baseline Val Loss")
    plt.plot([h["epoch"] for h in tl_hist], [h["train_loss"] for h in tl_hist], label="Transfer Train Loss", linestyle="--")
    plt.plot([h["epoch"] for h in tl_hist], [h["val_loss"] for h in tl_hist], label="Transfer Val Loss")
    plt.title("Training & Validation Loss Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Combo Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([h["epoch"] for h in b_hist], [h["val_dice"] for h in b_hist], label="Baseline Val Dice (F1)")
    plt.plot([h["epoch"] for h in tl_hist], [h["val_dice"] for h in tl_hist], label="Transfer Val Dice (F1)", linewidth=2.5)
    plt.title("Validation Dice Score Over Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score (F1)")
    plt.legend()

    plt.tight_layout()
    curves_plot_path = os.path.join("evaluation_results", "training_convergence_curves.png")
    plt.savefig(curves_plot_path, dpi=300)
    plt.close()
    print(f"Convergence curves saved to: {curves_plot_path}")

if __name__ == "__main__":
    main()
