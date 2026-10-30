import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Computes the soft Dice Loss for binary segmentation.
    Addresses severe foreground-background class imbalance by directly
    optimizing the spatial overlap (Dice coefficient) between prediction and ground truth.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss down-weights easy background examples and focuses gradient updates
    on hard-to-detect foreground anomalies (e.g. small rocks, distant tires).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_loss = focal_weight * ((1.0 - p_t) ** self.gamma) * bce_loss
        return focal_loss.mean()


class ComboLoss(nn.Module):
    """
    Combines Binary Cross-Entropy (BCE), Dice Loss, and Focal Loss.
    Loss = alpha * BCE + beta * Dice + gamma * Focal
    """
    def __init__(self, alpha=0.4, beta=1.0, gamma=0.6):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets) if self.alpha > 0 else 0.0
        loss_dice = self.dice(logits, targets) if self.beta > 0 else 0.0
        loss_focal = self.focal(logits, targets) if self.gamma > 0 else 0.0
        return self.alpha * loss_bce + self.beta * loss_dice + self.gamma * loss_focal


def compute_metrics(probs, targets, threshold=0.5, eps=1e-7):
    """
    Computes standard semantic segmentation and anomaly evaluation metrics:
    - IoU (Intersection over Union / Jaccard Index)
    - Dice Coefficient (F1-score)
    - Precision (Sensitivity of detected hazard pixels)
    - Recall (Fraction of true anomaly pixels detected)
    - Accuracy (Overall pixel accuracy)
    """
    preds = (probs >= threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    # Smoke test loss functions
    dummy_logits = torch.randn(2, 1, 64, 64)
    dummy_targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    criterion = ComboLoss()
    loss = criterion(dummy_logits, dummy_targets)
    probs = torch.sigmoid(dummy_logits)
    metrics = compute_metrics(probs, dummy_targets)
    
    print(f"Loss smoke test passed! Combo Loss: {loss.item():.4f}")
    print("Metrics smoke test:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
