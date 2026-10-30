import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def generate_stratified_splits(df, save_path="dataset_splits.json", seed=42):
    """
    Splits the 60 frames into Train (~67%), Val (~17%), and Test (~17%) sets
    stratified by hazard category. For minority categories (3 samples like Lost Tires and
    Unusual Vehicles), ensures exactly 1 sample in Train, 1 in Val, and 1 in Test.
    """
    np.random.seed(seed)
    
    train_filenames = []
    val_filenames = []
    test_filenames = []
    
    for category, group in df.groupby("category"):
        filenames = group["filename"].sample(frac=1.0, random_state=seed).tolist()
        n = len(filenames)
        
        if n == 3:
            # 1 to train, 1 to val, 1 to test
            train_filenames.append(filenames[0])
            val_filenames.append(filenames[1])
            test_filenames.append(filenames[2])
        elif n == 8:
            # Cones: 6 train, 1 val, 1 test
            train_filenames.extend(filenames[:6])
            val_filenames.append(filenames[6])
            test_filenames.append(filenames[7])
        elif n == 15:
            # Obstacles: 11 train, 2 val, 2 test
            train_filenames.extend(filenames[:11])
            val_filenames.extend(filenames[11:13])
            test_filenames.extend(filenames[13:15])
        elif n == 31:
            # Animals: 21 train, 5 val, 5 test
            train_filenames.extend(filenames[:21])
            val_filenames.extend(filenames[21:26])
            test_filenames.extend(filenames[26:31])
        else:
            # Fallback general split: 70 / 15 / 15
            n_val = max(1, int(round(n * 0.15)))
            n_test = max(1, int(round(n * 0.15)))
            n_train = n - n_val - n_test
            train_filenames.extend(filenames[:n_train])
            val_filenames.extend(filenames[n_train:n_train + n_val])
            test_filenames.extend(filenames[n_train + n_val:])
            
    # Calculate category distributions
    train_df = df[df["filename"].isin(train_filenames)]
    val_df = df[df["filename"].isin(val_filenames)]
    test_df = df[df["filename"].isin(test_filenames)]
    
    splits = {
        "train": train_filenames,
        "val": val_filenames,
        "test": test_filenames,
        "meta": {
            "train_count": len(train_filenames),
            "val_count": len(val_filenames),
            "test_count": len(test_filenames),
            "train_category_dist": train_df["category"].value_counts().to_dict(),
            "val_category_dist": val_df["category"].value_counts().to_dict(),
            "test_category_dist": test_df["category"].value_counts().to_dict()
        }
    }
    
    with open(save_path, "w") as f:
        json.dump(splits, f, indent=2)
        
    print(f"Stratified splits successfully generated:")
    print(f"  Train: {len(train_filenames)} frames ({len(train_filenames)/len(df)*100:.1f}%)")
    print(f"  Val  : {len(val_filenames)} frames ({len(val_filenames)/len(df)*100:.1f}%)")
    print(f"  Test : {len(test_filenames)} frames ({len(test_filenames)/len(df)*100:.1f}%)")
    return splits


def get_transforms(img_size=(384, 512)):
    """
    Data Augmentation and Preprocessing pipeline via Albumentations.
    Standardizes dimensions to (H, W) = (384, 512), exact multiples of 32.
    """
    height, width = img_size
    
    train_transform = A.Compose([
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.GaussNoise(p=0.25),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(height, width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


class RoadAnomalyDataset(Dataset):
    """
    PyTorch Dataset for RoadAnomaly semantic segmentation.
    Loads images and semantic masks (0 = background, 2 = anomaly).
    Normalizes masks to binary tensors: 1.0 = Anomaly, 0.0 = Road/Background.
    """
    def __init__(self, filenames, root_dir=None, transform=None):
        self.root_dir = root_dir or os.getcwd()
        self.frames_dir = os.path.join(self.root_dir, "frames")
        self.filenames = filenames
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        base_name = os.path.splitext(filename)[0]
        
        img_path = os.path.join(self.frames_dir, filename)
        mask_path = os.path.join(self.frames_dir, f"{base_name}.labels", "labels_semantic.png")
        
        image = np.array(Image.open(img_path).convert("RGB"))
        raw_mask = np.array(Image.open(mask_path))
        
        # Binary remapping: label 2 is anomaly -> 1, label 0 is background -> 0
        binary_mask = (raw_mask == 2).astype(np.float32)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=binary_mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].unsqueeze(0)  # Shape: [1, H, W]
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()
            
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": filename,
            "base_name": base_name
        }


def get_dataloaders(splits_file="dataset_splits.json", img_size=(384, 512), batch_size=4, num_workers=0):
    with open(splits_file, "r") as f:
        splits = json.load(f)
        
    train_tf, val_tf = get_transforms(img_size)
    
    train_ds = RoadAnomalyDataset(splits["train"], transform=train_tf)
    val_ds = RoadAnomalyDataset(splits["val"], transform=val_tf)
    test_ds = RoadAnomalyDataset(splits["test"], transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    csv_file = os.path.join("eda_outputs", "dataset_summary.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        generate_stratified_splits(df)
        train_loader, val_loader, test_loader = get_dataloaders()
        sample = next(iter(train_loader))
        print("Sample batch loaded successfully:")
        print("  Image batch shape:", sample["image"].shape)
        print("  Mask batch shape :", sample["mask"].shape)
        print("  Mask unique vals :", torch.unique(sample["mask"]))
