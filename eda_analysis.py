import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(root_dir, "frames")
    frame_list_file = os.path.join(root_dir, "frame_list.json")
    output_dir = os.path.join(root_dir, "eda_outputs")
    os.makedirs(output_dir, exist_ok=True)

    with open(frame_list_file, "r") as f:
        frame_list = json.load(f)

    print(f"Total frames listed: {len(frame_list)}")

    records = []
    category_counts = {}

    for name in frame_list:
        base_name = os.path.splitext(name)[0]
        category = base_name.split("0")[0].split("1")[0].split("2")[0].split("3")[0].split("_")[0]
        # Clean category name
        if "animals" in base_name:
            cat = "Animals"
        elif "cones" in base_name:
            cat = "Traffic Cones"
        elif "lost_tires" in base_name:
            cat = "Lost Tires"
        elif "obstacles" in base_name:
            cat = "Obstacles/Rocks"
        elif "vehicle" in base_name:
            cat = "Unusual Vehicles"
        else:
            cat = "Other"

        img_path = os.path.join(frames_dir, name)
        semantic_mask_path = os.path.join(frames_dir, f"{base_name}.labels", "labels_semantic.png")
        instance_mask_path = os.path.join(frames_dir, f"{base_name}.labels", "labels_instance.png")

        img_exists = os.path.exists(img_path)
        semantic_exists = os.path.exists(semantic_mask_path)
        instance_exists = os.path.exists(instance_mask_path)

        if not (img_exists and semantic_exists):
            print(f"Warning: missing file for {base_name}")
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        mask_np = np.array(Image.open(semantic_mask_path))
        unique_labels = np.unique(mask_np)

        total_pixels = mask_np.size
        # Semantic labels: 0 = background, 2 = anomaly
        anomaly_pixels = np.sum(mask_np == 2)
        bg_pixels = np.sum(mask_np == 0)
        other_pixels = total_pixels - (anomaly_pixels + bg_pixels)
        anomaly_ratio = (anomaly_pixels / total_pixels) * 100

        has_instance = instance_exists and os.path.getsize(instance_mask_path) > 0

        records.append({
            "filename": name,
            "base_name": base_name,
            "category": cat,
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 2),
            "total_pixels": total_pixels,
            "anomaly_pixels": anomaly_pixels,
            "bg_pixels": bg_pixels,
            "other_pixels": other_pixels,
            "anomaly_ratio_pct": round(anomaly_ratio, 2),
            "unique_labels": str(unique_labels.tolist()),
            "has_instance_mask": has_instance
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "dataset_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset summary saved to: {csv_path}")

    # Aggregated statistics
    print("\n--- Summary Statistics ---")
    print(f"Total valid images analyzed: {len(df)}")
    print(f"Categories distribution:\n{df['category'].value_counts()}")
    print(f"\nImage Resolutions: min {df['width'].min()}x{df['height'].min()}, max {df['width'].max()}x{df['height'].max()}")
    print(f"Mean Anomaly Pixel Ratio: {df['anomaly_ratio_pct'].mean():.2f}% (min: {df['anomaly_ratio_pct'].min()}%, max: {df['anomaly_ratio_pct'].max()}%)")
    print(f"Total anomaly pixels across dataset: {df['anomaly_pixels'].sum()} / {df['total_pixels'].sum()} ({df['anomaly_pixels'].sum() / df['total_pixels'].sum() * 100:.2f}%)")
    print(f"All observed unique semantic label values: {sorted(list(set([val for sublist in df['unique_labels'].apply(eval) for val in sublist])))}")

    # Visualization 1: Category Distribution and Anomaly Area Distribution
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x="category", palette="viridis", order=df["category"].value_counts().index)
    plt.title("Image Count per Hazard Category", fontsize=12, fontweight="bold")
    plt.xlabel("Category")
    plt.ylabel("Number of Frames")
    plt.xticks(rotation=25)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x="category", y="anomaly_ratio_pct", palette="magma", order=df["category"].value_counts().index)
    plt.title("Anomaly Pixel % by Category (Class Imbalance)", fontsize=12, fontweight="bold")
    plt.xlabel("Category")
    plt.ylabel("Anomaly Pixels (%)")
    plt.xticks(rotation=25)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "eda_distribution_plots.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"EDA distribution plots saved to: {plot_path}")

    # Visualization 2: Sample Image + Mask Overlay for each Category
    unique_categories = df["category"].unique()
    n_cats = len(unique_categories)
    fig, axes = plt.subplots(n_cats, 3, figsize=(12, 3.2 * n_cats))

    for i, cat in enumerate(unique_categories):
        sample_row = df[df["category"] == cat].iloc[0]
        img_p = os.path.join(frames_dir, sample_row["filename"])
        mask_p = os.path.join(frames_dir, f"{sample_row['base_name']}.labels", "labels_semantic.png")

        img = Image.open(img_p).convert("RGB")
        mask = np.array(Image.open(mask_p))

        # Binary anomaly mask: 1 where label == 2
        binary_mask = (mask == 2).astype(np.uint8)

        # Create overlay
        img_np = np.array(img).copy()
        overlay = img_np.copy()
        # Highlight anomaly in semi-transparent red
        overlay[binary_mask == 1] = [255, 30, 30]
        blended = (0.6 * img_np + 0.4 * overlay).astype(np.uint8)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"[{cat}] Original Image", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(binary_mask, cmap="gray")
        axes[i, 1].set_title(f"Ground Truth (Anomaly %: {sample_row['anomaly_ratio_pct']}%)", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(blended)
        axes[i, 2].set_title(f"Mask Overlay ({sample_row['base_name'][:20]}...)", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()
    overlay_path = os.path.join(output_dir, "eda_sample_overlays.png")
    plt.savefig(overlay_path, dpi=300)
    plt.close()
    print(f"EDA sample overlays saved to: {overlay_path}")

if __name__ == "__main__":
    main()
