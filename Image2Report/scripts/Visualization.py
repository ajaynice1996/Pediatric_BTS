# import os
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt

# # Paths
# pred_dir = "../nnUNet/nnUNet_results/Dataset2024/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation/"
# gt_dir = "../nnUNet/nnUNet_preprocessed/Dataset2024/gt_segmentations/"

# # Get common case names
# pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
# print(f"Found {len(pred_files)} prediction files.")
# case_names = [f.replace('.nii.gz', '') for f in pred_files]

# # Only show 5 examples
# samples = case_names[:5]

# def middle_slice(volume):
#     # Return middle axial slice
#     return volume[:, :, volume.shape[2] // 3]

# def get_rgb_overlay(gt, pred):
#     """
#     Return an RGB overlay image:
#     - Green: True Positive
#     - Red: False Negative
#     - Blue: False Positive
#     """
#     tp = np.logical_and(gt > 0, pred > 0)
#     fn = np.logical_and(gt > 0, pred == 0)
#     fp = np.logical_and(gt == 0, pred > 0)

#     overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
#     overlay[..., 0] = fn * 255  # Red
#     overlay[..., 1] = tp * 255  # Green
#     overlay[..., 2] = fp * 255  # Blue
#     return overlay

# # Plot
# fig, axes = plt.subplots(len(samples), 3, figsize=(12, len(samples) * 3))
# if len(samples) == 1:
#     axes = axes[np.newaxis, :]  # Ensure 2D indexing

# for row, case in enumerate(samples):
#     pred_path = os.path.join(pred_dir, f"{case}.nii.gz")
#     gt_path = os.path.join(gt_dir, f"{case}.nii.gz")

#     pred = nib.load(pred_path).get_fdata().astype(np.uint8)
#     gt = nib.load(gt_path).get_fdata().astype(np.uint8)

#     pred_slice = middle_slice(pred)
#     gt_slice = middle_slice(gt)
#     overlay = get_rgb_overlay(gt_slice, pred_slice)

#     axes[row, 0].imshow(gt_slice, cmap='gray')
#     axes[row, 0].set_title(f"{case} - Ground Truth")
#     axes[row, 1].imshow(pred_slice, cmap='gray')
#     axes[row, 1].set_title("Prediction")
#     axes[row, 2].imshow(overlay)
#     axes[row, 2].set_title("Overlay (TP=G, FN=R, FP=B)")

#     for ax in axes[row]:
#         ax.axis('off')

# plt.tight_layout()
# plt.savefig("prediction_vs_gt_overlay.png", dpi=150)
# plt.show()

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib 
from sklearn.metrics import f1_score

# Paths
pred_dir = "nnUNet_results/Dataset202_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation"
gt_dir = "nnUNet_preprocessed/Dataset202_BraTS/gt_segmentations"

# Composite label definitions
LABELS = {
    "ET": [1],
    "TC": [1, 2, 3],
    "WT": [1, 2, 3, 4],
}

def middle_slice(volume):
    return volume[:, :, volume.shape[2] // 3]

def get_overlay(gt, pred):
    """Red=FN, Green=TP, Blue=FP"""
    tp = np.logical_and(gt, pred)
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(~gt, pred)
    overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
    overlay[..., 0] = fn * 255  # Red
    overlay[..., 1] = tp * 255  # Green
    overlay[..., 2] = fp * 255  # Blue
    return overlay

def compute_dice(pred, gt):
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    intersection = np.logical_and(pred_flat, gt_flat).sum()
    return (2. * intersection) / (pred.sum() + gt.sum() + 1e-8)

# List of cases to plot
case_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
case_names = sorted([f.replace(".nii.gz", "") for f in case_files])[:5]  # Choose 5 samples

# Create figure: rows = cases, columns = 3 (GT/Pred/Overlay) × 3 labels
fig, axes = plt.subplots(len(case_names), 9, figsize=(18, 3 * len(case_names)))

if len(case_names) == 1:
    axes = axes[np.newaxis, :]

for i, case_id in enumerate(case_names):
    pred_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
    gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")

    pred = nib.load(pred_path).get_fdata().astype(np.uint8)
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)

    col = 0
    for label_name, label_vals in LABELS.items():
        pred_mask = np.isin(pred, label_vals)
        gt_mask = np.isin(gt, label_vals)

        pred_slice = middle_slice(pred_mask)
        gt_slice = middle_slice(gt_mask)
        overlay = get_overlay(gt_slice, pred_slice)
        dice = compute_dice(pred_slice, gt_slice)

        # GT
        axes[i, col].imshow(gt_slice, cmap='gray')
        axes[i, col].set_title(f"{case_id} - GT ({label_name})")
        axes[i, col].axis("off")
        col += 1

        # Prediction
        axes[i, col].imshow(pred_slice, cmap='gray')
        axes[i, col].set_title(f"Pred ({label_name})")
        axes[i, col].axis("off")
        col += 1

        # Overlay
        axes[i, col].imshow(overlay)
        axes[i, col].set_title(f"Overlay\nDice={dice:.3f}")
        axes[i, col].axis("off")
        col += 1

plt.tight_layout()
plt.savefig("brats_composite_comparison0-5.png", dpi=150)
print("Saved as brats_composite_comparison.png")
plt.show()

# import os
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt

# # Paths
# pred_dir = "../nnUNet/nnUNet_results/Dataset2024/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation"
# gt_dir = "../nnUNet/nnUNet_preprocessed/Dataset2024/gt_segmentations"

# # Composite label definitions
# LABELS = {
#     "ET": [1],
#     "TC": [1, 2, 3],
#     "WT": [1, 2, 3, 4],
# }

# def middle_slice(volume):
#     return volume[:, :, volume.shape[2] // 2]

# def get_overlay(gt, pred):
#     """
#     RGB overlay: Green=TP, Red=FN, Blue=FP
#     """
#     tp = np.logical_and(gt, pred)
#     fn = np.logical_and(gt, np.logical_not(pred))
#     fp = np.logical_and(np.logical_not(gt), pred)

#     overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
#     overlay[..., 0] = fn * 255  # Red
#     overlay[..., 1] = tp * 255  # Green
#     overlay[..., 2] = fp * 255  # Blue
#     return overlay

# # Choose 5 cases
# case_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
# case_names = sorted([f.replace(".nii.gz", "") for f in case_files])[:5]

# # Start plotting
# for label_name, label_values in LABELS.items():
#     fig, axes = plt.subplots(len(case_names), 3, figsize=(12, 3 * len(case_names)))

#     if len(case_names) == 1:
#         axes = axes[np.newaxis, :]  # Ensure 2D index

#     for i, case_id in enumerate(case_names):
#         pred_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
#         gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")

#         pred = nib.load(pred_path).get_fdata().astype(np.uint8)
#         gt = nib.load(gt_path).get_fdata().astype(np.uint8)

#         pred_mask = np.isin(pred, label_values)
#         gt_mask = np.isin(gt, label_values)

#         pred_slice = middle_slice(pred_mask)
#         gt_slice = middle_slice(gt_mask)
#         overlay = get_overlay(gt_slice, pred_slice)

#         # Plot GT, Prediction, Overlay
#         axes[i, 0].imshow(gt_slice, cmap='gray')
#         axes[i, 0].set_title(f"{case_id} - GT ({label_name})")
#         axes[i, 1].imshow(pred_slice, cmap='gray')
#         axes[i, 1].set_title(f"Prediction ({label_name})")
#         axes[i, 2].imshow(overlay)
#         axes[i, 2].set_title("Overlay (G=TP, R=FN, B=FP)")

#         for ax in axes[i]:
#             ax.axis("off")

#     plt.tight_layout()
#     out_file = f"brats_comparison_{label_name}.png"
#     plt.savefig(out_file, dpi=150)
#     print(f"Saved {out_file}")
#     plt.close()
