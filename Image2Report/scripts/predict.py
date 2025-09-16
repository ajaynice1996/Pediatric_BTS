
# === CONFIG ===
import os
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.metrics import jaccard_score
import glob
models = os.listdir("../TriALS/nnUNet_results/Dataset202_BraTS")

# Keep only items that start with "nnUNetTrainer"
model_folders = [item for item in models if item.startswith("nnUNetTrainer")]

print(f"Available models: {model_folders}")

model_name = model_folders[4]  # Choose the first model for evaluation
print(f"Using model: {model_name}")

##########################################################################################################################

pred_dir = f"../TriALS/nnUNet_results/Dataset202_BraTS/{model_name}/fold_2/validation"
gt_dir = "../TriALS/nnUNet_preprocessed/Dataset202_BraTS/gt_segmentations"

import os
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.metrics import jaccard_score

# Subject  ET_Dice  ET_IoU  TC_Dice  TC_IoU  WT_Dice  WT_IoU
# 0  BraTSPED_00003   0.8673  0.7657   0.8412  0.7260   0.8647  0.7617
# 1  BraTSPED_00005   0.8952  0.8102   0.9067  0.8293   0.9067  0.8293
# 2  BraTSPED_00010   0.6753  0.5097   0.9353  0.8784   0.9528  0.9099
# 3  BraTSPED_00017   0.0000  0.0000   0.1147  0.0609   0.4204  0.2661
# 4  BraTSPED_00028   0.8617  0.7570   0.7645  0.6188   0.9294  0.8682

# # Region label logic
# def extract_regions(seg):
#     # print(np.unique(seg))
#     regions = {}
#     regions["ET"] = (seg == 1)          # Enhancing Tumor
#     regions["TC"] = np.isin(seg, [1,2,3])  # Tumor Core (ET + Necrosis)
#     regions["WT"] = seg > 0            # Whole Tumor
#     return regions

# # Dice and IoU
# def compute_metrics(pred, gt):
#     intersection = np.logical_and(pred, gt).sum()
#     union = np.logical_or(pred, gt).sum()
#     dice = (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)
#     iou = intersection / (union + 1e-8)
#     return round(dice, 4), round(iou, 4)

# # Compare all matching cases
# results = []
# for fname in sorted(os.listdir(pred_dir)):
#     if not fname.endswith(".nii.gz"):
#         continue
#     case_id = fname.replace(".nii.gz", "")
#     pred_path = os.path.join(pred_dir, fname)
#     gt_path = os.path.join(gt_dir, fname)
    
#     if not os.path.exists(gt_path):
#         print(f"Missing GT: {gt_path}")
#         continue

#     pred = nib.load(pred_path).get_fdata().astype(np.uint8)
#     gt = nib.load(gt_path).get_fdata().astype(np.uint8)

#     pred_regions = extract_regions(pred)
#     gt_regions = extract_regions(gt)

#     row = {"Subject": case_id}
#     for region in ["ET", "TC", "WT"]:
#         dice, iou = compute_metrics(pred_regions[region], gt_regions[region])
#         row[f"{region}_Dice"] = dice
#         row[f"{region}_IoU"] = iou
#     results.append(row)

# # Save as CSV
# df = pd.DataFrame(results)
# df.to_csv("nnunet_eval_per_case.csv", index=False)
# print(df.head())

# # Display Statistics

# import os
# import nibabel as nib
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# out_dir = "nnunet_seg_vis"

# os.makedirs(out_dir, exist_ok=True)

# # Choose axis: 0 = sagittal, 1 = coronal, 2 = axial
# slice_axis = 2

# def overlay_diff(gt, pred, slice_idx):
#     # Binary mask of each
#     gt_mask = gt > 0
#     pred_mask = pred > 0

#     # Overlay codes
#     tp = np.logical_and(gt_mask, pred_mask)      # Green
#     fp = np.logical_and(pred_mask, ~gt_mask)     # Red
#     fn = np.logical_and(gt_mask, ~pred_mask)     # Blue

#     # Initialize blank RGB image
#     overlay = np.zeros(gt.shape + (3,), dtype=np.uint8)
#     overlay[tp] = [0, 255, 0]     # Green = TP
#     overlay[fp] = [255, 0, 0]     # Red = FP
#     overlay[fn] = [0, 0, 255]     # Blue = FN

#     if slice_axis == 0:
#         return overlay[slice_idx, :, :]
#     elif slice_axis == 1:
#         return overlay[:, slice_idx, :]
#     else:
#         return overlay[:, :, slice_idx]

# # Visualize
# for fname in sorted(os.listdir(pred_dir)):
#     if not fname.endswith(".nii.gz"):
#         continue

#     pred_path = os.path.join(pred_dir, fname)
#     gt_path = os.path.join(gt_dir, fname)
#     if not os.path.exists(gt_path):
#         continue

#     pred = nib.load(pred_path).get_fdata().astype(np.uint8)
#     gt = nib.load(gt_path).get_fdata().astype(np.uint8)

#     # Choose central slice
#     slice_idx = pred.shape[slice_axis] // 2
#     overlay = overlay_diff(gt, pred, slice_idx)

#     plt.figure(figsize=(6, 6))
#     plt.imshow(overlay)
#     plt.title(f"{fname} (slice {slice_idx})")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f"{fname.replace('.nii.gz', '')}_slice{slice_idx}.png"))
#     plt.close()

# print(f"Saved overlays to: {out_dir}")

# #SAmple output:Below is the output of the script that computes Dice and IoU scores for each label and region, and saves the results to a CSV file.
# # Mean Dice per class:
# # label_1    0.570142
# # label_2    0.719676
# # label_3    0.421594
# # label_4    0.601536
# # dtype: float64

# # Mean IoU per class:
# # label_1    0.498540
# # label_2    0.630735
# # label_3    0.387940
# # label_4    0.573058
# # dtype: float64
# import os
# import numpy as np
# import nibabel as nib
# from medpy.metric import binary
# from glob import glob

# # Output metrics
# dice_scores = {}
# iou_scores = {}

# # Helper: Dice and IoU per class
# def compute_metrics(pred, gt, label_id):
#     pred_bin = (pred == label_id).astype(np.uint8)
#     gt_bin = (gt == label_id).astype(np.uint8)

#     if np.sum(pred_bin) == 0 and np.sum(gt_bin) == 0:
#         return 1.0, 1.0  # Perfect match (both empty)

#     try:
#         dice = binary.dc(pred_bin, gt_bin)
#         iou = binary.jc(pred_bin, gt_bin)
#     except:
#         dice, iou = 0.0, 0.0

#     return dice, iou

# # Get list of predicted files
# pred_files = sorted(glob(os.path.join(pred_dir, "*.nii.gz")))
# print(f"Found {len(pred_files)} predicted files.")

# # Loop through all predicted files
# for pred_path in pred_files:
#     base = os.path.basename(pred_path)
#     subject_id = base.replace(".nii.gz", "")
#     print(f"Processing {subject_id}...")
#     gt_path = os.path.join(gt_dir, base)
#     if not os.path.exists(gt_path):
#         print(f"Missing GT for {subject_id}")
#         continue

#     pred = nib.load(pred_path).get_fdata().astype(np.uint8)
#     gt = nib.load(gt_path).get_fdata().astype(np.uint8)

#     # Compute metrics per class (1–3)
#     scores = {}
#     for label in [1, 2, 3, 4]:  # adjust for your dataset
#         dice, iou = compute_metrics(pred, gt, label)
#         scores[f"label_{label}"] = {"Dice": dice, "IoU": iou}

#     dice_scores[subject_id] = {k: v["Dice"] for k, v in scores.items()}
#     iou_scores[subject_id] = {k: v["IoU"] for k, v in scores.items()}

# # Show sample
# import pandas as pd

# df_dice = pd.DataFrame.from_dict(dice_scores, orient="index")
# df_iou = pd.DataFrame.from_dict(iou_scores, orient="index")

# print("Mean Dice per class:")
# print(df_dice.mean())

# print("\nMean IoU per class:")
# print(df_iou.mean())

# # Optional: Save to CSV
# df_dice.to_csv("dice_scores.csv")
# df_iou.to_csv("iou_scores.csv")

# for all statistics
# Subject

# Label1_Dice, Label1_IoU → for ET

# Label2_Dice, Label2_IoU → for Necrosis

# Label3_Dice, Label3_IoU → for Edema

# TC_Dice, TC_IoU → all tumor classes (core)

# WT_Dice, WT_IoU → whole tumor (any label > 0)
# ✅ Subject-wise Dice scores saved as 'nnunet_eval_labels_and_regions.csv'
#           Subject  Label1_Dice  Label2_Dice  Label3_Dice  Label4_Dice  TC_Dice  WT_Dice
# 0  BraTSPED_00003       0.8673       0.0065       0.0000       0.5707   0.8369   0.8647
# 1  BraTSPED_00005       0.8952       0.2193       0.1042       0.0000   0.9067   0.9067
# 2  BraTSPED_00010       0.6753       0.9187       0.0000       0.0000   0.9352   0.9528
# 3  BraTSPED_00017       0.0000       0.0697       0.0000       0.4477   0.2662   0.4204
# 4  BraTSPED_00028       0.8617       0.3589       0.8963       0.7931   0.6249   0.9294

# ✅ Mean Dice Scores (per label and region):
#                                                     Model  Label1_Dice  Label2_Dice  Label3_Dice  Label4_Dice  TC_Dice  WT_Dice
# nnUNetTrainerUMambaBot_100epochs__nnUNetPlans__3d_fullres       0.4932       0.7197       0.1908       0.1015   0.8541   0.8863

import os
import numpy as np
import nibabel as nib
import pandas as pd

# === METRIC FUNCTION ===
def compute_metrics(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    return round(dice, 4), round(iou, 4)

# === EVALUATE EACH SUBJECT ===
results = []
for fname in sorted(os.listdir(pred_dir)):
    if not fname.endswith(".nii.gz"):
        continue

    case_id = fname.replace(".nii.gz", "")
    pred_path = os.path.join(pred_dir, fname)
    gt_path = os.path.join(gt_dir, fname)

    if not os.path.exists(gt_path):
        print(f"Missing GT: {gt_path}")
        continue

    pred = nib.load(pred_path).get_fdata().astype(np.uint8)
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)

    row = {"Subject": case_id}

    for label in [1, 2, 3, 4]:
        pred_mask = (pred == label)
        gt_mask = (gt == label)
        dice, _ = compute_metrics(pred_mask, gt_mask)
        row[f"Label{label}_Dice"] = dice

    # Composite Regions
    tc_pred = np.isin(pred, [1, 2, 3])  # Tumor Core
    tc_gt = np.isin(gt, [1, 2, 3, 4])
    dice_tc, _ = compute_metrics(tc_pred, tc_gt)
    row["TC_Dice"] = dice_tc

    wt_pred = pred > 0
    wt_gt = gt > 0
    dice_wt, _ = compute_metrics(wt_pred, wt_gt)
    row["WT_Dice"] = dice_wt

    results.append(row)

# === SAVE SUBJECT-WISE RESULTS ===
df = pd.DataFrame(results)
df.to_csv("nnunet_eval_labels_and_regions.csv", index=False)
print("\n✅ Subject-wise Dice scores saved as 'nnunet_eval_labels_and_regions.csv'")
print(df.head())

# === COMPUTE & DISPLAY MEAN DICE SCORES ===
mean_dice_row = df[[col for col in df.columns if col.endswith("_Dice")]].mean().round(4)
mean_dice_df = pd.DataFrame([mean_dice_row])
mean_dice_df.insert(0, "Model", model_name)

print("\n✅ Mean Dice Scores (per label and region):")
print(mean_dice_df.to_string(index=False))

# === SAVE MEAN SUMMARY ===
mean_dice_df.to_csv("nnunet_mean_dice_per_label.csv", index=False)