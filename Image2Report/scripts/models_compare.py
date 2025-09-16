import os
import nibabel as nib

# Folder containing GT segmentations
gt_path = "nnUNet_results/Dataset202_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation"

# List all nii.gz files
files = [f for f in os.listdir(gt_path) if f.endswith(".nii.gz")]

print(f"📂 Found {len(files)} ground truth files.\n")

for f in sorted(files):
    case_id = f.replace(".nii.gz", "")
    nii = nib.load(os.path.join(gt_path, f))
    data = nii.get_fdata()
    print(f"{case_id}: shape = {data.shape}")
