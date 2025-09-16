import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

# -------------------------------
# Paths
# -------------------------------
pred_dir = "/nnUNet_results/Dataset202_BraTS/nnUNetTrainerUMambaBot_100epochs__nnUNetPlans__3d_fullres/fold_2/validation"
gt_dir = "/nnUNet_preprocessed/Dataset202_BraTS/gt_segmentations"

# -------------------------------
# Load volumes
# -------------------------------
gt_vol = nib.load(gt_path).get_fdata().astype(int)

# Load prediction: handle both .npz and .nii.gz
if pred_path.endswith(".npz"):
    pred_npz = np.load(pred_path, allow_pickle=True)
    pred_vol = pred_npz['seg']
elif pred_path.endswith(".nii.gz"):
    pred_vol = nib.load(pred_path).get_fdata().astype(int)
else:
    raise ValueError("Prediction file must be .nii.gz or .npz")

# -------------------------------
# Define colormap for 4 labels
# -------------------------------
colors = ['black', 'red', 'green', 'blue']  # label 0-3
cmap = ListedColormap(colors)

# -------------------------------
# Setup figure and slider
# -------------------------------
slice_idx = gt_vol.shape[2] // 2
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

img_gt = axes[0].imshow(gt_vol[:, :, slice_idx], cmap=cmap, vmin=0, vmax=3)
axes[0].set_title("Ground Truth")
axes[0].axis('off')

img_pred = axes[1].imshow(pred_vol[:, :, slice_idx], cmap=cmap, vmin=0, vmax=3)
axes[1].set_title("Prediction")
axes[1].axis('off')

# Slider axis
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 0, gt_vol.shape[2]-1, valinit=slice_idx, valstep=1)

# -------------------------------
# Update function
# -------------------------------
def update(val):
    idx = int(slider.val)
    img_gt.set_data(gt_vol[:, :, idx])
    img_pred.set_data(pred_vol[:, :, idx])
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()