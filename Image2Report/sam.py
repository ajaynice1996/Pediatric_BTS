import torch
import os

sam_checkpoint = "sam_vit_b_01ec64.pth"
sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not os.path.exists(sam_checkpoint):
    torch.hub.download_url_to_file(sam_url, sam_checkpoint)

from segment_anything import sam_model_registry, SamPredictor

model_type = 'vit_b'
checkpoint_path = 'sam_vit_b_01ec64.pth'

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)

import nibabel as nib
import numpy as np

nii_file_path = "/drive/MyDrive/auto-rapno/data/raw/Brats_ped_2024_training/BraTS-PED-00001-000/BraTS-PED-00001-000-t2w.nii.gz"

# Load the NIfTI image
nii_img = nib.load(nii_file_path)

# Get the image data as a NumPy array
img_data = nii_img.get_fdata()

# Select a middle slice along the z-axis (assuming z is the last dimension)
slice_index = img_data.shape[-1] // 2
image_slice = img_data[..., slice_index]

print(f"Shape of the loaded image data: {img_data.shape}")
print(f"Shape of the extracted slice: {image_slice.shape}")

# Convert the image slice to uint8
image_slice_uint8 = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice)) * 255
image_slice_uint8 = image_slice_uint8.astype(np.uint8)

# Ensure the image has 3 channels
if len(image_slice_uint8.shape) == 2:
    image_slice_uint8 = np.stack([image_slice_uint8] * 3, axis=-1)

# Prepare the image for SAM prediction
predictor.set_image(image_slice_uint8)

print(f"Shape of the preprocessed image slice: {image_slice_uint8.shape}")

# Get the image dimensions
height, width, _ = image_slice_uint8.shape

# Define a point prompt in the center of the image
input_point = np.array([[width // 4, height // 4]])
input_label = np.array([1]) # 1 indicates a foreground point

# Predict segmentation
masks, iou_scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False, # Set to True to get multiple masks
)

# Print the shape of the masks
print(f"Shape of the returned masks: {masks.shape}")

import matplotlib.pyplot as plt

# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image slice
axes[0].imshow(image_slice, cmap='gray')
axes[0].set_title('Original Image Slice')
axes[0].axis('off')

# Display the segmentation mask
axes[1].imshow(masks[0], cmap='gray') # masks is a list of masks, so we take the first one
axes[1].set_title('Segmentation Mask')
axes[1].axis('off')

# Adjust layout and display the figure
plt.tight_layout()
plt.show()