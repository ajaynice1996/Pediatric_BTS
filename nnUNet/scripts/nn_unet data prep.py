import os
import json
import gzip
import shutil

import nibabel
import numpy as np
import matplotlib.pyplot as plt
# Create directories if they don't exist
os.makedirs('Data/nnUNet_raw', exist_ok=True)  # Stores Raw Training Data
os.makedirs('Data/nnUNet_preprocessed', exist_ok=True)  # Stores nnU-Net Preprocessed Data
os.makedirs('Data/nnUNet_results', exist_ok=True)  # Stores nnU-Net Results

# Folder Structure of nnUNet_raw
# nnUNet_raw
# ↳Dataset2024
#   ↳imagesTr
#   ↳labelsTr

os.makedirs('Data/nnUNet_raw/Dataset2024/imagesTr', exist_ok=True)
os.makedirs('Data/nnUNet_raw/Dataset2024/labelsTr', exist_ok=True)



# Creating JSON file for nnU-Net Needed For Preprocessing
# This is self-descriptive enough
tranining_len = len(os.listdir('Data/raw/BraTS-PEDs2024_Training'))
print(f"Number of training subjects: {tranining_len}")


data = {
    "channel_names": {
        "0": "t1n",
        "1": "t1c",
        "2": "t2f",
        "3": "t2w"
    },
    
    "labels": {
        "background" : 0,
        "ET" : 1,
        "NET" : 2,
        "CC" : 3,
        "ED" : 4,
    },

    "replacements" : {
    't1n':'_0000',
    't1c':'_0001',
    't2f':'_0002',
    't2w':'_0003',
    'seg':'',
    '-PED':'PED',
    '-000-':'',
    '-' : '_',
    },
    "numTraining": tranining_len,
    "file_ending": ".nii.gz"
}

filename = 'dataset.json'

with open(f'Data/nnUNet_raw/Dataset2024/{filename}', 'w') as file:
    json.dump(data, file, indent=4)

# Moving Brain MRI Modalities to nnUNet_raw/Dataset2023/imagesTr
# Moving Segmentation Masks to nnUNet_raw/Dataset2023/labelsTr

import os
import shutil

imagesTr_path = 'Data/nnUNet_raw/Dataset2024/imagesTr'
labelsTr_path = 'Data/nnUNet_raw/Dataset2024/labelsTr'
nnunet_preprocessed = 'Data/nnUNet_preprocessed/'
nnunet_preprocessed_res = 'Data/nnUNet_preprocessed/'


os.makedirs(imagesTr_path, exist_ok=True)
os.makedirs(labelsTr_path, exist_ok=True)
os.makedirs(nnunet_preprocessed, exist_ok=True)
os.makedirs(nnunet_preprocessed_res, exist_ok=True)

replacements = {
    't1n':'_0000',
    't1c':'_0001',
    't2f':'_0002',
    't2w':'_0003',
    'seg':'',
    '-000-':'',
    '-PED' : 'PED',
    '-' : '_',
}

def replace_func(original_name):
    result = original_name
    for old, new in replacements.items():
        print(f"Replacing {old} with {new} in {result}")
        result = result.replace(old, new)
    return result

def prepare_brats_nnunet(data_dir: str, imagesTr: str, labelsTr: str):

    print(f"Preparing BraTS nnUNet data from {data_dir}...")
    items = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                items.append((root, file))
    # Sort by filename for deterministic order
    items.sort(key=lambda x: x[1])
    print(items)
    for root, item in items:
        new_name = replace_func(item)
        print(f"Copying {item} to {new_name}")
        # Place your copy logic here
        # Example:
        if 't1n' in item:
            shutil.copy(f'{root}/{item}', f'{imagesTr}/{new_name}')
        elif 't1c' in item:
            shutil.copy(f'{root}/{item}', f'{imagesTr}/{new_name}')
        elif 't2f' in item:
            shutil.copy(f'{root}/{item}', f'{imagesTr}/{new_name}')
        elif 't2w' in item:
            shutil.copy(f'{root}/{item}', f'{imagesTr}/{new_name}')
        elif 'seg' in item:
            shutil.copy(f'{root}/{item}', f'{labelsTr}/{new_name}')

prepare_brats_nnunet(data_dir = "Data/raw/BraTS-PEDs2024_Training", imagesTr=imagesTr_path, labelsTr=labelsTr_path)

# print(f'Images in imagesTr: {len(os.listdir("Data/nnUNet_raw/Dataset2024/imagesTr"))}')
# print(f'Images in labelsTr: {len(os.listdir("Data/nnUNet_raw/Dataset2024/labelsTr"))}')
# imagetr = os.listdir('Data/nnUNet_raw/Dataset2024/imagesTr')
# labelstr = os.listdir('Data/nnUNet_raw/Dataset2024/labelsTr')

# imagetr.sort()
# labelstr.sort()

# print(f"Sorted imagesTr: {imagetr}")
# print(f"Sorted labelsTr: {labelstr}")
# print("Data preparation for nnUNet completed successfully.")
