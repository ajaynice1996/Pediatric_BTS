import re
import os
import json
import torch
import shutil
import random
torch.__version__

import os
import numpy as np

os.environ["nnUNet_raw_data_base"] = "data/nnUNet_raw_data_base"
os.environ["RESULTS_FOLDER"] = "data/nnUNet_results"
os.environ["nnUNet_preprocessed"] = "data/nnUNet_preprocessed"

os.makedirs('data/nnUNet_raw_data_base/Dataset202_BraTS/imagesTr', exist_ok=True)
os.makedirs('data/nnUNet_raw_data_base/Dataset202_BraTS/labelsTr', exist_ok=True)

import os
import shutil

# ↳Dataset2023
#   ↳imagesTr
#     BraTS_0001_0000.nii.gz
#     BraTS_0001_0001.nii.gz
#     BraTS_0001_0002.nii.gz
#     BraTS_0001_0003.nii.gz
#   ↳labelsTr
#     BraTS_0001.nii.gz

# Moving Brain MRI Modalities to nnUNet_raw/Dataset2023/imagesTr
# Moving Segmentation Masks to nnUNet_raw/Dataset2023/labelsTr

import os

# Base directory containing folders like BraTS-PED-00224-000
base_path = '../raw data subset/Training'  # replace with your actual path
labels_dir = 'data/nnUNet_raw_data_base/Dataset202_BraTS/labelsTr'
images_dir = 'data/nnUNet_raw_data_base/Dataset202_BraTS/imagesTr'

# Paths and setup variables
output_file = 'data/nnUNet_raw_data_base/Dataset202_BraTS/dataset.json'
imagesTr = images_dir
imagesTs = 'data/nnUNet_raw_data_base/Dataset202_BraTS/imagesTs'
labelsTr = labels_dir

# Read all directories
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

print(f"Total folders found: {len(folders)}")
# Extract sorting key based on the second numeric component
def extract_key(folder_name):
    parts = folder_name.split('-')
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    elif len(parts) >= 3 and parts[1].isdigit():
        return int(parts[1])
    else:
        return float('inf')
case_id = 0
# Sort the folder list
sorted_folders = sorted(folders, key=extract_key)

import os
import shutil

# Loop through sorted folders
for folder in sorted_folders:
    parts = folder.split('-')
    if len(parts) >= 3:
        case_id = int(parts[2])  # Convert to integer for sorting
        print(f"Folder: {folder}, Case ID: {case_id}")
    else:
        print(f"Skipping malformed folder name: {folder}")
        continue

    folder_path = os.path.join(base_path, folder)

    for file in os.listdir(folder_path):
        print(f"Processing file: {file}")

        if 'seg' in file:
            shutil.copy(os.path.join(folder_path, file), labels_dir)
            os.rename(os.path.join(labels_dir, file),
                      os.path.join(labels_dir, f"BraTSPED_{case_id:04d}.nii.gz"))

        if 't1n' in file or 't1.nii' in file:
            shutil.copy(os.path.join(folder_path, file), images_dir)
            os.rename(os.path.join(images_dir, file),
                      os.path.join(images_dir, f"BraTSPED_{case_id:04d}_0000.nii.gz"))
            
        if 't1c' in file or 't1ce' in file:
            shutil.copy(os.path.join(folder_path, file), images_dir)
            os.rename(os.path.join(images_dir, file),
                      os.path.join(images_dir, f"BraTSPED_{case_id:04d}_0001.nii.gz"))


        if 't2f' in file or 'flair' in file:
            shutil.copy(os.path.join(folder_path, file), images_dir)
            os.rename(os.path.join(images_dir, file),
                      os.path.join(images_dir, f"BraTSPED_{case_id:04d}_0002.nii.gz"))

        if 't2w' in file or 't2.nii' in file:
            shutil.copy(os.path.join(folder_path, file), images_dir)
            os.rename(os.path.join(images_dir, file),
                      os.path.join(images_dir, f"BraTSPED_{case_id:04d}_0003.nii.gz"))


# Renaming imagesTr Files to nnU-Net Format

# def rename_file(filename, patient_counters):
#     pattern = re.compile(r'BraTS-SSA-(\d+)-(\d+)-(\d+)\.nii\.gz')
#     print(f"Processing : {pattern}")

#     match = pattern.match(filename)
#     if match:
#         patient_id = match.group(1)
#         modality_part = match.group(3)

#         if patient_id not in patient_counters:
#             patient_counters[patient_id] = len(patient_counters)
            
#         series_part = str(patient_counters[patient_id]).zfill(4)

#         new_filename = f'BraTS{patient_id}_{series_part}_{modality_part}.nii.gz'
#         return new_filename
#     return None

# patient_counters = {}
# file_list = sorted(os.listdir('data/nnUNet_raw_data_base/Task2023_BraTS/imagesTr/'))
# for filename in file_list:
#     new_filename = rename_file(filename, patient_counters)
#     print(f"Renaming {filename} to {new_filename}")
#     if new_filename:
#         os.rename(f'data/nnUNet_raw_data_base/Task2023_BraTS/imagesTr/{filename}', f'data/nnUNet_raw_data_base/Task2023_BraTS/imagesTr/{new_filename}')


import json
from typing import Tuple

# for nnUNetV2
tranining_len = len(os.listdir('data/nnUNet_raw_data_base/Dataset202_BraTS/labelsTr'))
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

with open(f'{output_file}', 'w') as file:
    json.dump(data, file, indent=4)


# # -----------------
# # for medNext; not working ofr NNUnetV2
# # modalities = ("t1c", "t1n", "t2f", "t2w")
# # labels = {0 : "background", 1 : "ET", 2 : "WT", 3 : "ET"}

# channel_names = {
#         0: "t1n",
#         1: "t1c",
#         2: "t2f",
#         3: "t2w"
#     }

# labels = {
#         0 : "background",
#         1 : "ET",
#         2 : "NET",
#         3 : "CC",
#         4 : "ED",
#     }
    
# # labels = {
# #     "background" : 0,
# #     "ET" : 1,
# #     "NET" : 2,
# #     "CC" : 3,
# #     "ED" : 4,
# # }

# replacements = {
#     't1n':'_0000',
#     't2f':'_0002',
#     't2w':'_0003',
#     'seg':'',
#     '-PED':'PED',
#     '-000-':'',
#     '-' : '_',
#     }

# filename = 'dataset.json'
# dataset_name = 'Dataset202_BraTS'

# def save_json(obj, file, sort_keys=True):
#     with open(file, 'w') as f:
#         json.dump(obj, f, indent=4, sort_keys=sort_keys)

# def subfiles(folder, suffix, join=True):
#     all_files = []
#     for root, _, files in os.walk(folder):
#         for file in files:
#             if file.endswith(suffix):
#                 if join:
#                     all_files.append(os.path.join(root, file))
#                 else:
#                     all_files.append(file)
#     return all_files

# def get_identifiers_from_splitted_files(folder: str):
#     uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
#     return uniques

# def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: tuple,
#                           labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!", dataset_description: str = "",
#                           dataset_reference="", dataset_release='0.0'):
#     train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

#     if imagesTs_dir is not None:
#         test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
#     else:
#         test_identifiers = []

#     json_dict = {}
#     json_dict['name'] = dataset_name
#     # json_dict['description'] = dataset_description
#     json_dict['tensorImageSize'] = "4D"
#     # json_dict['reference'] = dataset_reference
#     # json_dict['licence'] = license
#     # json_dict['release'] = dataset_release
#     json_dict['channel_names'] = {str(i): channel_names[i] for i in range(len(channel_names))}
#     json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

#     json_dict['numTraining'] = len(train_identifiers)
#     json_dict['file_ending'] = ".nii.gz"
#     json_dict['numTest'] = len(test_identifiers)
#     json_dict['training'] = [
#         {'image': f"./imagesTr/{i}.nii.gz", "label": f"./labelsTr/{i}.nii.gz"} for i in train_identifiers]
#     json_dict['test'] = [f"./imagesTs/{i}.nii.gz" for i in test_identifiers]

#     if not output_file.endswith("dataset.json"):
#         print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
#               "Proceeding anyways...")
#     save_json(json_dict, output_file, sort_keys=sort_keys)


# generate_dataset_json(output_file=output_file, 
#                       imagesTr_dir=imagesTr, 
#                       imagesTs_dir=imagesTs,
#                       modalities=channel_names,
#                       labels=labels, 
#                       dataset_name=dataset_name)