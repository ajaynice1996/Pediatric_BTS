import re
import os
import json
import torch
import shutil
import random
torch.__version__

import os
import numpy as np

os.environ["nnUNet_raw_data_base"] = "nnUNet_raw_data_base"
os.environ["RESULTS_FOLDER"] = "nnUNet_results"
os.environ["nnUNet_preprocessed"] = "nnUNet_preprocessed"

os.makedirs('nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr', exist_ok=True)
os.makedirs('nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/labelsTr', exist_ok=True)

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
base_path = '../Data/raw/BraTS-PEDs2024_Training/'  # replace with your actual path

# Read all directories
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# # Extract sorting key based on the second numeric component
# def extract_key(folder_name):
#     parts = folder_name.split('-')
#     if len(parts) >= 3 and parts[2].isdigit():
#         return int(parts[2])
#     elif len(parts) >= 3 and parts[1].isdigit():
#         return int(parts[1])
#     else:
#         return float('inf')
# case_id = 0
# # Sort the folder list
# sorted_folders = sorted(folders, key=extract_key)

# # Loop through sorted folders
# for folder in sorted_folders:
#     parts = folder.split('-')
#     if len(parts) >= 3:
#         case_id = parts[2]
#         case_id = int(case_id)  # Convert to integer for sorting
#         print(f"Folder: {folder}, Case ID: {case_id}")
#     else:
#         print(f"Skipping malformed folder name: {folder}")

#     for file in os.listdir(base_path + folder):
#         print(f"Processing file: {file}")
#         if 'seg' in file:
#             shutil.copy(f'{base_path}{folder}/{file}', 'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/labelsTr')
#             os.rename(f'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/labelsTr/{file}', f"nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/labelsTr/BraTSPED_{case_id:04d}.nii.gz")
#         if 't1c' in file or 't1ce' in file:
#             shutil.copy(f'{base_path}{folder}/{file}', 'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr')
#             os.rename(f'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/{file}', f"nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/BraTSPED_{case_id:04d}_0001.nii.gz")
#         if 't1n' in file or 't1.nii' in file:
#             shutil.copy(f'{base_path}{folder}/{file}', 'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr')
#             os.rename(f'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/{file}', f"nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/BraTSPED_{case_id:04d}_0000.nii.gz")
#         if 't2f' in file or 'flair' in file:
#             shutil.copy(f'{base_path}{folder}/{file}', 'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr')
#             os.rename(f'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/{file}', f"nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/BraTSPED_{case_id:04d}_0002.nii.gz")
#         if 't2w' in file or 't2.nii' in file:
#             shutil.copy(f'{base_path}{folder}/{file}', 'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr')
#             os.rename(f'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/{file}', f"nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/BraTSPED_{case_id:04d}_0003.nii.gz")


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
# file_list = sorted(os.listdir('nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/'))
# for filename in file_list:
#     new_filename = rename_file(filename, patient_counters)
#     print(f"Renaming {filename} to {new_filename}")
#     # if new_filename:
#     #     os.rename(f'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/{filename}', f'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr/{new_filename}')


# import os
# import json
# import numpy as np
# from typing import Tuple

# Paths and setup variables
output_file = 'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/dataset.json'
imagesTr = 'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTr'
imagesTs = 'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/imagesTs'
labelsTr = 'nnUNet_raw_data_base/nnUNet_raw_data/Task2023_BraTS/labelsTr'

# modalities = ("t1c", "t1n", "t2f", "t2w")
# labels = {0 : "background", 1 : "ET", 2 : "WT", 3 : "ET"}

modalities= (
        "1-t1c",
        "0-t1n",
        "2-t2f",
        "3-t2w"
        )

labels = {
        0 : "background",
        1 : "ET",
        2 : "NET",
        3 : "CC",
        4 : "ED",
    }

replacements = {
    't1n':'_0000',
    't2f':'_0002',
    't2w':'_0003',
    'seg':'',
    '-PED':'PED',
    '-000-':'',
    '-' : '_',
    }

# data = {
#     "modalities": {
#         "0": "t1c",
#         "1": "t1n",
#         "2": "t2f",
#         "3": "t2w"
#     },
    
#     "labels": {
#         "background" : 0,
#         "ET" : 1,
#         "NET" : 2,
#         "CC" : 3,
#         "ED" : 4,
#     },

#     "replacements" : {
#     't1n':'_0000',
#     't1c':'_0001',
#     't2f':'_0002',
#     't2w':'_0003',
#     'seg':'',
#     '-PED':'PED',
#     '-000-':'',
#     '-' : '_',
#     },
#     "numTraining": tranining_len,
#     "file_ending": ".nii.gz"
# }

filename = 'dataset.json'
dataset_name = 'Task2023_BraTS'

def save_json(obj, file, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=sort_keys)

def subfiles(folder, suffix, join=True):
    all_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(suffix):
                if join:
                    all_files.append(os.path.join(root, file))
                else:
                    all_files.append(file)
    return all_files

def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': f"./imagesTr/{i}.nii.gz", "label": f"./labelsTr/{i}.nii.gz"} for i in train_identifiers]
    json_dict['test'] = [f"./imagesTs/{i}.nii.gz" for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, output_file, sort_keys=sort_keys)


generate_dataset_json(output_file=output_file, 
                      imagesTr_dir=imagesTr, 
                      imagesTs_dir=imagesTs,
                      modalities=modalities,
                      labels=labels, 
                      dataset_name=dataset_name)