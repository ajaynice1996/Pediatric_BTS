import os
import numpy as np

plans_folder = "nnUNet_preprocessed/Dataset202_BraTS/nnUNetPlans_3d_fullres/"
npz_files = [f for f in os.listdir(plans_folder) if f.endswith(".npz")]

for f in sorted(npz_files[0:20]):
    case_id = f.replace(".npz", "")
    npz_path = os.path.join(plans_folder, f)
    npz = np.load(npz_path, allow_pickle=True)

    print(f"Case: {case_id}")
    print(f"  Keys in npz: {list(npz.keys())}")

    if 'data' in npz:
        data = npz['data']
        print(f"  Preprocessed data shape (C,H,W,D): {data.shape}")
        print(f"  Number of modalities: {data.shape[0]}")
    else:
        print("  No 'data' key found.")

    if 'seg' in npz:
        seg = npz['seg']
        print(f"  Segmentation shape: {seg.shape}")
    else:
        print("  No 'seg' key found.")

    if 'properties' in npz:
        props = npz['properties'].item()
        print(f"  Original shape: {props.get('original_shape','N/A')}")
        print(f"  Spacing: {props.get('spacing','N/A')}")
    else:
        print("  No 'properties' key in this npz.\n")