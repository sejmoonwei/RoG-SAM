import os
import cv2
import numpy as np
import random

def generate_obj_num_files():
    """
    Analyzes one mask file from each result folder to determine the number of objects
    and writes this count to an 'obj_num.txt' file in the corresponding source folder.
    """
    base_dir = "/data/myp/grasp/ori/toy_photo/unmask_photo"
    result_base_dir = os.path.join(base_dir, "grounding_result")

    print("--- Starting Generation of obj_num.txt files ---")

    if not os.path.isdir(result_base_dir):
        print(f"Error: Result base directory not found at '{result_base_dir}'")
        return

    # Find all zed_* directories in the result directory
    try:
        result_folders = [d for d in os.listdir(result_base_dir) if d.startswith('zed_') and os.path.isdir(os.path.join(result_base_dir, d))]
    except FileNotFoundError:
        print(f"Error: Result directory not found at '{result_base_dir}'")
        return

    for folder_name in sorted(result_folders):
        result_dir = os.path.join(result_base_dir, folder_name)
        source_dir = os.path.join(base_dir, folder_name)

        print(f"\nProcessing folder: {folder_name}")

        if not os.path.isdir(source_dir):
            print(f"  Warning: Corresponding source directory not found. Creating '{source_dir}'.")
            os.makedirs(source_dir)

        # Find all '_mask.png' files in the current result directory
        mask_files = [f for f in os.listdir(result_dir) if f.endswith('_mask.png')]

        if not mask_files:
            print(f"  Warning: No '_mask.png' files found in '{result_dir}'. Skipping.")
            continue

        # --- Pick one random mask file to analyze ---
        chosen_mask_file = random.choice(mask_files)
        mask_path = os.path.join(result_dir, chosen_mask_file)
        
        print(f"  Analyzing sample mask: {chosen_mask_file}")

        # Read the mask image
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask_image is None:
            print(f"  Error: Could not read mask file '{chosen_mask_file}'. Skipping.")
            continue
            
        # Find all unique pixel values
        unique_pixels = np.unique(mask_image)
        
        # Count non-zero pixel values to get the number of objects
        object_count = len(unique_pixels[unique_pixels != 0])
        
        # --- Write the count to obj_num.txt in the source directory ---
        obj_num_path = os.path.join(source_dir, 'obj_num.txt')
        try:
            with open(obj_num_path, 'w') as f:
                f.write(str(object_count))
            print(f"  Success: Wrote object count ({object_count}) to '{obj_num_path}'")
        except IOError as e:
            print(f"  Error: Could not write to '{obj_num_path}'. Reason: {e}")

    print("\n--- Generation of obj_num.txt files complete ---")

if __name__ == "__main__":
    generate_obj_num_files()