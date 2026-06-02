import cv2
import numpy as np
import os
import json

# --- Configuration ---
# Define the start and end index for the zed folders you want to process.
START_FOLDER_IDX = 5
END_FOLDER_IDX = 95
# ---------------------

# --- Paths ---
BASE_DIR = "/data/myp/grasp/ori/toy_photo/unmask_photo"
# -------------

# Global variables to store state between mouse clicks
g_current_mask = None
g_new_labeled_mask = None
g_original_mask_value = -1
g_object_count_total = 0
g_object_count_labeled = 0
g_window_name = "Annotation Window"


def on_mouse(event, x, y, flags, param):
    """Mouse callback function to handle user clicks. No coordinate transformation needed."""
    global g_current_mask, g_new_labeled_mask, g_original_mask_value, g_object_count_labeled

    if event == cv2.EVENT_LBUTTONDOWN:
        if g_current_mask is None:
            return

        # Directly use the (x, y) coordinates as the window size is now fixed
        g_original_mask_value = g_current_mask[y, x]

        if g_original_mask_value == 0:
            print("Info: You clicked on the background. Please click on an object mask.")
            return

        print(f"\nClicked on an object with original mask value: {g_original_mask_value}")
        
        while True:
            try:
                new_label_str = input("Enter the new label for this object (1-13): ")
                new_label = int(new_label_str)
                if 1 <= new_label <= 13:
                    break
                else:
                    print("Error: Label must be between 1 and 13.")
            except ValueError:
                print("Error: Please enter a valid integer.")
        
        g_new_labeled_mask[g_current_mask == g_original_mask_value] = new_label
        print(f"Object relabeled from {g_original_mask_value} to {new_label}.")
        
        g_object_count_labeled += 1
        
        if g_object_count_labeled < g_object_count_total:
            remaining = g_object_count_total - g_object_count_labeled
            print(f"Success! {remaining} object(s) remaining to be labeled in this image.")
        else:
            print("\nAll objects in this image are labeled!")
            print("Press SPACE to save and move to the next image, or ESC to exit.")


def create_visual_overlay(image, mask):
    """Creates a colored overlay of the mask on the original image for visualization."""
    # Generate a color map for the mask values (1, 2, 3...)
    color_map = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [0, 255, 255], [255, 0, 255], [192, 192, 192], [128, 128, 128],
        [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128]
    ], dtype=np.uint8)

    color_mask = np.zeros_like(image)
    unique_values = np.unique(mask)
    for val in unique_values:
        if val == 0 or val >= len(color_map): continue
        color_mask[mask == val] = color_map[val]

    overlay = cv2.addWeighted(image, 0.6, color_mask, 0.4, 0)
    return overlay


def run_annotator():
    """Main function to run the annotation tool."""
    global g_current_mask, g_new_labeled_mask, g_object_count_total, g_object_count_labeled

    print("--- Interactive Mask Annotation Tool (v3) ---")
    print("Instructions:")
    print(" - The window shows the original image.")
    print(" - Press 'v' to toggle the mask overlay view on and off.")
    print(" - Click on an object area (in either view).")
    print(" - Enter its new label (1-13) in the terminal.")
    print(" - Press SPACE to save and process the next image.")
    print(" - Press ESC to quit at any time.")
    print("-" * 30)

    cv2.namedWindow(g_window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(g_window_name, on_mouse)

    for i in range(START_FOLDER_IDX, END_FOLDER_IDX + 1):
        dir_num = f"{i:04d}"
        dir_name = f"zed_{dir_num}"
        
        source_dir = os.path.join(BASE_DIR, dir_name)
        result_dir = os.path.join(BASE_DIR, "grounding_result", dir_name)
        
        if not os.path.isdir(result_dir):
            print(f"Warning: Result directory not found, skipping: {result_dir}")
            continue

        # Iterate based on files in grounding_result to ensure consistency
        for f_name in sorted(os.listdir(result_dir)):
            if f_name.endswith('_mask.png'):
                base_name = f_name.replace('_mask.png', '')
                
                # Define paths
                original_img_path = os.path.join(source_dir, f"{base_name}.png")
                mask_path = os.path.join(result_dir, f_name)
                obj_num_path = os.path.join(source_dir, "obj_num.txt")
                output_mask_path = os.path.join(source_dir, f"{base_name}_labeled_mask.png")

                # --- File existence checks ---
                if os.path.exists(output_mask_path):
                    print(f"Info: Already labeled. Skipping {base_name}")
                    continue
                if not os.path.exists(original_img_path):
                    print(f"Error: Original image not found for {base_name}. Skipping. Path: {original_img_path}")
                    continue
                if not os.path.exists(obj_num_path):
                    print(f"Error: obj_num.txt not found in {source_dir}. Skipping image.")
                    continue

                # --- Load data ---
                original_image = cv2.imread(original_img_path)
                g_current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                with open(obj_num_path, 'r') as f:
                    g_object_count_total = int(f.read().strip())

                if original_image is None or g_current_mask is None:
                    print(f"Error loading files for {base_name}. Skipping.")
                    continue
                
                # --- Reset state for the new image ---
                g_new_labeled_mask = np.zeros_like(g_current_mask)
                g_object_count_labeled = 0
                
                print(f"\nNow annotating: {base_name}.png")
                print(f"This image contains {g_object_count_total} object(s). Please label them all.")

                while g_object_count_labeled < g_object_count_total:
                    # By default, show the original image
                    display_image = original_image
                
                    cv2.imshow(g_window_name, display_image)
                    key = cv2.waitKey(0) & 0xFF
                    
                    if key == 27: # ESC key
                        print("Exiting.")
                        cv2.destroyAllWindows()
                        return
                    elif key == 32: # SPACE key
                        break
                
                if g_object_count_labeled == g_object_count_total:
                    cv2.imwrite(output_mask_path, g_new_labeled_mask)
                    print(f"Successfully saved new mask to {output_mask_path}")
                else:
                    print(f"Warning: Not all objects were labeled for {base_name}. Mask not saved.")

    print("\nAnnotation complete for all specified folders.")
    cv2.destroyAllWindows()


    print("\nAnnotation complete for all specified folders.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_annotator()