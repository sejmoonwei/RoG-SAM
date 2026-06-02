import cv2
import numpy as np
import os
from scipy import ndimage

def visualize_mask_on_image(image_path, mask_path, output_directory):
    """
    Visualizes mask labels on an image, prints the mask value on each mask,
    and saves the result.

    Args:
        image_path (str): The path to the original image.
        mask_path (str): The path to the labeled mask file.
        output_directory (str): The directory to save the analyzed image.
    """
    # Read the original image and the mask image
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not read image at '{image_path}'")
        return
    if mask is None:
        print(f"Error: Could not read mask at '{mask_path}'")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get unique non-zero mask values
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]

    # Create a color overlay for the masks
    overlay = image.copy()
    
    # Generate a random color for each label
    colors = {label: np.random.randint(0, 255, size=3).tolist() for label in unique_labels}

    for label in unique_labels:
        # Create a binary mask for the current label
        label_mask = np.uint8(mask == label) * 255
        
        # Find contours of the mask
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Apply color to the mask area on the overlay
        overlay[mask == label] = colors[label]

        for contour in contours:
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # If moment is zero, use the average of the contour points
                cX, cY = np.mean(contour.squeeze(), axis=0).astype(int)

            # Put the label number on the image
            cv2.putText(overlay, str(label), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Blend the overlay with the original image
    alpha = 0.6
    result_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Save the final image
    output_filename = os.path.basename(image_path).replace('.png', '_analyzed.png')
    output_path = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_path, result_image)
    print(f"Successfully saved analyzed image to: {output_path}")


if __name__ == "__main__":
    # Directory containing the source images
    target_directory = "/data/myp/grasp/ori/toy_photo/unmask_photo/zed_0004"
    
    # Directory to save the output
    output_dir = "/data/myp/grasp/ori/toy_photo/analyze"

    try:
        files = os.listdir(target_directory)
    except FileNotFoundError:
        print(f"Error: Directory not found at '{target_directory}'")
        files = []

    mask_files = [f for f in files if f.endswith('_labeled_mask.png')]

    if not mask_files:
        print(f"No '_labeled_mask.png' files found in '{target_directory}'.")
    else:
        print(f"Found {len(mask_files)} mask files to process.")

    for mask_name in sorted(mask_files):
        image_name = mask_name.replace('_labeled_mask.png', '.png')
        
        image_file_path = os.path.join(target_directory, image_name)
        mask_file_path = os.path.join(target_directory, mask_name)

        if not os.path.exists(image_file_path):
            print(f"Warning: Corresponding image '{image_name}' not found for mask '{mask_name}'. Skipping.")
            continue

        print(f"\nProcessing: {image_name}")
        visualize_mask_on_image(image_file_path, mask_file_path, output_dir)