import os

def sync_directories():
    """
    Synchronizes specific zed_* directories by deleting files in the source folders
    that do not have a corresponding result file in the grounding_result folders.
    This script specifically targets folders from zed_0079 to zed_0089 and
    preserves 'obj_num.txt' files.
    """
    base_dir = "/data/myp/grasp/ori/toy_photo/mask_photo"
    result_base_dir = os.path.join(base_dir, "grounding_result")

    print("--- Starting Directory Synchronization and Cleanup for zed_0079 to zed_0089 ---")

    # Define the range of folders to process
    start_index = 3
    end_index = 55
    target_folders = [f"zed_{i:04d}" for i in range(start_index, end_index + 1)]

    for folder_name in sorted(target_folders):
        source_dir = os.path.join(base_dir, folder_name)
        result_dir = os.path.join(result_base_dir, folder_name)

        print(f"\nProcessing folder: {folder_name}")

        if not os.path.isdir(source_dir):
            print(f"  Info: Source directory not found. Skipping '{folder_name}'.")
            continue
        
        if not os.path.isdir(result_dir):
            print(f"  Warning: Corresponding result directory not found. Skipping '{folder_name}'.")
            continue

        # --- Step 1: Get the set of base filenames to keep ---
        # These are determined by the .jpg files in the result directory
        files_to_keep = set()
        for f in os.listdir(result_dir):
            if f.endswith('.jpg') and not f.endswith('_mask_visual.jpg'):
                base_name = f.replace('.jpg', '')
                files_to_keep.add(base_name)
        
        if not files_to_keep:
            print(f"  Info: No result files found in '{result_dir}'. No files will be deleted from source.")
            continue
            
        print(f"  Found {len(files_to_keep)} result files to use as reference.")

        # --- Step 2: Iterate through the source directory and delete unnecessary files ---
        files_deleted_count = 0
        for f in os.listdir(source_dir):
            # --- Condition to preserve obj_num.txt ---
            if f == 'obj_num.txt':
                continue

            source_file_path = os.path.join(source_dir, f)
            
            # Get the base name of the source file, with special handling for label files
            file_base, file_ext = os.path.splitext(f)
            
            # If it's a label file, remove 'Label' from the base name for comparison
            if file_ext == '.txt' and 'Label' in file_base:
                comparison_base = file_base.replace('Label', '')
            else:
                comparison_base = file_base

            # Check if the processed base name is in our set of files to keep
            if comparison_base not in files_to_keep:
                try:
                    os.remove(source_file_path)
                    print(f"  - Deleted: {f}")
                    files_deleted_count += 1
                except OSError as e:
                    print(f"  - Error deleting {f}: {e}")

        if files_deleted_count > 0:
            print(f"  Finished cleanup for {folder_name}. Deleted {files_deleted_count} file(s).")
        else:
            print(f"  No files needed to be deleted in {folder_name}.")
            
    print("\n--- Synchronization and Cleanup Complete ---")


if __name__ == "__main__":
    # Add a confirmation step to prevent accidental deletion
    confirm = input("This script will delete files from the '/data/myp/grasp/ori/toy_photo/unmask_photo/zed_0079' to 'zed_0089' directories.\nIt will NOT delete 'obj_num.txt' files.\nAre you sure you want to continue? (yes/no): ")
    if confirm.lower() == 'yes':
        sync_directories()
    else:
        print("Operation cancelled.")