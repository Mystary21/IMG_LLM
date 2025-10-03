import sys
import os
import time
import traceback
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfFolder

# --- 1. Configuration and Helper Function ---
hf_auth_token = "YOUR HF ACESS TOKEN"
MAX_RETRIES = 3  # Maximum number of retries for a download
RETRY_DELAY = 10 # Seconds to wait between retries

def download_with_retry(download_function, **kwargs):
    """
    A wrapper function that attempts to run a download function with a retry mechanism.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Try to run the provided download function (e.g., hf_hub_download)
            download_function(**kwargs)
            # If it succeeds, exit the loop
            return True
        except Exception as e:
            # Check if the error is a temporary network/DNS issue
            error_str = str(e).lower()
            if "dns error" in error_str or "temporary failure" in error_str or "timeout" in error_str:
                print(f"    [Attempt {attempt + 1}/{MAX_RETRIES}] Network error detected: {e}")
                if attempt + 1 < MAX_RETRIES:
                    print(f"    Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"    [Attempt {attempt + 1}/{MAX_RETRIES}] Max retries reached. Download failed.")
                    return False
            else:
                # If it's another type of error (e.g., file not found, no permission), fail immediately
                print(f"    An unrecoverable error occurred: {e}")
                return False
    return False

# --- 2. Main Execution ---
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <repo_id>")
    sys.exit(1)
repo_id = sys.argv[1].strip()#用來清除前後可能不小心輸入的空格

print(f"--- Analyzing repository: {repo_id} ---")

try:
    api = HfApi()
    all_repo_files = api.list_repo_files(repo_id=repo_id, token=hf_auth_token)
    weight_files = sorted([f for f in all_repo_files if f.endswith(".safetensors")])

    if not weight_files:
        print("No .safetensors files found in this repository. Downloading config files only.")
        files_to_download = []
    else:
        print("\n--- Model Weight Files Menu ---")
        for i, filename in enumerate(weight_files, 1):
            print(f"  [{i}] {filename}")
        print("-----------------------------")

        selected_indices = []
        while True:
            raw_input = input("Please enter the numbers of the files you want... [e.g., 1,3,5]: ")
            if not raw_input:
                selected_indices = []
                break
            try:
                selected_indices = [int(i.strip()) for i in raw_input.split(',')]
                if all(1 <= i <= len(weight_files) for i in selected_indices):
                    break
                else:
                    print("Error: One or more numbers are out of range. Please try again.")
            except ValueError:
                print("Error: Invalid input. Please enter numbers separated by commas.")

        files_to_download = [weight_files[i-1] for i in selected_indices]

    if not files_to_download:
        print("\nNo weight files selected for download.")
    else:
        print("\n--- Order Confirmation ---")
        for filename in files_to_download:
            print(f"  - {filename}")
        print("------------------------")

        confirm_input = input("Proceed with downloading these selected files? [y/n]: ").lower().strip()
        if confirm_input == 'y':
            print("\n--- Starting download of selected files... ---")
            for filename in files_to_download:
                print(f"    Downloading '{filename}'...")

                # Call the download through our new retry wrapper function
                success = download_with_retry(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename=filename,
                    token=hf_auth_token,
                    local_dir_use_symlinks=False,
                )

                if success:
                    print(f"    Successfully downloaded '{filename}'.")
                else:
                    print(f"    Failed to download '{filename}' after multiple retries.")
        else:
            print("Download cancelled by user.")

    print("\n--- Automatically downloading remaining essential config files... ---")
    # Also wrap the final download in the retry function
    success = download_with_retry(
        snapshot_download,
        repo_id=repo_id,
        token=hf_auth_token,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors", "*.bin", "*.ckpt", "*.pt"]
    )

    if success:
        print(f"--- Essential config files for '{repo_id}' are up to date. ---")
    else:
        print(f"--- Failed to download essential config files for '{repo_id}' after multiple retries. ---")

    print("--- Download process complete. ---")

except Exception as e:
    print(f"--- An unhandled error occurred for model: {repo_id} ---")
    traceback.print_exc()