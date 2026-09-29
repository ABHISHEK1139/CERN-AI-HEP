import os
import time

import requests

url = "https://opendata.cern.ch/record/12353/files/DYJetsToLL.root"
output_path = "data/cms/dyjets/DYJetsToLL.root"

def download_file(url, path):
    print(f"Starting download of {url}")
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    headers = {}

    try:
        # NOTE: no Range/resume support — partial files are deleted on failure
        # and the whole file is re-downloaded on retry. No checksum is
        # verified; validate the ROOT file with uproot after download.
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1 Megabyte

        with open(path, 'wb') as f:
            downloaded = 0
            next_report = 100 * 1024 * 1024  # report every ~100MB
            for data in response.iter_content(block_size):
                if not data:
                    continue
                f.write(data)
                downloaded += len(data)

                # Print progress every ~100MB
                if downloaded >= next_report:
                    total_gb = total_size / 1024 / 1024 / 1024 if total_size else 0
                    print(f"Downloaded {downloaded / 1024 / 1024 / 1024:.2f} GB / {total_gb:.2f} GB")
                    next_report += 100 * 1024 * 1024

        print(f"Download completed successfully: {path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return False

if __name__ == "__main__":
    # Try up to 10 times
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}")
        if download_file(url, output_path):
            break
        else:
            if attempt == max_retries:
                print("All retries exhausted.")
                raise SystemExit(1)
            print("Retrying in 10 seconds...")
            time.sleep(10)

