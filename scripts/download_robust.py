import requests
import os
import time

url = "https://opendata.cern.ch/record/12353/files/DYJetsToLL.root"
output_path = "data/cms/dyjets/DYJetsToLL.root"

def download_file(url, path):
    print(f"Starting download of {url}")
    # ensure dir exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    headers = {}
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1 Megabyte
        
        with open(path, 'wb') as f:
            downloaded = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                
                # Print progress every ~100MB
                if downloaded % (100 * block_size) == 0:
                    print(f"Downloaded {downloaded / 1024 / 1024 / 1024:.2f} GB / {total_size / 1024 / 1024 / 1024:.2f} GB")
                    
        print(f"Download completed successfully: {path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

# Try up to 10 times
max_retries = 10
for attempt in range(1, max_retries + 1):
    print(f"Attempt {attempt}/{max_retries}")
    if download_file(url, output_path):
        break
    else:
        print("Retrying in 10 seconds...")
        time.sleep(10)
