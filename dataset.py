import os
import random
import urllib.request
import tarfile

# This file that contains the list of image URLs for Places365 dataset (gathered from MIT)
URL = "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"

def download_file(url, path):
    print(f"\nDownloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")

def extract_tar(path, out_dir):
    print("\nExtracting file list...")
    with tarfile.open(path, "r") as tar:
        tar.extractall(out_dir)
    print("Extraction complete.")

def load_image_urls(list_dir):
    urls = []
    for txt in ["places365_train_standard.txt",
                "places365_val.txt"]:
        txt_path = os.path.join(list_dir, txt)
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        # The first field is the relative path
                        rel_path = parts[0]
                        full_url = f"http://data.csail.mit.edu/places/places365/{rel_path}"
                        urls.append(full_url)
    return urls

def download_small_subset(num_images=5000, out_dir="places_small"):
    os.makedirs(out_dir, exist_ok=True)

    # Download list tar
    tar_path = "filelist_places365-standard.tar"
    download_file(URL, tar_path)

    # Extract tar
    extract_tar(tar_path, "filelists")

    # Load URLs from extracted lists
    print("\nLoading URLs from extracted lists...")
    urls = load_image_urls("filelists")
    print(f"Found {len(urls)} URLs.")

    # Sample subset of URLs
    print(f"Selecting {num_images} random images...")
    sample_urls = random.sample(urls, num_images)

    # Download images from sampled URLs
    print("Starting download...")
    count = 0
    for i, url in enumerate(sample_urls):
        filename = os.path.join(out_dir, f"img_{i}.jpg")

        try:
            urllib.request.urlretrieve(url, filename)
            count += 1
        except:
            continue  # skip bad links

        if (i + 1) % 500 == 0:
            print(f"{i+1}/{num_images} downloaded...")

    print(f"\n✅ Finished! Downloaded {count} images into '{out_dir}'")

if __name__ == "__main__":
    download_small_subset(num_images=5000)