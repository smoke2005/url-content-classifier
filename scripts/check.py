import os
from PIL import Image

def find_corrupt_images(directory):
    bad_images = []
    for root, _, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                img = Image.open(fpath)
                img.verify()  # Check for corruption
            except Exception as e:
                print(f"Corrupt or unreadable file: {fpath} -> {e}")
                bad_images.append(fpath)
    return bad_images

bad_files = find_corrupt_images("C:\\Users\\mokit\\Downloads\\harmful_content_collection_scripts\\data")

print(f"\nTotal bad files found: {len(bad_files)}")
