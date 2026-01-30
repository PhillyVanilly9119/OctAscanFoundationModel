import os
import argparse
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def download_octdl(output_dir):
    print(f"Downloading OCTDL dataset from Hugging Face to {output_dir}...")
    
    # Load dataset (streaming mode to avoid massive initial download if possible, though OCTDL is small enough to just cache)
    # Using 'train' split
    try:
        ds = load_dataset("davanstrien/OCTDL", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading OCTDL: {e}")
        print("Trying secondary dataset: zacharielegault/Kermany2017-OCT (subset)")
        ds = load_dataset("zacharielegault/Kermany2017-OCT", split="train", streaming=True)

    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    max_images = 2000  # Download 2000 images (approx full OCTDL)
    
    print(f"Saving first {max_images} images...")
    
    for i, item in tqdm(enumerate(ds)):
        if count >= max_images:
            break
            
        # Extract image and label
        # Structure varies, typically 'image' key
        if 'image' in item:
            image = item['image']
            
            # Label might be 'label' or 'label_name'
            label = "unknown"
            if 'label' in item:
                label = str(item['label'])
            
            # Create label folder
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            
            # Save
            image.save(os.path.join(label_dir, f"oct_{count}.jpg"))
            count += 1
            
    print(f"Successfully saved {count} images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/OCTDL")
    args = parser.parse_args()
    
    download_octdl(args.output_dir)
