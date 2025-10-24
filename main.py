import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import sys

def get_device():
    if torch.cuda.is_available():
        print("CUDA detected. Using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS detected. Using Apple Metal.")
        return torch.device("mps")
    else:
        print("No GPU or MPS detected. Using CPU.")
        return torch.device("cpu")

def validate_dataset(train_path, class_names):
    # Checks if the training directory and class subdirectories exist and are not empty.
    
    if not os.path.isdir(train_path):
        print(f"Error: Training directory not found at '{train_path}'")
        sys.exit(1)

    print("Dataset validation:")
    for class_name in class_names:
        class_path = os.path.join(train_path, class_name)
        if not os.path.isdir(class_path):
            print(f"  - Error: Class directory not found: '{class_path}'")
            sys.exit(1)

        # Check for at least one jpg image
        images = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]
        if not images:
            print(f"  - Error: No JPG images found in '{class_path}'")
            sys.exit(1)
        print(f"  - Success: Found {len(images)} images in '{class_name}'.")

    return train_path


def extract_features(train_data_path='train_jpg', output_csv='features.csv'): 
    # Extracts features from images using a pre-trained DenseNet model.
    
    # 1. Setup and Validation
    device = get_device()
    class_names = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
    train_path = validate_dataset(train_data_path, class_names)

    # 2. Load Pre-trained DenseNet model
    print("\nLoading pre-trained DenseNet-121 model...")
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    feature_extractor = model.features
    feature_extractor.to(device)
    feature_extractor.eval()
    print("Model loaded successfully.")

    # 3. Define Image Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Prepare for Feature Extraction
    all_image_paths = []
    for class_name in class_names:
        class_path = os.path.join(train_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith('.jpg'):
                all_image_paths.append(os.path.join(class_path, img_name))

    extracted_data = []
    first_image_processed = False

    print(f"\nStarting feature extraction for {len(all_image_paths)} images...")
    # 5. Loop through images and extract features
    with torch.no_grad():
        for img_path in tqdm(all_image_paths, desc="Extracting Features"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                features = feature_extractor(img_tensor)
                pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
                flat_features = torch.flatten(pooled_features, 1)
                feature_vector = flat_features.cpu().numpy().flatten().tolist()

                if not first_image_processed:
                    print(f"\n--- First Image Diagnostic ---")
                    print(f"Image Path: {img_path}")
                    print(f"Feature vector length: {len(feature_vector)}")
                    if not feature_vector:
                         print("Warning: The feature vector for the first image is EMPTY!")
                    print(f"--------------------------\n")
                    first_image_processed = True

                if not feature_vector:
                    print(f"Warning: Empty feature vector for {img_path}. Skipping.")
                    continue

                label = os.path.basename(os.path.dirname(img_path))
                row = {'image_path': img_path, 'label': label}
                row.update({f'feature_{i}': val for i, val in enumerate(feature_vector)})
                extracted_data.append(row)

            except Exception as e:
                print(f"Warning: Could not process file {img_path}. Error: {e}")

    # 6. Save results to CSV
    if not extracted_data:
        print("\nNo features were extracted. The output file will not be created.")
        return

    print("\nFeature extraction complete. Preparing to save...")
    print(f"Number of processed images with features: {len(extracted_data)}")

    df = pd.DataFrame(extracted_data)

    print("\nDataFrame created. Displaying info and first 5 rows:")
    print(f"DataFrame Shape (rows, columns): {df.shape}")
    print("DataFrame Head:")
    print(df.head())

    print(f"\nSaving DataFrame with shape {df.shape} to CSV...")
    try:
        with open(output_csv, 'w', newline='') as f:
            df.to_csv(f, index=False)
        print(f"Successfully saved features for {len(df)} images to '{output_csv}'.")
    except Exception as e:
        print(f"\n--- An error occurred while saving the file ---")
        print(f"Error: {e}")
        print("--------------------------------------------------")


if __name__ == '__main__':
    extract_features(train_data_path='dataset/train', output_csv='save_features.csv')