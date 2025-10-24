import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2 # Used for image resizing and colormapping
import joblib 
import os
import matplotlib.pyplot as plt # Used for visualization
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Configuration (Should match main.py and train_dnn.py) ---
MODEL_WEIGHTS = models.DenseNet121_Weights.DEFAULT 
# This layer is the final Batch Normalization layer before the final ReLU and Pooling
TARGET_LAYER_NAME = 'features.norm5' 

# Helper Class to hook the forward pass and gradients
class GradCam:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.feature_extractor = self.model.features
        
        # Get the target layer module
        self.target_layer = dict(self.model.named_modules())[target_layer_name]

        # Hook storage
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        # Hook for forward pass (to get feature maps/activations)
        def forward_hook(module, input, output):
            self.activations = output.detach()
        self.target_layer.register_forward_hook(forward_hook)

        # Hook for backward pass (to get gradients)
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates the Grad-CAM heatmap using a manual forward pass 
        to bypass the DenseNet inplace ReLU issue.
        """
        
        # 1. Manual Forward pass to get feature maps (F_k)
        # This triggers the hook on features.norm5 to save activations (self.activations)
        features = self.model.features(input_tensor) 
        
        # 2. Complete the forward pass manually, ensuring no inplace=True operation is used.
        # This is the CRITICAL FIX for the RuntimeError.
        out = F.relu(features, inplace=False) 
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        output = self.model.classifier(out) # Final score from PyTorch model (needed for gradient)

        # Determine the target class index
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 3. Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True) 

        # 4. Get the feature map and gradients
        activations = self.activations[0] 
        gradients = self.gradients[0] 

        # 5. Compute the neuron importance weights (alpha_k)
        alpha = torch.mean(gradients, dim=(1, 2), keepdim=True) 

        # 6. Weighted combination and ReLU
        weighted_activations = activations * alpha
        heatmap = torch.sum(weighted_activations, dim=0)
        heatmap = F.relu(heatmap)

        # 7. Normalize the heatmap
        # Check for division by zero
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        
        # Resize to 224x224 (image input size)
        # FIX: Change mode='linear' to mode='bilinear' to fix the NotImplementedError
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0), # Input shape is [1, 1, H, W] (4D)
            size=(224, 224), 
            mode='bilinear', # Use bilinear mode for 2D image resizing
            align_corners=False
        )
        heatmap = heatmap.squeeze().cpu().numpy()

        return heatmap

def get_full_model(device):
    """Loads the DenseNet-121 model."""
    model = models.densenet121(weights=MODEL_WEIGHTS)
    model.to(device)
    model.eval()
    return model

def visualize_heatmap(image_path, heatmap, prediction_label):
    """Overlay the heatmap onto the original image and display the result using matplotlib."""
    
    # Read the original image in RGB format using PIL for matplotlib
    img_orig = Image.open(image_path).convert('RGB')
    img_np = np.array(img_orig)
    
    # Resize heatmap to match the original image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Convert heatmap back to RGB since OpenCV uses BGR by default
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create the overlayed image: 0.6*img + 0.4*heatmap
    superimposed_img = heatmap_colored_rgb * 0.4 + img_np * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # 1. Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img_orig)
    plt.title(f'Original X-Ray')
    plt.axis('off')

    # 2. Grad-CAM Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f'Grad-CAM Heatmap')
    plt.axis('off')
    
    plt.suptitle(f"XAI Diagnosis: {prediction_label}", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("Heatmap visualized successfully.")

def run_grad_cam(image_path):
    """Main function to run the classification and Grad-CAM visualization."""
    
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load the full DenseNet model (PyTorch model)
    model = get_full_model(device) 

    # 3. Load trained scikit-learn model and encoders for actual prediction
    try:
        dnn_model = joblib.load('dnn_classifier.pkl')
        scaler = joblib.load('dnn_scaler.pkl')
        label_encoder = joblib.load('dnn_label_encoder.pkl')
    except FileNotFoundError:
        print("Error: DNN classifier files not found. Ensure train_dnn.py was run successfully.")
        return

    # 4. Image Preprocessing (Must match main.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 5. Feature Extraction (To feed the DNN for prediction)
    with torch.no_grad():
        # Get features up to features.norm5
        features = model.features(input_tensor)
        
        # Manually apply the final steps (non-in-place ReLU and Pooling) 
        # to get the 1024 features for the scikit-learn DNN
        pooled_features = F.adaptive_avg_pool2d(F.relu(features, inplace=False), (1, 1)) 
        flat_features = torch.flatten(pooled_features, 1).cpu().numpy()
    
    # 6. Actual Prediction using the scikit-learn DNN
    scaled_features = scaler.transform(flat_features)
    predicted_class_idx = dnn_model.predict(scaled_features)[0]
    prediction_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    print(f"Prediction for {os.path.basename(image_path)}: {prediction_label} (Index: {predicted_class_idx})")

    # 7. Generate Grad-CAM Heatmap
    print("Generating Grad-CAM heatmap...")
    cam_extractor = GradCam(model, TARGET_LAYER_NAME)
    heatmap = cam_extractor.generate_heatmap(input_tensor, target_class=predicted_class_idx)
    
    # 8. Visualize the result
    visualize_heatmap(image_path, heatmap, prediction_label)


if __name__ == '__main__':
    # --- Example Usage (UPDATE THIS PATH TO A VALID IMAGE IN YOUR FILE SYSTEM) ---
    # Example: run_grad_cam('dataset/test/COVID19/COVID19(460).jpg') 
    
    # The image path from your previous attempt:
    run_grad_cam('dataset/test/TURBERCULOSIS/Tuberculosis-664.png')

    