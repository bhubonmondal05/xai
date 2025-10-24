import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import joblib 
import os
import base64
from io import BytesIO

# Flask imports
from flask import Flask, request, jsonify

# --- Configuration (MUST MATCH main.py and grad_cam.py) ---
MODEL_WEIGHTS = models.DenseNet121_Weights.DEFAULT 
TARGET_LAYER_NAME = 'features.norm5' 

# Global variables for model storage
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None # PyTorch DenseNet model
dnn_model = None # Scikit-learn classifier
scaler = None # Scikit-learn scaler
label_encoder = None # Scikit-learn label encoder

# --- GradCam Class (Copied from grad_cam.py) ---
# NOTE: The manual forward pass fix is implemented here.
class GradCam:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.target_layer = dict(self.model.named_modules())[target_layer_name]
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        self.target_layer.register_forward_hook(forward_hook)

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class):
        # Manual Forward Pass (Fixes the DenseNet RuntimeError)
        features = self.model.features(input_tensor) 
        out = F.relu(features, inplace=False) 
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        output = self.model.classifier(out) 

        # Backward Pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True) 

        # Grad-CAM calculation
        activations = self.activations[0] 
        gradients = self.gradients[0] 
        alpha = torch.mean(gradients, dim=(1, 2), keepdim=True) 
        weighted_activations = activations * alpha
        heatmap = torch.sum(weighted_activations, dim=0)
        heatmap = F.relu(heatmap)

        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        
        # Resizing (Fixes the NotImplementedError)
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0), 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        return heatmap.squeeze().cpu().numpy()

# --- Utility Functions ---

def load_models():
    """Load models once at server startup."""
    global model, dnn_model, scaler, label_encoder
    
    # 1. Load PyTorch DenseNet Model
    print("Loading DenseNet-121...")
    model = models.densenet121(weights=MODEL_WEIGHTS)
    model.to(device)
    model.eval()
    
    # 2. Load Scikit-learn DNN and Preprocessors
    print("Loading DNN classifier and preprocessors...")
    try:
        dnn_model = joblib.load('dnn_classifier.pkl')
        scaler = joblib.load('dnn_scaler.pkl')
        label_encoder = joblib.load('dnn_label_encoder.pkl')
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}. Ensure train_dnn.py was run.")
        raise RuntimeError("Model files missing.")

def process_and_encode_heatmap(img_path, heatmap, prediction_label):
    """Overlays heatmap on original image and returns the Base64 string."""
    
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        return None
        
    img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # 1. Resize and Colormap Heatmap
    heatmap_resized = cv2.resize(heatmap, (img_orig_rgb.shape[1], img_orig_rgb.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Convert heatmap back to RGB since OpenCV uses BGR
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # 2. Create Overlay
    superimposed_img = heatmap_colored_rgb * 0.4 + img_orig_rgb * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Convert to BGR for encoding with cv2
    superimposed_img_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)

    # 3. Base64 Encoding
    # Encode as PNG byte array
    _, buffer = cv2.imencode('.png', superimposed_img_bgr)
    # Convert byte array to Base64 string
    base64_encoded = base64.b64encode(buffer).decode('utf-8')
    
    return base64_encoded

# --- Flask API Endpoint ---

@app.route('/api/predict', methods=['POST'])
def predict_and_explain():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']
    
    # 1. Save File Temporarily
    temp_path = os.path.join('/tmp', image_file.filename)
    # Use a safe name in a reliable temporary directory
    if not os.path.exists('/tmp'): # Fallback for non-Linux/Mac systems
        os.makedirs('/tmp') 
    
    image_file.save(temp_path)
    
    try:
        # 2. Image Preprocessing (Matches main.py)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img = Image.open(temp_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        # 3. Feature Extraction & Prediction
        with torch.no_grad():
            features = model.features(input_tensor)
            # Manual final steps for features used by scikit-learn DNN
            pooled_features = F.adaptive_avg_pool2d(F.relu(features, inplace=False), (1, 1)) 
            flat_features = torch.flatten(pooled_features, 1).cpu().numpy()
        
        # Scale features and predict
        scaled_features = scaler.transform(flat_features)
        
        # Get prediction index and label
        predicted_class_idx = dnn_model.predict(scaled_features)[0]
        prediction_label = label_encoder.inverse_transform([predicted_class_idx])[0]

        # Get prediction probabilities for confidence
        probabilities = dnn_model.predict_proba(scaled_features)[0]
        confidence = probabilities[predicted_class_idx]
        
        # 4. Generate Grad-CAM Heatmap
        cam_extractor = GradCam(model, TARGET_LAYER_NAME)
        heatmap = cam_extractor.generate_heatmap(input_tensor, target_class=predicted_class_idx)
        
        # 5. Process and Encode Heatmap to Base64
        base64_img = process_and_encode_heatmap(temp_path, heatmap, prediction_label)

        # 6. Return Result
        return jsonify({
            'prediction': prediction_label,
            'confidence': f"{confidence:.4f}",
            'heatmap_image': base64_img,
            'status': 'success'
        })

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Server Start ---
if __name__ == '__main__':
    # Load models before running the app
    try:
        load_models()
        print("\n--- Flask Server Starting ---")
        # Run on a common port (5000) accessible externally
        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False) 
        # use_reloader=False prevents the model from loading twice
    except RuntimeError as e:
        print(f"\nFATAL ERROR: Could not start server. {e}")
        print("Please run train_dnn.py to generate the required model files.")