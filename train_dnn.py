

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import joblib # To save the trained model

# # --- Configuration ---
# FEATURE_CSV_PATH = 'save_features.csv'
# MODEL_SAVE_PATH = 'dnn_classifier.pkl'

# def train_and_evaluate_dnn(csv_path):
#     print("1. Loading features and labels...")
#     try:
#         df = pd.read_csv(csv_path)
#     except FileNotFoundError:
#         print(f"Error: Feature CSV not found at '{csv_path}'. Run main.py first.")
#         return

#     # Extract features (columns starting with 'feature_')
#     feature_columns = [col for col in df.columns if col.startswith('feature_')]
#     if not feature_columns:
#         print("Error: No feature columns found in the CSV.")
#         return

#     X = df[feature_columns]
#     y = df['label'] # The disease class (e.g., 'COVID19', 'NORMAL')
    
#     # 2. Preprocessing: Encoding Labels
#     print(f"Unique classes: {y.unique()}")
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)

#     # 3. Data Splitting: 80% Training, 10% Validation, 10% Test
#     # First split: 80% Train, 20% Temp
#     X_train, X_temp, y_train, y_temp = train_test_split(
#         X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#     )
#     # Second split: 10% Validation, 10% Test (from Temp)
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
#     )
#     print(f"Data split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

#     # 4. Preprocessing: Scaling Features
#     print("Scaling features...")
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test)

#     # 5. Model Training (Multi-Layer Perceptron)
#     # Hyperparameters: 3 hidden layers of size 512, 256, 128
#     # 'adam' solver is good for large datasets.
#     print("Starting DNN (MLP) training...")
#     dnn_model = MLPClassifier(
#         hidden_layer_sizes=(512, 256, 128),
#         activation='relu',
#         solver='adam',
#         max_iter=500, # Increased iterations for better convergence
#         random_state=42,
#         verbose=True
#     )
    
#     # Train the model on the training set
#     dnn_model.fit(X_train_scaled, y_train)
#     print("Training complete.")

#     # 6. Model Evaluation (using the Test Set)
#     print("\n--- Model Evaluation on Test Set ---")
#     y_pred_test = dnn_model.predict(X_test_scaled)
    
#     test_accuracy = accuracy_score(y_test, y_pred_test)
#     print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
#     target_names = label_encoder.classes_
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred_test, target_names=target_names))

#     # 7. Save the trained model and the scaler
#     joblib.dump(dnn_model, MODEL_SAVE_PATH)
#     joblib.dump(scaler, 'dnn_scaler.pkl')
#     joblib.dump(label_encoder, 'dnn_label_encoder.pkl')
#     print(f"\nModel, Scaler, and Encoder saved successfully to '{MODEL_SAVE_PATH}' and associated files.")

# if __name__ == '__main__':
#     train_and_evaluate_dnn(FEATURE_CSV_PATH)






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib 

# --- Configuration ---
FEATURE_CSV_PATH = 'save_features.csv'
MODEL_SAVE_PATH = 'dnn_classifier.pkl'

def train_and_evaluate_dnn(csv_path):
    print("1. Loading features and labels...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Feature CSV not found at '{csv_path}'. Run main.py first.")
        return

    # Extract features (columns starting with 'feature_')
    feature_columns = [col for col in df.columns if col.startswith('feature_')]
    if not feature_columns:
        print("Error: No feature columns found in the CSV.")
        return

    X = df[feature_columns]
    y = df['label']
    
    # 2. Preprocessing: Encoding Labels
    print(f"Unique classes: {y.unique()}")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 3. Data Splitting: 80% Training, 10% Validation, 10% Test
    # First split: 80% Train, 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    # Second split: 10% Validation, 10% Test (from Temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Data split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

    # 4. Preprocessing: Scaling Features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Training (Multi-Layer Perceptron)
    # FIX: Increase hidden layer capacity and max_iter to encourage better generalization
    print("Starting DNN (MLP) training with increased capacity...")
    dnn_model = MLPClassifier(
        # Increased model size: Added one layer and more neurons for complexity
        hidden_layer_sizes=(1024, 512, 256, 128),
        activation='relu',
        solver='adam',
        max_iter=1000, # Increased max iterations to prevent early stopping bias
        random_state=42,
        tol=1e-5, # Tighter tolerance to find a better minimum
        verbose=True
    )
    
    # Train the model on the training set
    dnn_model.fit(X_train_scaled, y_train)
    print("Training complete.")

    # 6. Model Evaluation (using the Test Set)
    print("\n--- Model Evaluation on Test Set ---")
    y_pred_test = dnn_model.predict(X_test_scaled)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    target_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    # 7. Save the trained model and the scaler
    joblib.dump(dnn_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, 'dnn_scaler.pkl')
    joblib.dump(label_encoder, 'dnn_label_encoder.pkl')
    print(f"\nModel, Scaler, and Encoder saved successfully to '{MODEL_SAVE_PATH}' and associated files.")

if __name__ == '__main__':
    train_and_evaluate_dnn(FEATURE_CSV_PATH)