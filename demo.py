"""
Enhanced Agricultural Model Training Pipeline
This script combines crop recommendation and fertilizer optimization 
in a unified training pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """
    Load and preprocess both crop and fertilizer datasets
    Returns preprocessed features and targets
    """
    logger.info("Loading datasets...")
    
    # Load crop recommendation data
    try:
        crop_df = pd.read_csv('crop_data.csv')
        logger.info(f"Crop data loaded successfully! Shape: {crop_df.shape}")
    except Exception as e:
        logger.error(f"Error loading crop data: {e}")
        raise
        
    # Load fertilizer optimization data
    try:
        fertilizer_df = pd.read_csv('fertilizer.csv')
        logger.info(f"Fertilizer data loaded successfully! Shape: {fertilizer_df.shape}")
    except Exception as e:
        logger.error(f"Error loading fertilizer data: {e}")
        raise

    # Preprocess crop data
    crop_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X_crop = crop_df[crop_features]
    y_crop = crop_df['label']

    # Preprocess fertilizer data
    # Align fertilizer data with crop features
    fertilizer_df = fertilizer_df.drop(['Unnamed: 0'], axis=1, errors='ignore')
    fertilizer_features = ['N', 'P', 'K', 'pH']
    X_fertilizer = fertilizer_df[fertilizer_features]
    y_fertilizer = fertilizer_df['Crop']

    # Combine datasets (focusing on common features: N, P, K, pH)
    common_features = ['N', 'P', 'K', 'ph']
    X_crop_common = X_crop[common_features]
    X_fertilizer_common = X_fertilizer.rename(columns={'pH': 'ph'})

    # Combine features and targets
    X_combined = pd.concat([X_crop_common, X_fertilizer_common], axis=0)
    y_combined = pd.concat([y_crop, y_fertilizer], axis=0)

    logger.info(f"Combined dataset shape: {X_combined.shape}")
    return X_combined, y_combined, crop_features

def train_model(X, y):
    """
    Train a RandomForest model with comprehensive logging
    """
    logger.info("Starting model training pipeline...")
    
    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Feature scaling completed")

    # Initialize and train model
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Training loop with progress logging
    logger.info("Starting training loop...")
    train_start = datetime.now()
    
    model.fit(X_train_scaled, y_train)
    
    train_end = datetime.now()
    training_time = (train_end - train_start).total_seconds()
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Model evaluation
    logger.info("Evaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    class_names = label_encoder.classes_
    logger.info("\nDetailed Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=class_names))

    return model, scaler, label_encoder

def save_model(model, scaler, label_encoder, crop_features):
    """
    Save trained model and preprocessing objects
    """
    logger.info("Saving model and preprocessors...")
    try:
        with open('model_demo.pkl', 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'features': crop_features
            }, f)
        logger.info("Model and preprocessors saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    """
    Main training pipeline
    """
    logger.info("=" * 50)
    logger.info("Starting Agricultural Model Training Pipeline")
    logger.info("=" * 50)

    try:
        # Load and preprocess data
        X_combined, y_combined, crop_features = load_and_preprocess_data()

        # Train model
        model, scaler, label_encoder = train_model(X_combined, y_combined)

        # Save trained model and preprocessors
        save_model(model, scaler, label_encoder, crop_features)

        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()