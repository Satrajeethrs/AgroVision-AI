import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings

warnings.filterwarnings('ignore')

def train_crop_recommendation_model():
    """
    Trains a Random Forest classifier for crop recommendation using a larger dataset.
    Features: N, P, K, temperature, humidity, ph, rainfall.
    """
    
    print("=" * 60)
    print("SMART AGRICULTURAL ADVISOR - V3 MODEL TRAINING")
    print("=" * 60)
    
    # Load the enhanced crop dataset
    print("\n[1] Loading dataset 'crop_data.csv'...")
    try:
        # The user should have renamed 'Crop_recommendation.csv' to 'crop_data.csv'
        df = pd.read_csv('crop_data.csv')
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        print(f"Crops in dataset: {len(df['label'].unique())} unique varieties")
    except FileNotFoundError:
        print("Error: crop_data.csv not found!")
        print("Please ensure you have renamed 'Crop_recommendation.csv' to 'crop_data.csv'.")
        return None, None
    
    # Prepare features and target. NOTE: 'SOC' is no longer a feature.
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_names]
    y = df['label']
    
    print(f"\n[2] Feature Analysis:")
    print(f"Features used: {feature_names}")
    print(f"Target variable: 'label' (Crop Name)")
    
    # Split the data into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[3] Data Preparation:")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Initialize and fit the StandardScaler
    print("\n[4] Feature Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the fitted scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to 'scaler.pkl'")
    
    # Enhanced Random Forest with hyperparameter tuning
    print("\n[5] Model Training with Hyperparameter Optimization...")
    
    param_grid = {
        'n_estimators': [150, 200],
        'max_depth': [15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    base_rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        base_rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    rf_classifier = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Make predictions on the test set
    print("\n[6] Model Evaluation...")
    y_pred = rf_classifier.predict(X_test_scaled)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Print a detailed classification report
    print("\n[7] Detailed Classification Report:")
    print("=" * 50)
    print(classification_report(y_test, y_pred))
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n[8] Feature Importance Analysis:")
    print(feature_importance)
    
    # Save the trained model
    with open('model.pkl', 'wb') as f:
        pickle.dump(rf_classifier, f)
    print("\n[9] Model successfully saved to 'model.pkl'")
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("You can now run 'python app.py' to start the web application.")
    
    return rf_classifier, scaler

if __name__ == "__main__":
    train_crop_recommendation_model()
