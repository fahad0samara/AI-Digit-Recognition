import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_mnist_data(file_path):
    """Load MNIST data from CSV"""
    try:
        print("Loading data...")
        data = pd.read_csv(file_path)
        
        # Separate features and labels
        X = data.drop('label', axis=1) if 'label' in data.columns else data.iloc[:, 1:]
        y = data['label'] if 'label' in data.columns else data.iloc[:, 0]
        
        # Normalize pixel values to [0, 1]
        X = X / 255.0
        
        return X, y
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    rf = RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=30,      # Deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf

def train_neural_network(X_train, y_train):
    """Train Neural Network model"""
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # Larger network
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate='adaptive',
        max_iter=300,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    return mlp

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    return accuracy

def save_model(model, scaler, model_dir='models'):
    """Save the trained model and scaler"""
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        print("\nSaving model and scaler...")
        joblib.dump(model, os.path.join(model_dir, 'digit_classifier_ensemble.joblib'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
        print("Model and scaler saved successfully!")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def main():
    # File paths
    train_file = "Train Data.csv"
    
    # Load and prepare data
    X, y = load_mnist_data(train_file)
    if X is None or y is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train_scaled, y_train)
    rf_accuracy = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    
    print("\nTraining Neural Network...")
    nn_model = train_neural_network(X_train_scaled, y_train)
    nn_accuracy = evaluate_model(nn_model, X_test_scaled, y_test, "Neural Network")
    
    # Choose best model
    if rf_accuracy > nn_accuracy:
        print("\nUsing Random Forest model (better accuracy)")
        final_model = rf_model
    else:
        print("\nUsing Neural Network model (better accuracy)")
        final_model = nn_model
    
    # Save model
    save_model(final_model, scaler)

if __name__ == "__main__":
    main()
