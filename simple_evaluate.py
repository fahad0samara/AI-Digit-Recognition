import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def main():
    # Load data
    print("Loading data...")
    data = pd.read_csv('Test Data.csv')
    
    # Split data
    print("Splitting data...")
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load model and scaler
    print("Loading model and scaler...")
    model = joblib.load('digit_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Scale test data
    print("Scaling test data...")
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test_scaled)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Save predictions
    results = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': predictions
    })
    results.to_csv('test_results.csv', index=False)
    print("\nResults saved to test_results.csv")

if __name__ == "__main__":
    main()
