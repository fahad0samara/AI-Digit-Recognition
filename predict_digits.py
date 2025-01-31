import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import joblib

def load_and_train_model():
    """Load data, train the model and save it"""
    print("Loading training data...")
    train_data = pd.read_csv('Test Data.csv')
    X = train_data.drop('label', axis=1)
    y = train_data['label']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train the ensemble model
    print("Training ensemble model...")
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', 
                                     learning_rate_init=0.01, max_iter=300, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    
    ensemble.fit(X_scaled, y)
    
    # Save the model and scaler
    print("Saving model and scaler...")
    joblib.dump(ensemble, 'digit_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return ensemble, scaler

def load_saved_model():
    """Load the saved model and scaler"""
    try:
        model = joblib.load('digit_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except:
        print("No saved model found. Training new model...")
        return load_and_train_model()

def predict_digit(image_data, model, scaler):
    """Predict a single digit"""
    # Ensure the input is in the right shape
    if image_data.shape != (784,):
        raise ValueError("Input image must be flattened to 784 pixels")
    
    # Scale the data
    scaled_data = scaler.transform(image_data.reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)
    
    return prediction[0], probabilities[0]

def visualize_prediction(image_data, prediction, probabilities):
    """Visualize the image and prediction probabilities"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Show the image
    ax1.imshow(image_data.reshape(28, 28), cmap='gray')
    ax1.set_title(f'Predicted Digit: {prediction}')
    ax1.axis('off')
    
    # Show probability distribution
    digits = range(10)
    ax2.bar(digits, probabilities)
    ax2.set_title('Prediction Probabilities')
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Probability')
    
    plt.tight_layout()
    plt.show()

def predict_test_data(test_file):
    """Predict digits for a test dataset"""
    # Load model and scaler
    model, scaler = load_saved_model()
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    test_data = pd.read_csv(test_file)
    
    # Scale the data
    X_test_scaled = scaler.transform(test_data)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)
    
    # Add predictions to the test data
    results = pd.DataFrame({
        'Predicted_Digit': predictions,
        'Confidence': np.max(probabilities, axis=1)
    })
    
    # Save results
    output_file = 'predictions.csv'
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return results

def main():
    # Example usage
    print("Digit Recognition System")
    print("1. Train/Load Model")
    print("2. Predict Test Data")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        load_and_train_model()
        print("Model trained and saved successfully!")
        
    elif choice == '2':
        test_file = input("Enter the path to test data CSV file: ")
        try:
            results = predict_test_data(test_file)
            print("\nPrediction Summary:")
            print(f"Total predictions made: {len(results)}")
            print("\nSample of predictions:")
            print(results.head())
        except Exception as e:
            print(f"Error: {str(e)}")
            
    elif choice == '3':
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
