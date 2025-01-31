import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def split_and_save_data():
    """Split the data into training and test sets and save them"""
    print("Loading data...")
    data = pd.read_csv('Test Data.csv')
    
    # Split the data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Save the splits
    train_data.to_csv('training_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    
    print(f"Data split and saved:")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    return test_data

def load_model_and_predict(test_data):
    """Load the trained model and make predictions"""
    try:
        print("\nLoading model and scaler...")
        model = joblib.load('digit_model.joblib')
        scaler = joblib.load('scaler.joblib')
    except:
        print("Error: Model files not found. Please train the model first using predict_digits.py")
        return None
    
    # Separate features and labels
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    
    # Scale the features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)
    
    return predictions, probabilities, y_test

def visualize_results(test_data, predictions, y_test, probabilities):
    """Visualize the results"""
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('test_confusion_matrix.png')
    plt.close()
    
    # 2. Sample of correct and incorrect predictions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Predictions (Green: Correct, Red: Incorrect)')
    
    # Get indices of correct and incorrect predictions
    correct = np.where(predictions == y_test)[0]
    incorrect = np.where(predictions != y_test)[0]
    
    # Plot some correct predictions
    for i, ax in enumerate(axes[0]):
        if i < len(correct):
            idx = correct[i]
            img = test_data.iloc[idx].drop('label').values.reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'True: {y_test.iloc[idx]}\nPred: {predictions[idx]}', color='green')
        ax.axis('off')
    
    # Plot some incorrect predictions
    for i, ax in enumerate(axes[1]):
        if i < len(incorrect):
            idx = incorrect[i]
            img = test_data.iloc[idx].drop('label').values.reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'True: {y_test.iloc[idx]}\nPred: {predictions[idx]}', color='red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()
    
    # 3. Confidence Distribution
    plt.figure(figsize=(10, 5))
    max_probs = np.max(probabilities, axis=1)
    plt.hist(max_probs, bins=50)
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.savefig('confidence_distribution.png')
    plt.close()

def main():
    # Split and save the data
    test_data = split_and_save_data()
    
    # Load model and make predictions
    results = load_model_and_predict(test_data)
    if results is None:
        return
    
    predictions, probabilities, y_test = results
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Calculate overall accuracy
    accuracy = (predictions == y_test).mean()
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Calculate confidence statistics
    confidence = np.max(probabilities, axis=1)
    print(f"Average Confidence: {confidence.mean():.2%}")
    print(f"Min Confidence: {confidence.min():.2%}")
    print(f"Max Confidence: {confidence.max():.2%}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(test_data, predictions, y_test, probabilities)
    print("\nVisualizations saved:")
    print("1. test_confusion_matrix.png - Shows the confusion matrix")
    print("2. sample_predictions.png - Shows examples of correct and incorrect predictions")
    print("3. confidence_distribution.png - Shows the distribution of prediction confidence")

if __name__ == "__main__":
    main()
