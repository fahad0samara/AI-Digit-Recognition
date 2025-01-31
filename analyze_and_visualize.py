import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def load_and_prepare_data():
    # Load the data
    print("Loading data...")
    df = pd.read_csv('Test Data.csv')
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def visualize_samples(X, y, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        img = X.iloc[i].values.reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {y.iloc[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.close()

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Visualize some samples
    print("Generating sample visualizations...")
    visualize_samples(X_train, y_train)
    
    # Train and evaluate model
    train_and_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
