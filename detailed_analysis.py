import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def load_and_prepare_data():
    print("Loading data...")
    df = pd.read_csv('Test Data.csv')
    X = df.drop('label', axis=1)
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def visualize_digit_distribution(y):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=y)
    plt.title('Distribution of Digits')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.savefig('digit_distribution.png')
    plt.close()

def visualize_sample_digits(X, y):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        img = X.iloc[i].values.reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Digit: {y.iloc[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_digits.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
    plt.close()

def visualize_feature_importance(model, X):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        plt.figure(figsize=(10, 4))
        plt.imshow(importances.reshape(28, 28), cmap='hot')
        plt.colorbar()
        plt.title('Feature Importance Heatmap')
        plt.savefig('feature_importance.png')
        plt.close()

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        plot_confusion_matrix(y_test, y_pred, name)
        
        if name == 'Random Forest':
            visualize_feature_importance(model, X_train)
        
        results[name] = {
            'model': model,
            'predictions': y_pred
        }
    
    return results

def perform_pca_analysis(X):
    print("\nPerforming PCA analysis...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='b', alpha=0.5)
    plt.title('PCA: First Two Principal Components')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig('pca_visualization.png')
    plt.close()
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Visualize data distribution
    print("Generating visualizations...")
    visualize_digit_distribution(y_train)
    visualize_sample_digits(X_train, y_train)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Perform PCA analysis
    perform_pca_analysis(X_train)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()
