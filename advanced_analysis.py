import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    print("Loading and preparing data...")
    df = pd.read_csv('Test Data.csv')
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def optimize_random_forest(X_train, y_train):
    print("\nOptimizing Random Forest...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    return grid_search.best_estimator_

def optimize_neural_network(X_train, y_train):
    print("\nOptimizing Neural Network...")
    param_grid = {
        'hidden_layer_sizes': [(100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01]
    }
    mlp = MLPClassifier(max_iter=300, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    return grid_search.best_estimator_

def create_ensemble(models):
    print("\nCreating Voting Ensemble...")
    estimators = [(name, model) for name, model in models.items()]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    return ensemble

def analyze_misclassifications(model, X_test, y_test, n_samples=5):
    y_pred = model.predict(X_test)
    misclassified_indices = np.where(y_pred != y_test)[0]
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(misclassified_indices[:n_samples]):
        plt.subplot(1, n_samples, i + 1)
        img = X_test[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f'True: {y_test.iloc[idx]}\nPred: {y_pred[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    plt.close()

def plot_learning_curves(model, X_train, y_train):
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=3, n_jobs=-1,
        train_sizes=train_sizes, scoring='accuracy'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('learning_curves.png')
    plt.close()

def visualize_model_comparison(models, X_test, y_test):
    scores = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        scores[name] = classification_report(y_test, y_pred, output_dict=True)['accuracy']
    
    plt.figure(figsize=(10, 5))
    plt.bar(scores.keys(), scores.values())
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Optimize individual models
    best_rf = optimize_random_forest(X_train, y_train)
    best_nn = optimize_neural_network(X_train, y_train)
    
    # Create model dictionary
    models = {
        'RandomForest': best_rf,
        'NeuralNetwork': best_nn,
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Train ensemble
    ensemble = create_ensemble(models)
    print("\nTraining Ensemble...")
    ensemble.fit(X_train, y_train)
    models['Ensemble'] = ensemble
    
    # Evaluate all models
    print("\nEvaluating models...")
    for name, model in models.items():
        if name != 'Ensemble':  # Ensemble is already trained
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_model_comparison(models, X_test, y_test)
    analyze_misclassifications(ensemble, X_test, y_test)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()
