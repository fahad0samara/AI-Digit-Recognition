# 🎯 AI Digit Recognition App

A modern, interactive web application that uses machine learning to recognize hand-drawn digits. Built with Streamlit and scikit-learn, featuring a beautiful UI and real-time predictions.

<div align="center">
  <img src="static/sample_digits.png" alt="Sample Digits" width="600"/>
</div>

## ✨ Features

- 🎨 **Interactive Drawing Canvas**: Draw digits naturally using your mouse or touch device
- 🤖 **Real-time Prediction**: Instant AI predictions as you draw
- 📊 **Confidence Visualization**: See how confident the AI is about its predictions
- 📱 **Responsive Design**: Beautiful, modern UI that works on all devices
- 💾 **History Tracking**: Keep track of your previous predictions
- 📈 **Advanced Analytics**: Jupyter notebook included for detailed model analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fahad0samara/AI-Digit-Recognition.git
cd AI-Digit-Recognition
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Use the app:
   - Draw a digit (0-9) on the canvas
   - Click "Predict" to see the AI's prediction
   - View prediction confidence and history
   - Try different digits and compare results

## 📊 Data Analysis

The project includes a comprehensive Jupyter notebook (`digit_recognition_analysis.ipynb`) for detailed analysis:

- 📈 Model performance metrics
- 🎯 Feature importance analysis
- 🔍 Misclassification analysis
- 📊 Confidence distribution
- 🖼️ Sample visualization

To run the analysis:
```bash
jupyter notebook digit_recognition_analysis.ipynb
```

## 🏗️ Project Structure

```
digit-recognition-app/
├── app.py                         # Main Streamlit application
├── digit_recognition_analysis.ipynb # Jupyter notebook for analysis
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
├── LICENSE                        # MIT license
├── .gitignore                    # Git ignore rules
├── models/                       # Trained model files
│   ├── digit_model.joblib        # Trained classifier
│   └── scaler.joblib            # Feature scaler
└── static/                      # Static files and images
    ├── sample_digits.png        # Sample digit visualizations
    ├── confusion_matrix.png     # Model performance visualization
    └── feature_importance.png   # Feature analysis visualization
```

## 🛠️ Technical Details

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Input**: 28x28 grayscale images (784 features)
- **Output**: Digit prediction (0-9) with confidence scores

### Performance Metrics
- Accuracy: ~98% on test set
- Fast inference time (<100ms)
- Robust to different writing styles

### Technologies Used
- **Frontend**: Streamlit, HTML/CSS
- **Backend**: Python, scikit-learn
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Image Processing**: OpenCV

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MNIST dataset for training data
- Streamlit team for the amazing framework
- scikit-learn community for machine learning tools

## 📧 Contact

Fahad - fahad0samara@gmail.com

Project Link: [https://github.com/fahad0samara/AI-Digit-Recognition](https://github.com/fahad0samara/AI-Digit-Recognition)
