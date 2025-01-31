# AI Digit Recognition App

An interactive web application that uses machine learning to recognize hand-drawn digits. Built with Streamlit and scikit-learn.

## Features

- ✏️ Draw digits using an interactive canvas
- 🔮 Real-time digit prediction
- 📊 Prediction confidence visualization
- 📱 Responsive design
- 💾 History tracking of predictions

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd digit-recognition-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open your web browser and go to the URL shown in the terminal (usually http://localhost:8501)

3. Use the drawing canvas to draw a digit (0-9) and click "Predict" to see the AI's prediction

## Project Structure

```
digit-recognition-app/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── models/            # Trained models
│   └── digit_model.joblib
├── static/           # Static files (if any)
└── README.md         # Project documentation
```

## Dependencies

- Python 3.8+
- Streamlit
- scikit-learn
- OpenCV
- NumPy
- Joblib

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
