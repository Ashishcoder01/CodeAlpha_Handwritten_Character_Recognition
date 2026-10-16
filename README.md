# Handwritten Character Recognition

A machine learning application that recognizes handwritten letters and digits using a Convolutional Neural Network (CNN). The project provides an interactive Streamlit interface for uploading an image or drawing a character and viewing the model's predictions.

## Project Overview

The application uses a CNN trained on the **EMNIST Balanced dataset** to classify handwritten characters across 47 classes.

### Key Features

- Upload an image containing a handwritten character.
- Draw a character directly on the application canvas.
- Predict the most likely character.
- Display confidence scores and the top five predictions.
- Preview the processed 28 × 28 image supplied to the model.
- Interactive web interface built with Streamlit.

## Model Information

| Property | Details |
|---|---|
| Model type | Convolutional Neural Network (CNN) |
| Dataset | EMNIST Balanced |
| Number of classes | 47 |
| Training samples | 112,800 |
| Testing samples | 18,800 |
| Input image size | 28 × 28 pixels |
| Reported test accuracy | 89.08% |
| Model format | Keras (`.keras`) |

*The accuracy above is the reported result from the model's evaluation. Actual predictions may vary depending on image quality, character style, and preprocessing.*

## Technology Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Streamlit Drawable Canvas

## Project Structure

```text
CodeAlpha_Handwritten_Character_Recognition/
├── app.py
├── class_names.json
├── handwritten_character_recognition.keras
├── training_history.json
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/CodeAlpha_Handwritten_Character_Recognition.git
cd CodeAlpha_Handwritten_Character_Recognition
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`.

## How It Works

1. The user uploads an image or draws a handwritten character.
2. The application converts the image into the expected grayscale input format.
3. The image is resized and normalized to match the model's input requirements.
4. The trained CNN generates class predictions.
5. The interface displays the predicted character, confidence, and alternative predictions.

## Dataset

The model was trained using the EMNIST Balanced handwritten-character dataset, which contains handwritten digits and letters across 47 classes.

Dataset reference: [EMNIST — NIST Special Database 19](https://www.nist.gov/itl/products-and-services/emnist-dataset)

## Future Improvements

- Improve recognition of visually similar characters.
- Evaluate performance on images from different sources.
- Add batch image prediction.
- Expand automated testing and model evaluation.
- Deploy the application publicly.

## Internship Project

Developed as part of the CodeAlpha Machine Learning Internship.

## License

No license has been specified yet. Unless a license is added, the repository's code is not automatically granted an open-source license.
