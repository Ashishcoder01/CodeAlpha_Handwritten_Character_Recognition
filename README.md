# Handwritten Character Recognition

A CNN-powered web application that recognizes handwritten digits and letters using the **EMNIST Balanced** dataset. Built with TensorFlow/Keras and Streamlit, it supports both image upload and on-screen drawing.

<p align="center">
  <a href="https://handwrittencharacter-recognition1.streamlit.app/"><strong>Live Demo</strong></a>
  ·
  <a href="https://github.com/Ashishcoder01/CodeAlpha_Handwritten_Character_Recognition"><strong>Source Code</strong></a>
</p>

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

## Overview

The application takes a handwritten character image, applies the preprocessing expected by the trained model, and returns the most likely class. Users can upload an image or draw a character directly in the browser.

## Features

- **Image upload:** Submit an image of a handwritten character.
- **Interactive drawing:** Draw a character on the canvas.
- **Character prediction:** Classify characters across 47 EMNIST Balanced classes.
- **Confidence score:** View the model's confidence for its leading prediction.
- **Top-five results:** Compare the five highest-scoring predictions.
- **Preprocessing preview:** Inspect the 28 × 28 image passed to the model.
- **Web interface:** Use the app in a browser through Streamlit.

## Model Details

| Property | Details |
|---|---|
| Model architecture | Convolutional Neural Network (CNN) |
| Dataset | EMNIST Balanced |
| Number of classes | 47 |
| Training samples | 112,800 |
| Test samples | 18,800 |
| Input dimensions | 28 × 28 pixels, grayscale |
| Reported test accuracy | 89.08% |
| Saved model | Keras `.keras` format |

The reported accuracy is from the model evaluation performed during development. Results for individual images can vary with handwriting style, image quality, and preprocessing.

## Technology Stack

- Python
- TensorFlow and Keras
- Streamlit
- Streamlit Drawable Canvas
- NumPy
- Pillow
- Matplotlib

## Repository Structure

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

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Ashishcoder01/CodeAlpha_Handwritten_Character_Recognition.git
cd CodeAlpha_Handwritten_Character_Recognition
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually [http://localhost:8501](http://localhost:8501).

## How It Works

1. The user uploads an image or draws a character.
2. The app converts the input to grayscale and prepares it for inference.
3. The image is resized to 28 × 28 pixels, oriented and normalized to match the model's expected input.
4. The trained CNN produces scores for the supported character classes.
5. The app displays the leading prediction, confidence, top-five alternatives, and processed input preview.

## Dataset

The model was trained with the **EMNIST Balanced** handwritten-character dataset.

- Dataset reference: [NIST — EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

## Live Application

Try the deployed app here:

**[Open Handwritten Character Recognition](https://handwrittencharacter-recognition1.streamlit.app/)**

For best results, use a clear image containing one character with strong contrast between the character and its background.

## Future Improvements

- Evaluate recognition on a broader set of real-world handwriting samples.
- Add batch prediction for multiple images.
- Expand automated tests and document per-class evaluation metrics.
- Add representative screenshots and examples from the deployed interface.

## Project Context

This project was developed as part of the **CodeAlpha Machine Learning Internship**.

## License

No open-source license has been added to this repository. Unless a license is included, reuse and redistribution are subject to the default copyright rules.
