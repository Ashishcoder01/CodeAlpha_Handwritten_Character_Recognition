import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .app-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        margin-top: -10px;
        margin-bottom: 25px;
    }

    .prediction-letter {
        font-size: 7rem;
        font-weight: 800;
        text-align: center;
        line-height: 1;
        padding: 20px;
    }

    .confidence-value {
        font-size: 2rem;
        font-weight: 700;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "handwritten_character_recognition.keras"
)

CLASS_NAMES_PATH = (
    BASE_DIR / "class_names.json"
)


# ============================================================
# EMNIST BALANCED CLASS MAPPING
# ============================================================

EXPECTED_CLASS_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "d",
    "e",
    "f",
    "g",
    "h",
    "n",
    "q",
    "r",
    "t",
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error("Model file not found.")

        st.code(
            str(MODEL_PATH)
        )

        st.stop()

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not CLASS_NAMES_PATH.exists():
        return EXPECTED_CLASS_NAMES

    try:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            names = json.load(f)

        if len(names) == 47:

            if all(
                str(names[i]) == str(i)
                for i in range(47)
            ):
                return EXPECTED_CLASS_NAMES

            return names

    except Exception:
        pass

    return EXPECTED_CLASS_NAMES


# ============================================================
# INITIALIZE
# ============================================================

model = load_model()
class_names = load_class_names()


# ============================================================
# SIDEBAR — MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.caption(
        "Handwritten Character Recognition"
    )

    st.divider()

    st.subheader("Dataset")

    st.write(
        "EMNIST Balanced"
    )

    st.subheader("Model")

    st.write(
        "Convolutional Neural Network (CNN)"
    )

    st.subheader("Classes")

    st.write(
        "47 handwritten character classes"
    )

    st.subheader("Input Size")

    st.write(
        "28 × 28 grayscale image"
    )

    st.subheader("Test Accuracy")

    st.metric(
        "Accuracy",
        "89.08%"
    )

    st.divider()

    st.subheader("Training Data")

    st.write(
        "Training Samples"
    )

    st.write(
        "112,800"
    )

    st.write(
        "Testing Samples"
    )

    st.write(
        "18,800"
    )

    st.divider()

    st.subheader("How It Works")

    st.write(
        "1. Upload or draw a character."
    )

    st.write(
        "2. The image is converted to grayscale."
    )

    st.write(
        "3. The image is resized to 28 × 28."
    )

    st.write(
        "4. EMNIST orientation is applied."
    )

    st.write(
        "5. The CNN predicts the character."
    )

    st.divider()

    st.subheader("Supported Input")

    st.write(
        "• Digits 0–9"
    )

    st.write(
        "• Uppercase letters A–Z"
    )

    st.write(
        "• Supported lowercase EMNIST characters"
    )

    st.divider()

    st.caption(
        "Built with TensorFlow, Keras, "
        "Streamlit and EMNIST Balanced."
    )

# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

if "input_image" not in st.session_state:
    st.session_state.input_image = None

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


# ============================================================
# EXACT MODEL PREPROCESSING
# ============================================================

def preprocess_character(image):

    if image is None:
        return None

    image = image.convert("L")

    img = np.array(
        image,
        dtype=np.uint8
    )

    # White background + black character
    # ->
    # Black background + white character

    img = 255 - img

    pil_image = Image.fromarray(img)

    pil_image = pil_image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    img = np.array(
        pil_image,
        dtype=np.float32
    )

    # Match EMNIST TFDS orientation

    img = np.transpose(img)

    # Same normalization used during training

    img = img / 255.0

    # Add channel dimension

    img = np.expand_dims(
        img,
        axis=-1
    )

    # Add batch dimension

    img = np.expand_dims(
        img,
        axis=0
    )

    return img.astype(
        np.float32
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_character(image):

    processed = preprocess_character(
        image
    )

    if processed is None:

        return (
            None,
            0.0,
            None,
            None
        )

    predictions = model.predict(
        processed,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    )

    predicted_class = class_names[
        predicted_index
    ]

    return (
        predicted_class,
        confidence,
        predictions,
        processed
    )


# ============================================================
# RESET RESULT
# ============================================================

def clear_result():

    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.probabilities = None
    st.session_state.processed_image = None
    st.session_state.input_image = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "Handwritten Character Recognition"
)

st.markdown(
    '<div class="app-subtitle">'
    'Draw or upload a single handwritten character '
    'and let the CNN identify it.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "Test Accuracy",
        "89.08%"
    )

with m2:

    st.metric(
        "Dataset",
        "EMNIST"
    )

with m3:

    st.metric(
        "Classes",
        "47"
    )

with m4:

    st.metric(
        "Architecture",
        "CNN"
    )


st.divider()


# ============================================================
# INPUT / RESULT
# ============================================================

input_column, result_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# CHARACTER INPUT
# ============================================================

with input_column:

    st.header(
        "Character Input"
    )

    # ========================================================
    # TABS
    # ========================================================

    tab_upload, tab_draw = st.tabs(
        [
            "Upload Image",
            "Draw Character"
        ]
    )


    # ========================================================
    # UPLOAD IMAGE TAB
    # ========================================================

    with tab_upload:

        st.caption(
            "Upload an image containing one handwritten character."
        )

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp"
            ],
            key="uploaded_character"
        )

        if uploaded_file is not None:

            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGBA")

            st.image(
                uploaded_image,
                caption="Uploaded Character",
                width=300
            )

            upload_recognize = st.button(
                "Recognize Uploaded Image",
                type="primary",
                use_container_width=True,
                key="upload_recognize"
            )

            if upload_recognize:

                (
                    predicted_class,
                    confidence,
                    probabilities,
                    processed
                ) = predict_character(
                    uploaded_image
                )

                st.session_state.prediction = (
                    predicted_class
                )

                st.session_state.confidence = (
                    confidence
                )

                st.session_state.probabilities = (
                    probabilities
                )

                st.session_state.processed_image = (
                    processed
                )

                st.session_state.input_image = (
                    uploaded_image
                )


    # ========================================================
    # DRAW CHARACTER TAB
    # ========================================================

    with tab_draw:

        st.caption(
            "Draw one character clearly in the center."
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=15,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=350,
            width=350,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            return_image_data=True,
        )

        st.write("")

        draw_recognize, draw_clear = st.columns(2)

        with draw_recognize:

            draw_button = st.button(
                "Recognize Drawing",
                type="primary",
                use_container_width=True,
                key="draw_recognize"
            )

        with draw_clear:

            clear_button = st.button(
                "Clear Canvas",
                use_container_width=True,
                key="draw_clear"
            )

        if clear_button:

            clear_result()

            st.session_state.canvas_key += 1

            st.rerun()

        if draw_button:

            if canvas_result.image_data is None:

                st.warning(
                    "Please draw a character first."
                )

            else:

                canvas_array = (
                    canvas_result
                    .image_data
                    .astype(np.uint8)
                )

                rgb = canvas_array[:, :, :3]

                darkness = (
                    255 - rgb
                ).mean()

                if darkness < 1:

                    st.warning(
                        "The canvas is empty."
                    )

                else:

                    input_image = Image.fromarray(
                        canvas_array,
                        mode="RGBA"
                    )

                    (
                        predicted_class,
                        confidence,
                        probabilities,
                        processed
                    ) = predict_character(
                        input_image
                    )

                    st.session_state.prediction = (
                        predicted_class
                    )

                    st.session_state.confidence = (
                        confidence
                    )

                    st.session_state.probabilities = (
                        probabilities
                    )

                    st.session_state.processed_image = (
                        processed
                    )

                    st.session_state.input_image = (
                        input_image
                    )


# ============================================================
# RECOGNITION RESULT
# ============================================================

with result_column:

    st.header(
        "Recognition Result"
    )

    if st.session_state.prediction is None:

        with st.container(border=True):

            st.subheader(
                "Ready to recognize"
            )

            st.write(
                "Draw or upload a single character "
                "to get started."
            )

            st.caption(
                "The CNN analyzes the character using "
                "the EMNIST Balanced model."
            )

    else:

        prediction = (
            st.session_state.prediction
        )

        confidence = (
            st.session_state.confidence
        )

        probabilities = (
            st.session_state.probabilities
        )


        # ----------------------------------------------------
        # MAIN RESULT
        # ----------------------------------------------------

        with st.container(border=True):

            st.caption(
                "PREDICTED CHARACTER"
            )

            st.markdown(
                f'<div class="prediction-letter">'
                f'{prediction}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.write(
                "Confidence"
            )

            st.progress(
                confidence
            )

            st.markdown(
                f'<div class="confidence-value">'
                f'{confidence * 100:.2f}%'
                f'</div>',
                unsafe_allow_html=True
            )


        st.write("")


        # ----------------------------------------------------
        # CONFIDENCE MESSAGE
        # ----------------------------------------------------

        if confidence >= 0.90:

            st.success(
                "High confidence prediction."
            )

        elif confidence >= 0.70:

            st.info(
                "Moderate confidence prediction."
            )

        else:

            st.warning(
                "Low confidence prediction. "
                "Try a clearer character."
            )


        # ----------------------------------------------------
        # TOP PREDICTIONS
        # ----------------------------------------------------

        st.subheader(
            "Top Predictions"
        )

        top_indices = np.argsort(
            probabilities
        )[-5:][::-1]

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            character = class_names[
                int(index)
            ]

            probability = float(
                probabilities[index]
            )

            c1, c2, c3 = st.columns(
                [0.4, 0.8, 3]
            )

            with c1:

                st.write(
                    f"**{rank}**"
                )

            with c2:

                st.write(
                    f"**{character}**"
                )

            with c3:

                st.progress(
                    probability
                )

                st.caption(
                    f"{probability * 100:.2f}%"
                )


# ============================================================
# MODEL INPUT PREVIEW
# ============================================================

if st.session_state.processed_image is not None:

    st.divider()

    st.header(
        "Model Input Preview"
    )

    st.caption(
        "The exact 28 × 28 image passed to the CNN."
    )

    processed = (
        st.session_state.processed_image
    )

    model_input = processed[
        0,
        :,
        :,
        0
    ]

    preview_column, info_column = st.columns(
        [1, 2]
    )

    with preview_column:

        st.image(
            model_input,
            width=220,
            clamp=True
        )

    with info_column:

        st.write(
            "**Input shape**"
        )

        st.code(
            str(processed.shape)
        )

        st.write(
            "**Expected**"
        )

        st.code(
            "(1, 28, 28, 1)"
        )

        st.write(
            "**Normalization**"
        )

        st.code(
            "float32 / 255.0"
        )

        st.write(
            "**EMNIST orientation**"
        )

        st.code(
            "Transpose applied"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header(
    "Model Information"
)

info1, info2, info3, info4 = st.columns(4)

with info1:

    st.metric(
        "Dataset",
        "EMNIST Balanced"
    )

with info2:

    st.metric(
        "Training Samples",
        "112,800"
    )

with info3:

    st.metric(
        "Test Samples",
        "18,800"
    )

with info4:

    st.metric(
        "Test Accuracy",
        "89.08%"
    )


# ============================================================
# SUPPORTED CHARACTERS
# ============================================================

with st.expander(
    "View Supported Characters"
):

    st.write(
        " • ".join(class_names)
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Handwritten Character Recognition • "
    "CNN • EMNIST Balanced"
)