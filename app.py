import streamlit as st

# Must be the first Streamlit call
st.set_page_config(
    page_title="Handwriting Style Transfer",
    page_icon="✍️",
    layout="wide",
)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU; remove if GPU available

import io
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gdown

# ------------------------------------------------------------------
# Model weights download (Google Drive)
# ------------------------------------------------------------------
WEIGHTS_PATH = "./weights/model_weights.h5"
WEIGHTS_FILE_ID = "1U1h_sfTx2F51vmBGnELfZvKenospyITo"


def ensure_weights():
    if os.path.exists(WEIGHTS_PATH):
        return
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with st.spinner("Downloading model weights (first run only)…"):
        try:
            gdown.download(
                id=WEIGHTS_FILE_ID,
                output=WEIGHTS_PATH,
                quiet=False,
            )
        except Exception as e:
            st.error(
                "Could not download model weights from Google Drive.\n\n"
                f"Details: {e}\n\n"
                "Make sure the file is publicly shared and that the ID is correct."
            )
            st.stop()


# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading diffusion model…")
def load_model():
    import tensorflow as tf
    import nn

    model = nn.DiffusionWriter(num_layers=2, c1=128, c2=192, c3=256)
    # Warm-up call so the model builds variables before loading
    _stroke = tf.random.normal([1, 400, 2])
    _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
    _noise = tf.random.uniform([1, 1])
    _style_vector = tf.random.normal([1, 14, 1280])
    _ = model(_stroke, _text, _noise, _style_vector)
    model.load_weights(WEIGHTS_PATH)
    return model


@st.cache_resource(show_spinner="Loading style extractor…")
def load_style_extractor():
    import nn
    return nn.StyleExtractor()


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def read_writer_image(file_obj):
    import preprocessing
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(file_obj.getvalue())
        tmp_path = tmp.name
    try:
        img = preprocessing.read_img(tmp_path, 96)
    finally:
        os.unlink(tmp_path)
    return img


def default_writer_image():
    import preprocessing
    if os.path.isdir("./assets"):
        assets = [
            f for f in os.listdir("./assets")
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        if assets:
            return preprocessing.read_img(os.path.join("./assets", assets[0]), 96)
    # Blank white canvas fallback
    return np.full((96, 400, 1), 255, dtype=np.uint8)


def strokes_to_png(strokes, linewidth=1.5):
    """Exact replica of utils.show() rendering logic."""
    pos = np.cumsum(strokes[:, :2], axis=0)
    prev_ind = 0
    W, H = np.max(pos, axis=0) - np.min(pos, axis=0)
    
    # Match local utils.show logic exactly
    scale = 1
    fig_w = scale * W / H if H > 0 else scale
    fig_h = scale

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for ind, val in enumerate(strokes[:, 2]):
        if val > 0.5:
            if ind > prev_ind:
                ax.plot(
                    pos[prev_ind:ind, 0],
                    pos[prev_ind:ind, 1],
                    color="black",
                    linewidth=linewidth,
                )
            prev_ind = ind
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def run_generation(
    text, style_vector, char_spacing, word_spacing, italics, boldness, seqlen=None
):
    import tensorflow as tf
    import utils
    import style_controller

    model = load_model()
    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()

    if not seqlen or seqlen <= 0:
        seqlen = len(text) * 16
    seqlen = seqlen - (seqlen % 8) + 8

    raw = utils.run_batch_inference(
        model,
        beta_set,
        text,
        style_vector,
        tokenizer=tokenizer,
        time_steps=seqlen,
        diffusion_mode="new",
        show_samples=False,
        path=None,
        char_spacing=0.0,
        word_spacing=0.0,
        italics=0.0,
        boldness=0.0,
    )[0]

    config = style_controller.StyleConfig()
    
    # We map the -3.0 to 3.0 sliders back into the safe -1.0 to 1.0 range
    # for the underlying math, but keep the UI responsive.
    safe_char = np.clip(char_spacing / 3.0, -1.0, 1.0)
    safe_word = np.clip(word_spacing / 3.0, -1.0, 1.0)
    safe_italics = np.clip(italics / 3.0, -1.0, 1.0)
    safe_boldness = np.clip(boldness / 3.0, -1.0, 1.0)

    styled, lw = style_controller.apply_all_styles(
        raw,
        text,
        char_spacing=safe_char,
        word_spacing=safe_word,
        italics=safe_italics,
        boldness=safe_boldness,
        config=config,
    )
    return styled, lw


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("✍️ Handwriting Style Transfer")

ensure_weights()

with st.sidebar:
    st.header("📝 Input")
    text = st.text_input("Text to generate", value="Hello World", max_chars=200)

    st.header("🎨 Style Controls")
    char_spacing = st.slider("Character spacing", -3.0, 3.0, 0.0, 0.1)
    word_spacing = st.slider("Word spacing", -3.0, 3.0, 0.0, 0.1)
    italics = st.slider("Italics (slant)", -3.0, 3.0, 0.0, 0.1)
    boldness = st.slider("Boldness", -3.0, 3.0, 0.0, 0.1)

    st.header("⚙️ Options")
    writer_file = st.file_uploader(
        "Writer style reference (optional)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )
    
    if not os.path.isdir("./assets") and writer_file is None:
        st.warning("⚠️ No `assets/` folder found. Please upload a reference image to get proper handwriting!")
        
    generate_btn = st.button("✨ Generate", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None

if generate_btn:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            with st.spinner("Generating handwriting — this can take a minute…"):
                style_extractor = load_style_extractor()

                if writer_file is not None:
                    writer_img = read_writer_image(writer_file)
                else:
                    writer_img = default_writer_image()

                import tensorflow as tf
                style_vec = style_extractor(tf.expand_dims(writer_img, 0))

                styled, lw = run_generation(
                    text,
                    style_vec,
                    char_spacing,
                    word_spacing,
                    italics,
                    boldness,
                )

                st.session_state.result = {
                    "mod_png": strokes_to_png(styled, linewidth=lw),
                    "text": text,
                }
        except Exception as e:
            st.error(f"Generation failed: {e}")

if st.session_state.result is not None:
    res = st.session_state.result
    st.subheader("Generated")
    st.image(res["mod_png"], use_container_width=True)
    st.download_button(
        "📥 Download image",
        data=res["mod_png"],
        file_name="handwriting.png",
        mime="image/png",
        use_container_width=True,
    )
else:
    st.info("Configure the controls on the left and click **✨ Generate** to create handwriting.")
