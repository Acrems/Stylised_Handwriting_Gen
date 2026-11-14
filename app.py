import os
import sys
import tempfile
import uuid

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# Make relative paths (./assets, ./weights) work no matter where Streamlit is launched from
APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import numpy as np
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stylised Handwriting Generator", page_icon="✍️", layout="wide")

ASSET_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DEFAULT_WEIGHTS = "./weights/model_weights.h5"


# --------------------------------------------------------------------------
# Cached heavy resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading TensorFlow and model weights (first run only)...")
def load_models(weights_path: str, channels: int, num_attlayers: int):
    import tensorflow as tf
    import nn
    import utils

    c1 = channels
    c2 = c1 * 3 // 2
    c3 = c1 * 2

    style_extractor = nn.StyleExtractor()
    model = nn.DiffusionWriter(num_layers=num_attlayers, c1=c1, c2=c2, c3=c3)

    # The model must be called once before weights can be loaded
    _stroke = tf.random.normal([1, 400, 2])
    _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
    _noise = tf.random.uniform([1, 1])
    _style_vector = tf.random.normal([1, 14, 1280])
    _ = model(_stroke, _text, _noise, _style_vector)
    model.load_weights(weights_path)

    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()
    return model, style_extractor, tokenizer, beta_set


def list_assets():
    if not os.path.isdir("./assets"):
        return []
    return sorted(f for f in os.listdir("./assets") if f.lower().endswith(ASSET_EXTS))


def generate(text, writer_path, opts, weights_path, channels, num_attlayers):
    import tensorflow as tf
    import utils
    import preprocessing
    import style_controller

    model, style_extractor, tokenizer, beta_set = load_models(weights_path, channels, num_attlayers)

    if opts["seed"] is not None:
        tf.random.set_seed(int(opts["seed"]))
        np.random.seed(int(opts["seed"]))

    writer_img = tf.expand_dims(preprocessing.read_img(writer_path, 96), 0)
    style_vector = style_extractor(writer_img)

    # Multi-line text: pass None so every line gets its own length-based timesteps
    if "\n" in text:
        time_steps = None
    elif opts["seqlen"]:
        time_steps = int(opts["seqlen"])
        time_steps = time_steps - (time_steps % 8) + 8
    else:
        time_steps = len(text) * 16
        time_steps = time_steps - (time_steps % 8) + 8

    out_path = os.path.join(tempfile.gettempdir(), f"hw_{uuid.uuid4().hex}.png")

    style_config = style_controller.StyleConfig(min_linewidth=0.5, max_linewidth=5.0)
    utils.run_batch_inference(
        model,
        beta_set,
        text,
        style_vector,
        tokenizer=tokenizer,
        time_steps=time_steps,
        diffusion_mode=opts["diffmode"],
        show_samples=False,
        path=out_path,
        char_spacing=opts["char_spacing"],
        word_spacing=opts["word_spacing"],
        italics=opts["italics"],
        boldness=opts["boldness"],
        text_size=opts["text_size"],
        line_spacing=opts["line_spacing"],
        width=opts["width"],
        height=opts["height"],
        autowrap=1 if opts["autowrap"] else 0,
        fit_overflow=opts["fit_overflow"],
        style_config=style_config,
    )
    plt.close("all")

    with open(out_path, "rb") as f:
        png_bytes = f.read()
    try:
        os.remove(out_path)
    except OSError:
        pass
    return png_bytes


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("✍️ Stylised Handwriting Generator")
st.caption("Diffusion-based handwriting synthesis with style controls for spacing, slant, boldness, size and layout.")

with st.sidebar:
    st.header("Writer style")
    source_mode = st.radio(
        "Handwriting source",
        ["Upload an image", "Choose from assets", "Random from assets"],
    )

    uploaded = None
    chosen_asset = None
    assets = list_assets()

    if source_mode == "Upload an image":
        uploaded = st.file_uploader(
            "Upload a sample of the handwriting (one line of text works best)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
        )
        if uploaded is not None:
            st.image(uploaded, caption="Writer sample", use_container_width=True)
    elif source_mode == "Choose from assets":
        if assets:
            chosen_asset = st.selectbox("Asset", assets)
            st.image(os.path.join("./assets", chosen_asset), use_container_width=True)
        else:
            st.warning("No images found in ./assets")
    else:
        if not assets:
            st.warning("No images found in ./assets")

    st.header("Style controls")
    char_spacing = st.slider("Character spacing", -1.0, 1.0, 0.0, 0.05)
    word_spacing = st.slider("Word spacing", -1.0, 1.0, 0.0, 0.05)
    italics = st.slider("Italics (slant)", -1.0, 1.0, 0.0, 0.05)
    boldness = st.slider(
        "Boldness", -1.0, 1.0, 0.0, 0.1,
        help="Controls rendered pen thickness. Press Generate handwriting to apply the selected value.",
    )
    text_size = st.slider("Text size", -1.0, 1.0, 0.0, 0.05)
    line_spacing = st.slider("Line spacing (multiline)", -1.0, 1.0, 0.0, 0.05)

    st.header("Layout")
    autowrap = st.checkbox("Auto-wrap words", value=False)
    fit_overflow = st.checkbox("Shrink to fit if text overflows canvas", value=False)
    c1, c2 = st.columns(2)
    width = c1.number_input("Width (px, 0 = auto)", min_value=0, max_value=4000, value=0, step=50)
    height = c2.number_input("Height (px, 0 = auto)", min_value=0, max_value=4000, value=0, step=50)

    with st.expander("Advanced"):
        diffmode = st.selectbox(
            "Diffusion mode", ["new", "standard"],
            help="'new' = Eq. 12 from the paper (default). 'standard' = Eq. 9.",
        )
        seqlen = st.number_input(
            "Sequence length (0 = 16 × text length)", min_value=0, max_value=8000, value=0, step=8,
            help="Single-line text only. Multi-line text sizes each line automatically.",
        )
        use_seed = st.checkbox("Fix random seed (reproducible output)")
        seed = st.number_input("Seed", min_value=0, value=42, step=1, disabled=not use_seed)
        weights_path = st.text_input("Weights path", DEFAULT_WEIGHTS)
        channels = st.number_input("Channels (match training)", min_value=32, max_value=512, value=128, step=32)
        num_attlayers = st.number_input("Attention layers (match training)", min_value=1, max_value=8, value=2, step=1)

st.subheader("Text")
text = st.text_area(
    "Text to write (press Enter for a new line)",
    value="Generating text",
    height=160,
)
st.caption(
    "Supported characters: letters, digits and . ? ! , ' \" - and space. "
    "Anything else is replaced with '_'."
)

go = st.button("Generate handwriting", type="primary")

if go:
    clean_text = text.replace("\r\n", "\n").strip("\n")
    if not clean_text.strip():
        st.error("Please enter some text.")
        st.stop()

    if not os.path.exists(weights_path):
        st.error(f"Weights file not found at `{weights_path}`.")
        st.stop()

    # Resolve the writer image
    tmp_upload_path = None
    if source_mode == "Upload an image":
        if uploaded is None:
            st.error("Please upload a handwriting image in the sidebar.")
            st.stop()
        suffix = os.path.splitext(uploaded.name)[1] or ".png"
        tmp_upload_path = os.path.join(tempfile.gettempdir(), f"writer_{uuid.uuid4().hex}{suffix}")
        with open(tmp_upload_path, "wb") as f:
            f.write(uploaded.getbuffer())
        writer_path = tmp_upload_path
    elif source_mode == "Choose from assets":
        if not chosen_asset:
            st.error("No asset selected.")
            st.stop()
        writer_path = os.path.join("./assets", chosen_asset)
    else:
        if not assets:
            st.error("No images in ./assets to pick from.")
            st.stop()
        writer_path = os.path.join("./assets", assets[np.random.randint(0, len(assets))])

    opts = dict(
        char_spacing=char_spacing,
        word_spacing=word_spacing,
        italics=italics,
        boldness=boldness,
        text_size=text_size,
        line_spacing=line_spacing,
        autowrap=autowrap,
        fit_overflow=fit_overflow,
        width=int(width) if width else None,
        height=int(height) if height else None,
        diffmode=diffmode,
        seqlen=int(seqlen) if seqlen else None,
        seed=int(seed) if use_seed else None,
    )

    try:
        with st.spinner("Running diffusion (60 denoising steps per line)... this can take a while on CPU."):
            png = generate(clean_text, writer_path, opts, weights_path, int(channels), int(num_attlayers))
        st.session_state["result_png"] = png
    except Exception as e:  # surface errors in the UI instead of a blank page
        st.exception(e)
    finally:
        if tmp_upload_path and os.path.exists(tmp_upload_path):
            try:
                os.remove(tmp_upload_path)
            except OSError:
                pass

if "result_png" in st.session_state:
    st.subheader("Result")
    st.image(st.session_state["result_png"], use_container_width=True)
    st.download_button(
        "Download PNG",
        data=st.session_state["result_png"],
        file_name="handwriting.png",
        mime="image/png",
    )
