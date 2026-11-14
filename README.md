# Stylised Handwriting Generator

Generate handwritten images from text using a diffusion model conditioned on a writer sample. Adjust the rendered handwriting with controls for spacing, slant, stroke weight, text size, line spacing, and page layout.

## Features

- Streamlit interface for entering text, choosing a writer sample, adjusting controls, and downloading a PNG.
- Optional FastAPI WebSocket service for applications that need streamed generation progress.
- Command-line inference and style-controller demo scripts.
- Automatic word wrapping, explicit multiline input, and configurable output canvas dimensions.
- Cached model loading in the Streamlit app so the model is reused across reruns.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Main Streamlit application. |
| `api.py` | FastAPI service with `/health` and `/ws` endpoints. |
| `inference.py` | Command-line model inference. |
| `demo.py` | Command-line style-control demo and comparison output. |
| `style_controller.py` | Mathematical handwriting transforms, wrapping, and canvas sizing. |
| `nn.py` | Diffusion writer and style-extractor model definitions. |
| `preprocessing.py` | Image and IAM dataset preprocessing helpers. |
| `utils.py` | Inference loop, tokenization support, and stroke rendering. |
| `weights/model_weights.h5` | Trained handwriting diffusion model weights. |
| `weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5` | Style-extractor backbone weights. |
| `assets/` | Example writer samples used by the apps and CLI. |

## Requirements

The core ML dependencies are listed in `requirements.txt`. Its version constraints target a recent Python/TensorFlow environment (the file notes Python 3.14). TensorFlow installation depends on your Python version and platform; use a Python version for which a compatible TensorFlow wheel is available.

Create an environment and install the core packages:

```bash
python -m venv .venv
```

Activate it, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the UI and API packages as needed:

```bash
python -m pip install streamlit
```

```bash
python -m pip install fastapi "uvicorn[standard]"
```

The trained model files are expected under `weights/`; example writer images are under `assets/`. The repository includes both weight files and several sample images.

## Run the Streamlit app

From the repository root:

```bash
streamlit run app.py
```

The app opens in your browser. Enter the text, choose a writer sample in the sidebar, adjust style and layout controls, and select **Generate handwriting**. The first generation loads TensorFlow and the model weights; later generations reuse the cached models. Generation can take a while on CPU.

The writer sample can be uploaded, selected from `assets/`, or randomly selected from the bundled assets. When upload mode is selected, an image is required. A one-line sample with clear, dark handwriting on a light background is recommended.

### Streamlit controls

Style controls use normalized values from `-1` to `+1`; `0` is the neutral setting.

| Control | Effect |
| --- | --- |
| Character spacing | Expands or contracts spacing between characters. |
| Word spacing | Changes spacing between words. |
| Italics | Slants the handwriting. |
| Boldness | Changes rendered pen thickness. |
| Text size | Scales the writing. |
| Line spacing | Changes spacing between lines in multiline output. |
| Auto-wrap words | Wraps whole words to fit the chosen canvas width. |
| Width / height | Sets output dimensions in pixels; `0` selects automatic sizing. |
| Shrink to fit | Scales handwriting down when it exceeds the canvas. |

Boldness changes are applied when **Generate handwriting** is pressed. The Streamlit app uses a more visible stroke-width range than the default style-controller configuration.

## Run the FastAPI service

Install FastAPI and Uvicorn as above, then start the service from the repository root:

```bash
python api.py
```

The API listens on `http://localhost:8000`.

- `GET /health` reports service status and whether the model loaded.
- `WS /ws` accepts a JSON request, then an optional binary writer-image frame, and streams status/progress messages followed by the generated JPEG as a binary frame.

Example request JSON:

```json
{
  "text": "A note in my handwriting",
  "has_writer_image": false,
  "char_spacing": 0.0,
  "word_spacing": 0.0,
  "italics": 0.0,
  "boldness": 0.0,
  "text_size": 0.0,
  "line_spacing": 0.0,
  "autowrap": true,
  "width": 900,
  "height": null
}
```

If `has_writer_image` is `true`, send the image bytes as the next WebSocket frame after the JSON request. The API limits text to 200 characters, writer images to 8 MB, width to 100 to 1400 px, and height to 50 to 1400 px.

Check model startup at `http://localhost:8000/health`. A response with `"model_loaded": false` means the model failed to initialize; inspect the API process logs and confirm the files in `weights/` match the model configuration.

## Command-line inference

Generate an image with the bundled example images and default weights:

```bash
python inference.py --textstring "Generating text" --name output.png
```

Specify a writer sample and controls:

```bash
python inference.py \
  --textstring "A handwritten note" \
  --writersource assets/1.jpeg \
  --boldness 0.5 \
  --char_spacing 0.2 \
  --word_spacing 0.1 \
  --name output.png
```

Use `python inference.py --help` to see all options. Style controls are in the `-1` to `+1` range. Use `--width`, `--height`, `--line_spacing`, and `--autowrap 1` for page layout. To pass line breaks from a shell, use the literal `\\n` in `--textstring`.

## Style-controller demo

`demo.py` demonstrates the style controls and can save a side-by-side baseline comparison:

```bash
python demo.py --text "Generative AI Style Control" --boldness 0.6 --compare
```

Use `python demo.py --help` for all options. This script also loads the model and writer sample, so it requires valid weights and an image in `assets/` or a supplied `--writersource`.

## Training

`train.py` contains the training entry point. Training requires the IAM online-handwriting dataset in the directory structure expected by `preprocessing.py`; that dataset is not included here. Review `python train.py --help` and the preprocessing helpers before preparing a training run. Training can require substantial disk space, memory, and accelerator time.

## Troubleshooting

- **Model weights not found:** Run commands from the repository root and confirm both files exist in `weights/`.
- **Weights fail to load:** The model architecture settings must match the checkpoint. The included weights use the default architecture (`channels=128`, `num_attlayers=2`).
- **No writer images:** Upload a sample in Streamlit or add supported image files (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`) to `assets/` for asset selection and CLI defaults.
- **TensorFlow installation fails:** Check that your Python version and operating system have a compatible TensorFlow wheel, then choose a supported Python environment.
- **Generation is slow:** The diffusion loop performs 60 denoising steps per generated line. CPU inference is slower than accelerator-backed inference.

