Code for the Paper "Diffusion Models for Handwriting Generation": https://arxiv.org/abs/2011.06704

Originally developed in Python 3.7 and TensorFlow 2.3.
Migrated and tested for Python 3.12 and TensorFlow 2.21.

To run the model, install required packages with 
pip install -r requirements.txt 

### Basic Inference:
Run `inference.py` and specify arguments as needed:
```bash
python inference.py --textstring "Your text here" --name sample.png
```

### Mathematical Handwriting Style Controllers (Python 3.12):
Four post-generation deterministic mathematical controllers are available:
1. `--char_spacing`: Character spacing control [-1.0, 1.0] (default: 0.0)
2. `--word_spacing`: Word spacing control [-1.0, 1.0] (default: 0.0)
3. `--italics`: Italics / slant shear control [-1.0, 1.0] (default: 0.0)
4. `--boldness`: Stroke thickness control [-1.0, 1.0] (default: 0.0)

Example:
```bash
python inference.py --textstring "Hello World" --char_spacing 0.5 --word_spacing 0.8 --italics 0.4 --boldness 0.6 --name styled.png
```

### Visual Demonstration & Side-by-Side Comparison:
To compare original (all controls = 0) vs styled handwriting side-by-side:
```bash
python demo.py --text "Generative AI" --char_spacing 0.4 --word_spacing 0.8 --italics 0.3 --boldness 0.5 --compare
```

### Automated Unit Test Suite:
Run the comprehensive test suite verifying zero-control identity, individual controls, and edge cases:
```bash
python test_style_controllers.py -v
```


To retrain model, run train.py, and specify arguments to change hyperparameters
All models will be saved in the ./weights directory


Before running the training script, download the following things from 
https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database

data/lineStrokes-all.tar.gz   -   the stroke xml for the online dataset
data/lineImages-all.tar.gz    -   the images for the offline dataset
ascii-all.tar.gz              -   the text labels for the dataset
extract these contents and put them in the ./data directory 
