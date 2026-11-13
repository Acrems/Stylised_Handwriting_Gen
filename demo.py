"""
Interactive / CLI Demonstration Script for Mathematical Handwriting Style Controllers.

Allows users to specify:
- Input text string
- Character spacing (-1.0 to +1.0)
- Word spacing (-1.0 to +1.0)
- Italics (slant) (-1.0 to +1.0)
- Boldness (thickness) (-1.0 to +1.0)

Also provides a --compare flag to generate and visually compare the original handwriting
(all controls = 0) side-by-side with the styled handwriting.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import utils
import nn
import preprocessing
import style_controller


def parse_arguments():
    parser = argparse.ArgumentParser(description="Demonstrate Mathematical Handwriting Style Controllers.")
    parser.add_argument('--text', type=str, default='Generative AI Style Control',
                        help='Text to generate (default: "Generative AI Style Control")')
    parser.add_argument('--char_spacing', type=float, default=0.0,
                        help='Character spacing control [-1.0 to +1.0] (default: 0.0)')
    parser.add_argument('--word_spacing', type=float, default=0.0,
                        help='Word spacing control [-1.0 to +1.0] (default: 0.0)')
    parser.add_argument('--italics', type=float, default=0.0,
                        help='Italics (slant) control [-1.0 to +1.0] (default: 0.0)')
    parser.add_argument('--boldness', type=float, default=0.0,
                        help='Boldness (stroke width) control [-1.0 to +1.0] (default: 0.0)')
    parser.add_argument('--writersource', type=str, default=None,
                        help='Path to writer style image (default: random from ./assets)')
    parser.add_argument('--weights', type=str, default='./weights/model_weights.h5',
                        help='Path to model weights file')
    parser.add_argument('--output', type=str, default='demo_output.png',
                        help='Path for output image (default: demo_output.png)')
    parser.add_argument('--compare', action='store_true',
                        help='Generate side-by-side comparison between original and styled handwriting')
    parser.add_argument('--show', action='store_true',
                        help='Display sample in a GUI window')
    return parser.parse_args()


def render_to_axis(ax, strokes, linewidth, title=None):
    """Renders strokes to a matplotlib axis using the existing show() semantics."""
    pos = style_controller.relative_to_absolute(strokes)
    prev_ind = 0
    for ind, val in enumerate(strokes[:, 2]):
        if val > 0.5:
            if ind > prev_ind:
                ax.plot(pos[prev_ind:ind, 0], pos[prev_ind:ind, 1], color='black', linewidth=linewidth)
            prev_ind = ind
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=12, pad=10)


def create_comparison_plot(orig_strokes, orig_lw, mod_strokes, mod_lw, text, args, save_path):
    """Creates a high-quality vertical comparison figure showing Original vs Modified."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))

    render_to_axis(
        ax1,
        orig_strokes,
        orig_lw,
        title=f'Original Handwriting (All Controls = 0.0)\nText: "{text}"'
    )

    style_desc = (f'Char Spacing: {args.char_spacing:+.2f} | '
                  f'Word Spacing: {args.word_spacing:+.2f} | '
                  f'Italics: {args.italics:+.2f} | '
                  f'Boldness: {args.boldness:+.2f}')

    render_to_axis(
        ax2,
        mod_strokes,
        mod_lw,
        title=f'Modified Handwriting ({style_desc})'
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[DEMO] Side-by-side comparison saved to: {save_path}")


def main():
    args = parse_arguments()

    if not args.show:
        matplotlib.use('Agg')

    print("=" * 70)
    print("MATHEMATICAL HANDWRITING STYLE CONTROLLERS - DEMO")
    print("=" * 70)
    print(f"Input text:           '{args.text}'")
    print(f"Character spacing:    {args.char_spacing:+.2f} (range: -1.0 to +1.0)")
    print(f"Word spacing:         {args.word_spacing:+.2f} (range: -1.0 to +1.0)")
    print(f"Italics:              {args.italics:+.2f} (range: -1.0 to +1.0)")
    print(f"Boldness:             {args.boldness:+.2f} (range: -1.0 to +1.0)")
    print(f"Comparison mode:      {args.compare}")
    print("=" * 70)

    # 1. Select writer source
    if args.writersource is None:
        assetdir = [f for f in os.listdir('./assets') if f.endswith(('.png', '.tif', '.jpg', '.jpeg'))]
        sourcename = os.path.join('./assets', assetdir[np.random.randint(0, len(assetdir))])
    else:
        sourcename = args.writersource
    print(f"[DEMO] Using writer source: {sourcename}")

    # 2. Setup diffusion pipeline
    import tensorflow as tf
    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()

    timesteps = len(args.text) * 16
    timesteps = timesteps - (timesteps % 8) + 8

    print("[DEMO] Loading diffusion model...")
    style_extractor = nn.StyleExtractor()
    model = nn.DiffusionWriter(num_layers=2, c1=128, c2=192, c3=256)

    _stroke = tf.random.normal([1, 400, 2])
    _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
    _noise = tf.random.uniform([1, 1])
    _style_vector = tf.random.normal([1, 14, 1280])
    _ = model(_stroke, _text, _noise, _style_vector)
    model.load_weights(args.weights)

    writer_img = tf.expand_dims(preprocessing.read_img(sourcename, 96), 0)
    style_vector = style_extractor(writer_img)

    # 3. Generate raw handwriting from diffusion model
    print("[DEMO] Generating base handwriting via diffusion model...")
    raw_strokes = utils.run_batch_inference(
        model,
        beta_set,
        args.text,
        style_vector,
        tokenizer=tokenizer,
        time_steps=timesteps,
        diffusion_mode='new',
        show_samples=False,
        path=None,
        char_spacing=0.0,
        word_spacing=0.0,
        italics=0.0,
        boldness=0.0
    )[0]

    config = style_controller.StyleConfig()
    orig_lw = config.default_linewidth

    # 4. Apply mathematical style controllers
    print("[DEMO] Applying mathematical style controllers...")
    mod_strokes, mod_lw = style_controller.apply_all_styles(
        raw_strokes,
        args.text,
        char_spacing=args.char_spacing,
        word_spacing=args.word_spacing,
        italics=args.italics,
        boldness=args.boldness,
        config=config
    )

    # 5. Save outputs
    # Save styled handwriting
    utils.show(mod_strokes, name=args.output, show_output=args.show, linewidth=mod_lw)
    print(f"[DEMO] Saved styled output to: {args.output}")

    if args.compare:
        orig_path = 'demo_original.png'
        comp_path = 'demo_comparison.png'
        utils.show(raw_strokes, name=orig_path, show_output=False, linewidth=orig_lw)
        create_comparison_plot(raw_strokes, orig_lw, mod_strokes, mod_lw, args.text, args, comp_path)

    print("[DEMO] Demonstration completed successfully.")


if __name__ == '__main__':
    main()
