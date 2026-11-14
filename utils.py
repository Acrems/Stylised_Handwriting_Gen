import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None
import matplotlib.pyplot as plt
import string
import pickle
import os

#notation clarification:
#we use the variable "alpha" for alpha_bar (cumprod 1-beta)
#the alpha in the paper is replaced with 1-beta
def explin(min, max, L):
    return tf.exp(tf.linspace(tf.math.log(min), tf.math.log(max), L))

def get_beta_set():
    beta_set = 0.02 + explin(1e-5, 0.4, 60)
    return beta_set
    
def show(strokes, name='', show_output=True, scale=1, linewidth=None,
         width=None, height=None, dpi=100, margin=None, fit_overflow=False):
    import style_controller
    import warnings

    canvas_w, canvas_h = style_controller.calculate_dynamic_canvas_dimensions(
        strokes,
        width=width,
        height=height
    )

    fig = plt.figure(figsize=(canvas_w / dpi, canvas_h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    pixels_per_unit = 15.0
    W_canvas = canvas_w / pixels_per_unit
    H_canvas = canvas_h / pixels_per_unit

    if margin is None:
        margin_x = 0.05 * W_canvas
        margin_y = 0.10 * H_canvas
    elif isinstance(margin, (int, float)):
        margin_x = float(margin)
        margin_y = float(margin)
    else:
        margin_x, margin_y = float(margin[0]), float(margin[1])

    W_avail = max(0.1, W_canvas - 2 * margin_x)
    H_avail = max(0.1, H_canvas - 2 * margin_y)

    positions = np.cumsum(strokes[:, :2], axis=0).T
    x_min, x_max = float(np.min(positions[0])), float(np.max(positions[0]))
    y_min, y_max = float(np.min(positions[1])), float(np.max(positions[1]))
    W_stroke = max(0.01, x_max - x_min)
    H_stroke = max(0.01, y_max - y_min)

    if W_stroke > W_avail or H_stroke > H_avail:
        if fit_overflow:
            scale_fit = min(W_avail / W_stroke, H_avail / H_stroke)
            warnings.warn(
                f"Handwriting dimensions ({W_stroke:.1f}x{H_stroke:.1f}) exceed available "
                f"canvas area ({W_avail:.1f}x{H_avail:.1f}). Uniformly scaled by {scale_fit:.3f} to fit canvas."
            )
            x_pos = x_min + scale_fit * (positions[0] - x_min)
            y_pos = y_max + scale_fit * (positions[1] - y_max)
            x_min, x_max = float(np.min(x_pos)), float(np.max(x_pos))
            y_min, y_max = float(np.min(y_pos)), float(np.max(y_pos))
        else:
            warnings.warn(
                f"Handwriting dimensions ({W_stroke:.1f}x{H_stroke:.1f}) exceed available "
                f"canvas area ({W_avail:.1f}x{H_avail:.1f}). Content will overflow canvas boundaries without cropping or shrinking."
            )
            x_pos = positions[0].copy()
            y_pos = positions[1].copy()
    else:
        x_pos = positions[0].copy()
        y_pos = positions[1].copy()

    x_pos = x_pos - x_min + margin_x
    y_pos = y_pos - y_max + (H_canvas - margin_y)

    plot_kwargs = {'color': 'black'}
    if linewidth is not None:
        plot_kwargs['linewidth'] = linewidth

    prev_ind = 0
    for ind, value in enumerate(strokes[:, 2]):
        if value > 0.5:
            if ind > prev_ind:
                ax.plot(x_pos[prev_ind:ind], y_pos[prev_ind:ind], **plot_kwargs)
            prev_ind = ind

    ax.set_xlim(0, W_canvas)
    ax.set_ylim(0, H_canvas)
    ax.set_aspect('equal')

    if name:
        save_path = name if name.endswith('.png') else name + '.png'
        plt.savefig(save_path, dpi=dpi)
    if show_output:
        plt.show()
    else:
        plt.close(fig)
    
def get_alphas(batch_size, alpha_set): 
    alpha_indices = tf.random.uniform([batch_size, 1], maxval=len(alpha_set) - 1, dtype=tf.int32)
    lower_alphas = tf.gather_nd(alpha_set, alpha_indices)
    upper_alphas = tf.gather_nd(alpha_set, alpha_indices+1)
    alphas = tf.random.uniform(lower_alphas.shape, maxval=1) * (upper_alphas - lower_alphas) 
    alphas += lower_alphas
    alphas = tf.reshape(alphas, [batch_size, 1, 1])
    return alphas

def standard_diffusion_step(xt, eps, beta, alpha, add_sigma=True):
    x_t_minus1 = (1 / tf.sqrt(1-beta)) * (xt - (beta * eps/tf.sqrt(1-alpha)))
    if add_sigma: x_t_minus1 += tf.sqrt(beta) * (tf.random.normal(xt.shape))
    return x_t_minus1
   
def new_diffusion_step(xt, eps, beta, alpha, alpha_next):
    x_t_minus1 = (xt - tf.sqrt(1-alpha)*eps) / tf.sqrt(1-beta)
    x_t_minus1 += tf.random.normal(xt.shape) * tf.sqrt(1-alpha_next)
    return x_t_minus1
    
def run_batch_inference(model, beta_set, text, style, tokenizer=None, time_steps=480, 
                        diffusion_mode='new', show_every=None, show_samples=True, path=None,
                        char_spacing=0.0, word_spacing=0.0, italics=0.0, boldness=0.0,
                        text_size=0.0, line_spacing=0.0, width=None, height=None,
                        autowrap=0, fit_overflow=False, style_config=None):
    if tf is None:
        raise ImportError(
            "TensorFlow is required to run diffusion inference, but is not installed in the current environment."
        )

    should_autowrap = (autowrap == 1)

    import style_controller
    if style_config is None:
        style_config = style_controller.StyleConfig()

    if width is not None:
        avail_w_coords = (width / 15.0) * (1.0 - 2.0 * 0.05)
    else:
        avail_w_coords = (style_controller.DEFAULT_MAX_CANVAS_WIDTH / 15.0) * (1.0 - 2.0 * 0.05)

    if isinstance(text, str) and '\n' in text:
        raw_lines = text.split('\n')
        line_strokes_list = []
        last_lw = style_config.default_linewidth

        for line_str in raw_lines:
            trimmed = line_str.strip()
            if not trimmed:
                line_strokes_list.append(None)
                continue

            line_tokenized = tf.constant([tokenizer.encode(line_str)])
            line_timesteps = len(line_str) * 16 if time_steps is None else (time_steps if time_steps != 480 else len(line_str) * 16)
            line_timesteps = line_timesteps - (line_timesteps % 8) + 8
            line_timesteps = max(16, line_timesteps)

            L = len(beta_set)
            alpha_set = tf.math.cumprod(1 - beta_set)
            x_line = tf.random.normal([1, line_timesteps, 2])

            for step_i in range(L - 1, -1, -1):
                alpha = alpha_set[step_i] * tf.ones([1, 1, 1])
                beta = beta_set[step_i] * tf.ones([1, 1, 1])
                a_next = alpha_set[step_i - 1] if step_i > 1 else 1.0
                model_out, pen_lifts, att = model(x_line, line_tokenized, tf.sqrt(alpha), style)
                if diffusion_mode == 'standard':
                    x_line = standard_diffusion_step(x_line, model_out, beta, alpha, add_sigma=bool(step_i))
                else:
                    x_line = new_diffusion_step(x_line, model_out, beta, alpha, a_next)

            x_line = tf.concat([x_line, pen_lifts], axis=-1)
            raw_line_strokes = x_line.numpy()[0]

            mod_line_strokes, last_lw = style_controller.apply_all_styles(
                raw_line_strokes,
                line_str,
                char_spacing=char_spacing,
                word_spacing=word_spacing,
                italics=italics,
                boldness=boldness,
                text_size=text_size,
                config=style_config
            )

            if should_autowrap:
                wrapped_para, _ = style_controller.wrap_strokes_by_word(
                    mod_line_strokes,
                    line_str,
                    available_width_coords=avail_w_coords,
                    line_spacing=line_spacing,
                    config=style_config
                )
                line_strokes_list.append(wrapped_para)
            else:
                line_strokes_list.append(mod_line_strokes)

        combined_strokes, _ = style_controller.layout_multiline_strokes(
            line_strokes_list,
            line_spacing=line_spacing,
            config=style_config
        )

        show(
            combined_strokes,
            name=path,
            show_output=show_samples,
            linewidth=last_lw,
            width=width,
            height=height,
            fit_overflow=fit_overflow
        )
        return np.array([combined_strokes])

    raw_text = text if isinstance(text, (str, list)) else None
    if isinstance(text, str):
        text = tf.constant([tokenizer.encode(text)])
    elif isinstance(text, list) and isinstance(text[0], str):
        tmp = []
        for i in text:
            tmp.append(tokenizer.encode(i))
        text = tf.constant(tmp)

    bs = text.shape[0]
    L = len(beta_set)
    alpha_set = tf.math.cumprod(1- beta_set)
    x = tf.random.normal([bs, time_steps, 2])
    
    for i in range(L-1, -1, -1):
        alpha = alpha_set[i] * tf.ones([bs, 1, 1]) 
        beta = beta_set[i] * tf.ones([bs, 1, 1]) 
        a_next = alpha_set[i-1] if i>1 else 1.
        model_out, pen_lifts, att = model(x, text, tf.sqrt(alpha), style)
        if diffusion_mode == 'standard':
            x = standard_diffusion_step(x, model_out, beta, alpha, add_sigma=bool(i)) 
        else: 
            x = new_diffusion_step(x, model_out, beta, alpha, a_next)
        
        if show_every is not None:
            if i in show_every:
                plt.imshow(att[0][0])
                plt.show()

    x = tf.concat([x, pen_lifts], axis=-1)
    raw_strokes = x.numpy()

    final_strokes = []
    for i in range(bs):
        curr_text = raw_text if isinstance(raw_text, str) else (raw_text[i] if raw_text is not None else "")
        mod_strokes, lw = style_controller.apply_all_styles(
            raw_strokes[i],
            curr_text,
            char_spacing=char_spacing,
            word_spacing=word_spacing,
            italics=italics,
            boldness=boldness,
            text_size=text_size,
            config=style_config,
            attention_map=att[i].numpy() if 'att' in locals() else None
        )
        if should_autowrap and curr_text:
            mod_strokes, _ = style_controller.wrap_strokes_by_word(
                mod_strokes,
                curr_text,
                available_width_coords=avail_w_coords,
                line_spacing=line_spacing,
                config=style_config
            )
        final_strokes.append(mod_strokes)
        show(mod_strokes, scale=1, show_output=show_samples, name=path, linewidth=lw,
             width=width, height=height, fit_overflow=fit_overflow)

    return np.array(final_strokes)
    
def pad_stroke_seq(x, maxlength):
    if len(x) > maxlength or np.amax(np.abs(x)) > 15: return None
    zeros = np.zeros((maxlength - len(x), 2))
    ones = np.ones((maxlength - len(x), 1))
    padding = np.concatenate((zeros, ones), axis=-1)
    x = np.concatenate((x, padding)).astype('float32')
    return x

def pad_img(img, width, height):
    pad_len = width - img.shape[1]
    padding = np.full((height, pad_len, 1), 255, dtype=np.uint8)
    img = np.concatenate((img, padding), axis=1)
    return img
	
def preprocess_data(path, max_text_len, max_seq_len, img_width, img_height):
    with open(path, 'rb') as f:
        ds = pickle.load(f)
        
    strokes, texts, samples = [], [], []
    for x, text, sample in ds:
        if len(text) < max_text_len:
            x = pad_stroke_seq(x, maxlength=max_seq_len)
            zeros_text = np.zeros((max_text_len-len(text), ))
            text = np.concatenate((text, zeros_text))
            h, w, _ = sample.shape

            if x is not None and sample.shape[1] < img_width: 
                sample = pad_img(sample, img_width, img_height)
                strokes.append(x)
                texts.append(text)
                samples.append(sample)
    texts = np.array(texts).astype('int32')
    samples = np.array(samples)
    return strokes, texts, samples
    
def create_dataset(strokes, texts, samples, style_extractor, batch_size, buffer_size):    
    #we DO NOT SHUFFLE here, because we will shuffle later
    samples = tf.data.Dataset.from_tensor_slices(samples).batch(batch_size)
    for count, s in enumerate(samples):
        style_vec = style_extractor(s)
        style_vec = style_vec.numpy()
        if count==0: style_vectors = np.zeros((0, style_vec.shape[1], 1280))
        style_vectors = np.concatenate((style_vectors, style_vec), axis=0)
    style_vectors = style_vectors.astype('float32')
    
    dataset = tf.data.Dataset.from_tensor_slices((strokes, texts, style_vectors))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset
    
class Tokenizer:
    def __init__(self):
        self.tokens = {}
        self.chars = {}
        self.text = '_' + string.ascii_letters + string.digits + '.?!,\'\"- '
        self.numbers = np.arange(2, len(self.text)+2)
        self.create_dict()
        self.vocab_size = len(self.text)+2
    
    def create_dict(self):
        for char, token, in zip(self.text, self.numbers):
            self.tokens[char] = token
            self.chars[token] = char
        self.chars[0], self.chars[1] = ' ', '<end>' #only for decoding

    def encode(self, text):
        tokenized = []
        for char in text:
            if char in self.text: tokenized.append(self.tokens[char])
            else: tokenized.append(2) #unknown character is '_', which has index 2
         
        tokenized.append(1) #1 is the end of sentence character
        return tokenized
    
    def decode(self, tokens):
        if tf is not None and isinstance(tokens, tf.Tensor): tokens = tokens.numpy()
        text = [self.chars[token] for token in tokens]
        return ''.join(text)
