"""
Mathematical Style Control Module for Diffusion-based Handwriting Generation.

This module provides deterministic mathematical transformations to control four
handwriting characteristics without altering the underlying diffusion model:
1. Character Spacing: Expands or contracts the horizontal distance between consecutive characters.
2. Word Spacing: Expands or contracts the horizontal distance between consecutive words.
3. Italics (Slant): Applies a horizontal shear transformation relative to a baseline reference.
4. Boldness: Adjusts the rendered stroke thickness in the rendering pipeline.

All controls accept normalized inputs in [-1.0, 1.0]:
- -1.0: Maximum decrease
-  0.0: Preserves original handwriting (exact zero-control identity)
- +1.0: Maximum increase
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np


@dataclass
class StyleConfig:
    """Central configuration for mathematical handwriting style controllers."""

    # Maximum horizontal displacement for character spacing (in coordinate units)
    max_char_displacement: float = 1.2

    # Maximum horizontal displacement for word spacing (in coordinate units)
    max_word_displacement: float = 5.0

    # Maximum shear factor for italics (tangent of slant angle, e.g. ~20 degrees)
    max_slant_factor: float = 0.35

    # Rendering thickness (linewidth) settings
    default_linewidth: float = 1.5
    min_linewidth: float = 0.6
    max_linewidth: float = 3.6

    # Percentile of y-coordinates used to estimate baseline reference
    baseline_percentile: float = 10.0

    # Minimum stroke span to be considered an active drawn stroke (filters stationary padding)
    active_stroke_span_threshold: float = 0.05

    # Horizontal proximity threshold for merging multi-stroke character components (e.g. crossbars)
    char_merge_gap_threshold: float = 0.25

    # Whether to clamp style control values to [-1.0, 1.0]
    clamp_controls: bool = True


# ==============================================================================
# Coordinate Conversion Utilities
# ==============================================================================

def relative_to_absolute(strokes: np.ndarray) -> np.ndarray:
    """
    Converts relative stroke coordinates (dx, dy) to absolute coordinates (x, y).

    Uses float64 for intermediate cumulative summation to prevent single-precision
    accumulation drift across long stroke sequences.

    Parameters:
        strokes: Array of shape (N, 2) or (N, 3) where [:, 0] is dx, [:, 1] is dy.

    Returns:
        pos: Array of shape (N, 2) with absolute coordinates [x, y].
    """
    return np.cumsum(strokes[:, :2].astype(np.float64), axis=0)


def absolute_to_relative(
    pos: np.ndarray,
    pen_lifts: np.ndarray,
    orig_dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Converts absolute coordinates back into relative stroke displacements (dx, dy).

    Mathematically:
        dx[0] = pos[0]
        dx[i] = pos[i] - pos[i - 1] for i >= 1

    Preserves pen lift states and original data type without loss of precision.

    Parameters:
        pos: Array of shape (N, 2) with absolute coordinates.
        pen_lifts: Array of shape (N, 1) or (N,) with pen lift probabilities/flags.
        orig_dtype: Numpy data type for the returned displacement values.

    Returns:
        strokes: Array of shape (N, 3) where [:, :2] are (dx, dy) and [:, 2] is pen lift.
    """
    dx = np.empty_like(pos)
    dx[0] = pos[0]
    dx[1:] = pos[1:] - pos[:-1]
    dx = dx.astype(orig_dtype)

    if pen_lifts.ndim == 1:
        pen_lifts = pen_lifts[:, np.newaxis]

    return np.concatenate([dx, pen_lifts.astype(orig_dtype)], axis=-1)


# ==============================================================================
# Stroke Segmentation and Boundary Detection
# ==============================================================================

def extract_active_segments(
    strokes: np.ndarray,
    config: Optional[StyleConfig] = None
) -> List[Dict[str, Any]]:
    """
    Partitions strokes into continuous active stroke segments bounded by pen lifts.

    A segment represents an unbroken drawn line where the pen remains down.
    Stationary padding points at sequence ends are filtered out using span threshold.

    Parameters:
        strokes: Array of shape (N, 3) containing [dx, dy, pen_lift].
        config: Optional StyleConfig instance.

    Returns:
        segments: List of dicts describing each active segment:
            - start: Start index in strokes array (inclusive)
            - end: End index in strokes array (exclusive)
            - x_min, x_max: Horizontal bounding box
            - y_min, y_max: Vertical bounding box
            - span: Diagonal geometric span (hypotenuse)
    """
    if config is None:
        config = StyleConfig()

    pos = relative_to_absolute(strokes)
    lift_indices = np.where(strokes[:, 2] > 0.5)[0]
    segments = []
    prev = 0

    for ind in lift_indices:
        if ind > prev:
            seg_pos = pos[prev:ind]
            x_min, x_max = float(seg_pos[:, 0].min()), float(seg_pos[:, 0].max())
            y_min, y_max = float(seg_pos[:, 1].min()), float(seg_pos[:, 1].max())
            span = float(np.hypot(x_max - x_min, y_max - y_min))

            if span > config.active_stroke_span_threshold:
                segments.append({
                    'start': prev,
                    'end': ind,
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'span': span
                })
        prev = ind

    return segments


def find_word_boundaries(
    strokes: np.ndarray,
    text: str,
    config: Optional[StyleConfig] = None,
    attention_map: Optional[np.ndarray] = None
) -> List[int]:
    """
    Identifies the stroke indices separating consecutive words in the generated handwriting.

    In handwriting, the transition between words is marked by a pen lift accompanied
    by a substantial forward horizontal displacement that is significantly larger than
    intra-word character transitions.

    Parameters:
        strokes: Array of shape (N, 3).
        text: Original input text string.
        config: Optional StyleConfig instance.
        attention_map: Optional cross-attention map from diffusion model.

    Returns:
        boundaries: List of stroke indices marking the start of word 1, word 2, ...
    """
    if config is None:
        config = StyleConfig()

    words = [w for w in text.split(' ') if w]
    num_words = len(words)
    if num_words <= 1:
        return []

    segments = extract_active_segments(strokes, config)
    if len(segments) < num_words:
        # Fallback if too few segments: divide evenly along active length
        active_len = segments[-1]['end'] if segments else len(strokes)
        return [int(active_len * (w + 1) / num_words) for w in range(num_words - 1)]

    # Compute horizontal gap between consecutive active segments
    gaps = []
    for i in range(len(segments) - 1):
        s_curr = segments[i]
        s_next = segments[i + 1]
        gap = s_next['x_min'] - s_curr['x_max']
        gaps.append((i, gap, s_next['start']))

    # The (num_words - 1) largest forward gaps correspond to inter-word spaces
    gaps.sort(key=lambda x: x[1], reverse=True)
    selected_gaps = gaps[:num_words - 1]

    # Sort chronologically by stroke sequence index
    selected_gaps.sort(key=lambda x: x[2])
    boundaries = [int(g[2]) for g in selected_gaps]
    return boundaries


def cluster_word_segments(
    word_segments: List[Dict[str, Any]],
    expected_chars: int,
    config: Optional[StyleConfig] = None
) -> List[List[Dict[str, Any]]]:
    """
    Groups stroke segments within a word into discrete character clusters.

    Preserves multi-stroke characters (e.g. 'H', 'E', 'F', 'i', 't') whose components
    overlap horizontally (crossbars, dots, stems), while separating horizontally distinct
    consecutive characters.

    Parameters:
        word_segments: List of active segment dicts belonging to one word.
        expected_chars: Number of characters in the corresponding word text.
        config: Optional StyleConfig instance.

    Returns:
        clusters: List of clusters, each containing one or more segment dicts.
    """
    if config is None:
        config = StyleConfig()

    if not word_segments:
        return []
    if len(word_segments) <= 1:
        return [[word_segments[0]]]

    # Step 1: Merge segments that horizontally overlap with previous segments
    clusters = [[word_segments[0]]]
    for s in word_segments[1:]:
        curr_cluster = clusters[-1]
        c_max = max(x['x_max'] for x in curr_cluster)

        # Overlapping horizontal bounding box indicates parts of the same character
        if s['x_min'] <= c_max + config.char_merge_gap_threshold:
            curr_cluster.append(s)
        else:
            clusters.append([s])

    # Step 2: If we produced more clusters than characters in the word,
    # merge adjacent clusters with the smallest horizontal distance.
    while len(clusters) > max(1, expected_chars):
        min_gap = float('inf')
        best_idx = 0
        for i in range(len(clusters) - 1):
            max_left = max(x['x_max'] for x in clusters[i])
            min_right = min(x['x_min'] for x in clusters[i + 1])
            gap = min_right - max_left
            if gap < min_gap:
                min_gap = gap
                best_idx = i
        clusters[best_idx].extend(clusters[best_idx + 1])
        clusters.pop(best_idx + 1)

    return clusters


# ==============================================================================
# Mathematical Style Controllers
# ==============================================================================

def apply_character_spacing(
    strokes: np.ndarray,
    text: str,
    char_spacing: float,
    config: Optional[StyleConfig] = None,
    attention_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Adjusts the horizontal distance between consecutive characters.

    Mathematical formulation:
        For character cluster c (0 <= c < C) within a word:
            Delta_x_char(c) = c * char_spacing * max_char_displacement
        Subsequent words receive an offset equal to the accumulated expansion of
        preceding words, ensuring that character spacing does not distort word spacing.

    Parameters:
        strokes: Array of shape (N, 3) representing (dx, dy, pen_lift).
        text: Corresponding text label.
        char_spacing: Control value in [-1.0, 1.0].
        config: Optional StyleConfig configuration.
        attention_map: Optional cross-attention map from diffusion model.

    Returns:
        modified_strokes: Array of shape (N, 3) with adjusted character spacing.
    """
    if config is None:
        config = StyleConfig()

    if config.clamp_controls:
        char_spacing = float(np.clip(char_spacing, -1.0, 1.0))

    if char_spacing == 0.0:
        return strokes.copy()

    words = [w for w in text.split(' ') if w]
    num_words = len(words)
    if num_words == 0:
        return strokes.copy()

    pos = relative_to_absolute(strokes)
    pen_lifts = strokes[:, 2:3]
    segments = extract_active_segments(strokes, config)
    word_boundaries = find_word_boundaries(strokes, text, config, attention_map)

    # Partition segments into words
    word_segments = [[] for _ in range(max(1, num_words))]
    for seg in segments:
        w_idx = 0
        for b in word_boundaries:
            if seg['start'] >= b:
                w_idx += 1
            else:
                break
        w_idx = min(w_idx, len(word_segments) - 1)
        word_segments[w_idx].append(seg)

    x_shifts = np.zeros(len(pos), dtype=np.float64)
    cumulative_expansion = 0.0
    last_active_stroke_end = segments[-1]['end'] if segments else len(pos)

    for w_idx in range(len(word_segments)):
        segs_in_word = word_segments[w_idx]
        expected_chars = len(words[w_idx]) if w_idx < len(words) else len(segs_in_word)
        char_clusters = cluster_word_segments(segs_in_word, expected_chars, config)
        num_clusters = len(char_clusters)

        base_word_shift = cumulative_expansion

        for c_idx, cluster in enumerate(char_clusters):
            c_shift = c_idx * char_spacing * config.max_char_displacement
            total_shift = base_word_shift + c_shift
            for seg in cluster:
                x_shifts[seg['start']:seg['end']] = total_shift

        if num_clusters > 1:
            word_expansion = (num_clusters - 1) * char_spacing * config.max_char_displacement
            cumulative_expansion += word_expansion

    # Trailing inactive padding receives the final shift
    if last_active_stroke_end < len(pos):
        x_shifts[last_active_stroke_end:] = cumulative_expansion

    pos[:, 0] += x_shifts
    return absolute_to_relative(pos, pen_lifts, orig_dtype=strokes.dtype)


def apply_word_spacing(
    strokes: np.ndarray,
    text: str,
    word_spacing: float,
    config: Optional[StyleConfig] = None,
    attention_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Adjusts the horizontal distance between consecutive words.

    Mathematical formulation:
        For word w (0 <= w < M):
            Delta_x_word(w) = w * word_spacing * max_word_displacement
        All points inside word w are shifted by the exact same constant offset,
        preserving the internal structure, shape, and character spacing of each word.

    Parameters:
        strokes: Array of shape (N, 3) representing (dx, dy, pen_lift).
        text: Corresponding text label.
        word_spacing: Control value in [-1.0, 1.0].
        config: Optional StyleConfig configuration.
        attention_map: Optional cross-attention map from diffusion model.

    Returns:
        modified_strokes: Array of shape (N, 3) with adjusted word spacing.
    """
    if config is None:
        config = StyleConfig()

    if config.clamp_controls:
        word_spacing = float(np.clip(word_spacing, -1.0, 1.0))

    if word_spacing == 0.0:
        return strokes.copy()

    words = [w for w in text.split(' ') if w]
    num_words = len(words)
    if num_words <= 1:
        return strokes.copy()

    pos = relative_to_absolute(strokes)
    pen_lifts = strokes[:, 2:3]
    word_boundaries = find_word_boundaries(strokes, text, config, attention_map)

    if not word_boundaries:
        return strokes.copy()

    x_shifts = np.zeros(len(pos), dtype=np.float64)

    # Word 0: [0, b_0) -> shift = 0
    # Word 1: [b_0, b_1) -> shift = 1 * delta
    # Word w: [b_{w-1}, b_w) -> shift = w * delta
    for w_idx in range(1, num_words):
        start_idx = word_boundaries[w_idx - 1]
        end_idx = word_boundaries[w_idx] if w_idx < len(word_boundaries) else len(pos)
        shift = w_idx * word_spacing * config.max_word_displacement
        x_shifts[start_idx:end_idx] = shift

    pos[:, 0] += x_shifts
    return absolute_to_relative(pos, pen_lifts, orig_dtype=strokes.dtype)


def apply_italics(
    strokes: np.ndarray,
    italics: float,
    config: Optional[StyleConfig] = None
) -> np.ndarray:
    """
    Applies an italics (slant) transformation using a horizontal shear.

    Mathematical transformation:
        x' = x + k * (y - y_base)
        y' = y
    Where:
        k = italics * max_slant_factor
        y_base is estimated from the lower boundary (baseline) of visible strokes.

    Since y increases upwards in the reconstructed coordinate system,
    points near the top of characters (y > y_base) shift to the right for k > 0,
    while points on the baseline (y ~= y_base) remain anchored, producing a natural forward slant.

    Parameters:
        strokes: Array of shape (N, 3) representing (dx, dy, pen_lift).
        italics: Control value in [-1.0, 1.0].
        config: Optional StyleConfig configuration.

    Returns:
        modified_strokes: Array of shape (N, 3) with slanted stroke trajectory.
    """
    if config is None:
        config = StyleConfig()

    if config.clamp_controls:
        italics = float(np.clip(italics, -1.0, 1.0))

    if italics == 0.0:
        return strokes.copy()

    pos = relative_to_absolute(strokes)
    pen_lifts = strokes[:, 2:3]

    # Robust baseline estimation using active stroke points
    active_segs = extract_active_segments(strokes, config)
    if active_segs:
        visible_y = []
        for s in active_segs:
            visible_y.extend(pos[s['start']:s['end'], 1])
        y_base = float(np.percentile(visible_y, config.baseline_percentile))
    else:
        y_base = float(np.percentile(pos[:, 1], config.baseline_percentile))

    k = italics * config.max_slant_factor
    pos[:, 0] += k * (pos[:, 1] - y_base)

    return absolute_to_relative(pos, pen_lifts, orig_dtype=strokes.dtype)


def compute_boldness_linewidth(
    boldness: float,
    config: Optional[StyleConfig] = None
) -> float:
    """
    Computes the rendering stroke thickness (linewidth) corresponding to the boldness control.

    When boldness is 0.0, returns the exact default linewidth (1.5).
    Positive values scale up to max_linewidth (3.6).
    Negative values scale down to min_linewidth (0.6).

    Parameters:
        boldness: Control value in [-1.0, 1.0].
        config: Optional StyleConfig configuration.

    Returns:
        linewidth: Float linewidth for matplotlib rendering.
    """
    if config is None:
        config = StyleConfig()

    if config.clamp_controls:
        boldness = float(np.clip(boldness, -1.0, 1.0))

    if boldness == 0.0:
        return config.default_linewidth
    elif boldness > 0.0:
        return config.default_linewidth + boldness * (config.max_linewidth - config.default_linewidth)
    else:
        return config.default_linewidth + boldness * (config.default_linewidth - config.min_linewidth)


def apply_all_styles(
    strokes: np.ndarray,
    text: str,
    char_spacing: float = 0.0,
    word_spacing: float = 0.0,
    italics: float = 0.0,
    boldness: float = 0.0,
    config: Optional[StyleConfig] = None,
    attention_map: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Applies all four mathematical style controllers in a unified, orthogonal pipeline.

    Order of operations:
    1. Zero-Control Bypass: If all controls are 0.0, returns an exact unmodified copy.
    2. Character Spacing: Shifts intra-word character clusters and accumulates word offsets.
    3. Word Spacing: Shifts words horizontally by word index.
    4. Italics (Slant): Applies horizontal shear on absolute coordinates.
    5. Boldness: Computes the rendering stroke thickness parameter.

    Parameters:
        strokes: Array of shape (N, 3) representing (dx, dy, pen_lift).
        text: Corresponding original text string.
        char_spacing: Normalized character spacing in [-1.0, 1.0].
        word_spacing: Normalized word spacing in [-1.0, 1.0].
        italics: Normalized italics (slant) in [-1.0, 1.0].
        boldness: Normalized stroke thickness in [-1.0, 1.0].
        config: Central configuration object.
        attention_map: Optional cross-attention map from diffusion model.

    Returns:
        modified_strokes: Array of shape (N, 3).
        linewidth: Rendering stroke width parameter.
    """
    if config is None:
        config = StyleConfig()

    if config.clamp_controls:
        char_spacing = float(np.clip(char_spacing, -1.0, 1.0))
        word_spacing = float(np.clip(word_spacing, -1.0, 1.0))
        italics = float(np.clip(italics, -1.0, 1.0))
        boldness = float(np.clip(boldness, -1.0, 1.0))

    # CRITICAL REQUIREMENT: Exact Zero-Control Identity Bypass
    # If all four controls are zero, bypass all transformations completely.
    is_zero_control = (char_spacing == 0.0 and word_spacing == 0.0 and
                       italics == 0.0 and boldness == 0.0)
    if is_zero_control:
        return strokes.copy(), config.default_linewidth

    linewidth = compute_boldness_linewidth(boldness, config)

    # If no geometric controls are active, return original stroke copy
    if char_spacing == 0.0 and word_spacing == 0.0 and italics == 0.0:
        return strokes.copy(), linewidth

    pos = relative_to_absolute(strokes)
    pen_lifts = strokes[:, 2:3]

    # Combined Spacing (Character Spacing and Word Spacing)
    if char_spacing != 0.0 or word_spacing != 0.0:
        words = [w for w in text.split(' ') if w]
        num_words = len(words)
        segments = extract_active_segments(strokes, config)
        word_boundaries = find_word_boundaries(strokes, text, config, attention_map)

        word_segments = [[] for _ in range(max(1, num_words))]
        for seg in segments:
            w_idx = 0
            for b in word_boundaries:
                if seg['start'] >= b:
                    w_idx += 1
                else:
                    break
            w_idx = min(w_idx, len(word_segments) - 1)
            word_segments[w_idx].append(seg)

        x_shifts = np.zeros(len(pos), dtype=np.float64)
        cumulative_char_expansion = 0.0
        last_active_stroke_end = segments[-1]['end'] if segments else len(pos)

        for w_idx in range(len(word_segments)):
            segs_in_word = word_segments[w_idx]
            expected_chars = len(words[w_idx]) if w_idx < len(words) else len(segs_in_word)
            char_clusters = cluster_word_segments(segs_in_word, expected_chars, config)
            num_clusters = len(char_clusters)

            # Word spacing displacement
            w_shift = w_idx * word_spacing * config.max_word_displacement
            base_word_shift = w_shift + cumulative_char_expansion

            # Character spacing within this word
            for c_idx, cluster in enumerate(char_clusters):
                c_shift = c_idx * char_spacing * config.max_char_displacement
                total_cluster_shift = base_word_shift + c_shift
                for seg in cluster:
                    x_shifts[seg['start']:seg['end']] = total_cluster_shift

            # Track word expansion so subsequent words stay at independent word spacing
            if num_clusters > 1:
                word_char_expansion = (num_clusters - 1) * char_spacing * config.max_char_displacement
                cumulative_char_expansion += word_char_expansion

        # Trailing padding / inactive strokes
        final_shift = (max(0, len(word_segments) - 1)) * word_spacing * config.max_word_displacement + cumulative_char_expansion
        if last_active_stroke_end < len(pos):
            x_shifts[last_active_stroke_end:] = final_shift

        pos[:, 0] += x_shifts

    # Italics Shear Transformation
    if italics != 0.0:
        active_segs = extract_active_segments(strokes, config)
        if active_segs:
            visible_y = []
            for s in active_segs:
                visible_y.extend(pos[s['start']:s['end'], 1])
            y_base = float(np.percentile(visible_y, config.baseline_percentile))
        else:
            y_base = float(np.percentile(pos[:, 1], config.baseline_percentile))

        k = italics * config.max_slant_factor
        pos[:, 0] += k * (pos[:, 1] - y_base)

    modified_strokes = absolute_to_relative(pos, pen_lifts, orig_dtype=strokes.dtype)
    return modified_strokes, linewidth


# ==============================================================================
# End-to-End Generation Convenience Interface
# ==============================================================================

def generate_handwriting(
    text: str,
    character_spacing: float = 0.0,
    word_spacing: float = 0.0,
    italics: float = 0.0,
    boldness: float = 0.0,
    writersource: Optional[str] = None,
    output_path: Optional[str] = None,
    show_sample: bool = False,
    weights_path: str = './weights/model_weights.h5',
    diffmode: str = 'new',
    seqlen: Optional[int] = None,
    style_config: Optional[StyleConfig] = None,
    model: Any = None,
    style_extractor: Any = None
) -> Tuple[np.ndarray, float]:
    """
    High-level end-to-end interface to generate stylized handwriting using the diffusion model.

    Workflow:
    1. User provides text.
    2. Original diffusion model generates base handwriting strokes.
    3. User provides four style-control values.
    4. Mathematical style-control module modifies the generated strokes.
    5. Modified strokes are rendered with the calculated boldness linewidth.
    6. Final handwriting image is displayed or saved.

    Parameters:
        text: Text string to generate.
        character_spacing: Normalized character spacing in [-1.0, 1.0].
        word_spacing: Normalized word spacing in [-1.0, 1.0].
        italics: Normalized italics (slant) in [-1.0, 1.0].
        boldness: Normalized stroke thickness in [-1.0, 1.0].
        writersource: Optional path to writer style image (random asset if None).
        output_path: Optional path to save image (e.g. './output.png').
        show_sample: Whether to display sample popup.
        weights_path: Path to diffusion model weights.
        diffmode: Diffusion sampling mode ('new' or 'standard').
        seqlen: Stroke sequence length (default 16 * length of text).
        style_config: Optional StyleConfig instance.
        model: Optional pre-loaded DiffusionWriter model to avoid reloading.
        style_extractor: Optional pre-loaded StyleExtractor.

    Returns:
        (strokes, linewidth): Modified strokes array and rendering stroke width.
    """
    import os
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import utils
    import nn
    import preprocessing

    if not show_sample:
        plt.switch_backend('Agg')

    timesteps = len(text) * 16 if seqlen is None else seqlen
    timesteps = timesteps - (timesteps % 8) + 8

    if writersource is None:
        assetdir = [f for f in os.listdir('./assets') if f.endswith(('.png', '.tif', '.jpg', '.jpeg'))]
        sourcename = os.path.join('./assets', assetdir[np.random.randint(0, len(assetdir))])
    else:
        sourcename = writersource

    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()

    if style_extractor is None:
        style_extractor = nn.StyleExtractor()

    if model is None:
        c1, c2, c3 = 128, 192, 256
        model = nn.DiffusionWriter(num_layers=2, c1=c1, c2=c2, c3=c3)
        _stroke = tf.random.normal([1, 400, 2])
        _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
        _noise = tf.random.uniform([1, 1])
        _style_vector = tf.random.normal([1, 14, 1280])
        _ = model(_stroke, _text, _noise, _style_vector)
        model.load_weights(weights_path)

    writer_img = tf.expand_dims(preprocessing.read_img(sourcename, 96), 0)
    style_vector = style_extractor(writer_img)

    strokes = utils.run_batch_inference(
        model,
        beta_set,
        text,
        style_vector,
        tokenizer=tokenizer,
        time_steps=timesteps,
        diffusion_mode=diffmode,
        show_samples=show_sample,
        path=output_path,
        char_spacing=character_spacing,
        word_spacing=word_spacing,
        italics=italics,
        boldness=boldness,
        style_config=style_config
    )

    lw = compute_boldness_linewidth(boldness, style_config)
    return strokes[0], lw
