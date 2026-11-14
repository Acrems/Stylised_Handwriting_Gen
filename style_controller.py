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


DEFAULT_MAX_CANVAS_WIDTH: int = 1400
DEFAULT_MAX_CANVAS_HEIGHT: int = 1400


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

    min_text_size_scale: float = 0.5
    max_text_size_scale: float = 2.0

    default_line_spacing: float = 0.0
    base_line_gap_ratio: float = 0.5
    max_line_gap_displacement: float = 3.0

    default_canvas_width: int = 1400
    default_canvas_height: int = 400
    max_canvas_width: int = DEFAULT_MAX_CANVAS_WIDTH
    max_canvas_height: int = DEFAULT_MAX_CANVAS_HEIGHT
    default_dpi: int = 100
    default_margin_ratio: float = 0.05
    pixels_per_unit: float = 15.0


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


def compute_text_size_scale(
    text_size: float,
    config: Optional[StyleConfig] = None
) -> float:
    """
    Computes the uniform scaling factor s corresponding to the text_size control value.

    Mapping:
        -1.0 -> config.min_text_size_scale (default 0.5: 50% size)
         0.0 -> 1.0 (exact original handwriting size)
        +1.0 -> config.max_text_size_scale (default 2.0: 200% size)

    Parameters:
        text_size: Control value in [-1.0, 1.0].
        config: Optional StyleConfig instance.

    Returns:
        s: Float uniform scaling factor.
    """
    if config is None:
        config = StyleConfig()

    if config.clamp_controls:
        text_size = float(np.clip(text_size, -1.0, 1.0))

    if text_size == 0.0:
        return 1.0
    elif text_size > 0.0:
        return 1.0 + text_size * (config.max_text_size_scale - 1.0)
    else:
        return 1.0 + text_size * (1.0 - config.min_text_size_scale)


def apply_text_size(
    strokes: np.ndarray,
    text_size: float,
    config: Optional[StyleConfig] = None
) -> np.ndarray:
    """
    Applies uniform scaling to handwriting stroke coordinates to control overall text size.

    Mathematical formulation:
        x' = x_anchor + s * (x - x_anchor)
        y' = y_anchor + s * (y - y_anchor)

    Where:
        - x, y are original absolute coordinates.
        - x', y' are scaled coordinates.
        - s is the uniform scaling factor computed via compute_text_size_scale.
        - x_anchor is the horizontal left boundary (bounding-box left reference).
        - y_anchor is the writing baseline reference estimated via baseline_percentile.

    Preserves:
        - Exact letter aspect ratio (uniform scaling, no independent horizontal/vertical stretch).
        - Pen-lift states.
        - Stroke sequence ordering.
        - Non-mutating: original strokes array is not modified in-place.
        - Exact identity when text_size == 0.0.

    Parameters:
        strokes: Array of shape (N, 3) representing (dx, dy, pen_lift).
        text_size: Control value in [-1.0, 1.0].
        config: Optional StyleConfig configuration.

    Returns:
        modified_strokes: Array of shape (N, 3) with scaled stroke coordinates.
    """
    if config is None:
        config = StyleConfig()

    if config.clamp_controls:
        text_size = float(np.clip(text_size, -1.0, 1.0))

    if text_size == 0.0:
        return strokes.copy()

    s = compute_text_size_scale(text_size, config)
    pos = relative_to_absolute(strokes)
    pen_lifts = strokes[:, 2:3]

    active_segs = extract_active_segments(strokes, config)
    if active_segs:
        x_anchor = float(min(seg['x_min'] for seg in active_segs))
        visible_y = []
        for seg in active_segs:
            visible_y.extend(pos[seg['start']:seg['end'], 1])
        y_anchor = float(np.percentile(visible_y, config.baseline_percentile))
    else:
        x_anchor = float(pos[:, 0].min()) if len(pos) > 0 else 0.0
        y_anchor = float(np.percentile(pos[:, 1], config.baseline_percentile)) if len(pos) > 0 else 0.0

    pos[:, 0] = x_anchor + s * (pos[:, 0] - x_anchor)
    pos[:, 1] = y_anchor + s * (pos[:, 1] - y_anchor)

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
    text_size: float = 0.0,
    config: Optional[StyleConfig] = None,
    attention_map: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Applies all five mathematical style controllers in a unified, orthogonal pipeline.

    Order of operations:
    1. Zero-Control Bypass: If all controls are 0.0, returns an exact unmodified copy.
    2. Character Spacing: Shifts intra-word character clusters and accumulates word offsets.
    3. Word Spacing: Shifts words horizontally by word index.
    4. Italics (Slant): Applies horizontal shear on absolute coordinates.
    5. Text Size: Applies uniform scaling relative to left boundary and baseline anchor.
    6. Boldness: Computes the rendering stroke thickness parameter.

    Parameters:
        strokes: Array of shape (N, 3) representing (dx, dy, pen_lift).
        text: Corresponding original text string.
        char_spacing: Normalized character spacing in [-1.0, 1.0].
        word_spacing: Normalized word spacing in [-1.0, 1.0].
        italics: Normalized italics (slant) in [-1.0, 1.0].
        boldness: Normalized stroke thickness in [-1.0, 1.0].
        text_size: Normalized handwriting size in [-1.0, 1.0].
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
        text_size = float(np.clip(text_size, -1.0, 1.0))

    # CRITICAL REQUIREMENT: Exact Zero-Control Identity Bypass
    is_zero_control = (char_spacing == 0.0 and word_spacing == 0.0 and
                       italics == 0.0 and boldness == 0.0 and text_size == 0.0)
    if is_zero_control:
        return strokes.copy(), config.default_linewidth

    linewidth = compute_boldness_linewidth(boldness, config)

    # If no geometric controls are active, return original stroke copy
    if char_spacing == 0.0 and word_spacing == 0.0 and italics == 0.0 and text_size == 0.0:
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

    if text_size != 0.0:
        s = compute_text_size_scale(text_size, config)
        active_segs = extract_active_segments(strokes, config)
        if active_segs:
            x_anchor = float(min(seg['x_min'] for seg in active_segs))
            visible_y = []
            for seg in active_segs:
                visible_y.extend(pos[seg['start']:seg['end'], 1])
            y_anchor = float(np.percentile(visible_y, config.baseline_percentile))
        else:
            x_anchor = float(pos[:, 0].min()) if len(pos) > 0 else 0.0
            y_anchor = float(np.percentile(pos[:, 1], config.baseline_percentile)) if len(pos) > 0 else 0.0

        pos[:, 0] = x_anchor + s * (pos[:, 0] - x_anchor)
        pos[:, 1] = y_anchor + s * (pos[:, 1] - y_anchor)

    modified_strokes = absolute_to_relative(pos, pen_lifts, orig_dtype=strokes.dtype)
    return modified_strokes, linewidth


def layout_multiline_strokes(
    lines_strokes: List[Optional[np.ndarray]],
    line_spacing: float = 0.0,
    config: Optional[StyleConfig] = None,
    x_margin: float = 0.0
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Arranges multiple lines of handwriting strokes into a single vertically organized stroke sequence.

    Each line has its own independent starting position:
        x' = x + x_offset[i]
        y' = y + y_offset[i]

    Lines are ordered top-to-bottom.
    Calculates actual bounding boxes for each generated line to ensure lines do not overlap.
    Preserves empty lines as blank line layout gaps.

    Parameters:
        lines_strokes: List of stroke arrays (shape (N_i, 3)) or None for blank lines.
        line_spacing: Normalized line spacing control in [-1.0, 1.0] (or positive spacing multiplier).
        config: Optional StyleConfig instance.
        x_margin: Horizontal left margin offset in coordinate units.

    Returns:
        combined_strokes: Array of shape (total_N, 3) representing the unified multiline stroke sequence.
        line_infos: List of dicts describing the bounding box and position of each line.
    """
    if config is None:
        config = StyleConfig()

    if config.clamp_controls and -1.0 <= line_spacing <= 1.0:
        line_spacing = float(np.clip(line_spacing, -1.0, 1.0))

    line_boxes = []
    valid_heights = []
    for strokes in lines_strokes:
        if strokes is not None and len(strokes) > 0:
            pos = relative_to_absolute(strokes)
            x_min, x_max = float(pos[:, 0].min()), float(pos[:, 0].max())
            y_min, y_max = float(pos[:, 1].min()), float(pos[:, 1].max())
            h = max(0.1, y_max - y_min)
            valid_heights.append(h)
            line_boxes.append({
                'pos': pos,
                'pen_lifts': strokes[:, 2:3],
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'h': h,
                'dtype': strokes.dtype
            })
        else:
            line_boxes.append(None)

    nom_h = float(np.mean(valid_heights)) if valid_heights else 6.0

    if -1.0 <= line_spacing <= 1.0:
        gap = nom_h * max(0.1, 0.5 + 0.5 * line_spacing)
    else:
        gap = nom_h * max(0.1, 0.5 * (1.0 + line_spacing))

    y_cursor = 0.0
    translated_positions = []
    translated_pen_lifts = []
    line_infos = []

    for idx, box in enumerate(line_boxes):
        if box is None:
            y_cursor -= (nom_h + gap)
            line_infos.append({
                'line_index': idx,
                'is_empty': True,
                'y_top': y_cursor + nom_h + gap,
                'y_bottom': y_cursor
            })
            continue

        pos = box['pos'].copy()
        pl = box['pen_lifts'].copy()
        pl[-1] = 1.0

        x_off = -box['x_min'] + x_margin
        y_top = y_cursor
        y_off = y_top - box['y_max']

        pos[:, 0] += x_off
        pos[:, 1] += y_off

        y_bottom = y_top - box['h']
        y_cursor = y_bottom - gap

        translated_positions.append(pos)
        translated_pen_lifts.append(pl)
        line_infos.append({
            'line_index': idx,
            'is_empty': False,
            'x_offset': x_off,
            'y_offset': y_off,
            'y_top': y_top,
            'y_bottom': y_bottom,
            'height': box['h']
        })

    if not translated_positions:
        dummy = np.zeros((1, 3), dtype=np.float32)
        dummy[0, 2] = 1.0
        return dummy, line_infos

    all_pos = np.vstack(translated_positions)
    all_pl = np.vstack(translated_pen_lifts)
    combined_strokes = absolute_to_relative(
        all_pos,
        all_pl,
        orig_dtype=line_boxes[0]['dtype'] if line_boxes[0] else np.float32
    )
    return combined_strokes, line_infos


def calculate_word_stroke_boxes(
    strokes: np.ndarray,
    text: str,
    config: Optional[StyleConfig] = None
) -> List[Dict[str, Any]]:
    """
    Calculates the exact bounding box, dimensions, and stroke slice for each word
    in the generated handwriting.

    Uses actual stroke coordinates rather than character counts or heuristics.

    Parameters:
        strokes: Array of shape (N, 3) representing (dx, dy, pen_lift).
        text: Text string corresponding to strokes.
        config: Optional StyleConfig instance.

    Returns:
        word_boxes: List of dicts describing each word:
            - word: Word text string
            - strokes: Stroke slice array (N_w, 3)
            - pos: Absolute coordinates of slice (N_w, 2)
            - start_idx, end_idx: Indices in original strokes array
            - x_min, x_max: Horizontal bounding box
            - y_min, y_max: Vertical bounding box
            - width: Actual horizontal width (x_max - x_min)
            - height: Actual vertical height (y_max - y_min)
            - trailing_gap: Horizontal distance to start of next word (or 0.0)
    """
    if config is None:
        config = StyleConfig()

    words = [w for w in text.split(' ') if w]
    num_words = len(words)
    pos = relative_to_absolute(strokes) if len(strokes) > 0 else np.zeros((0, 2), dtype=np.float64)

    if num_words <= 1 or len(strokes) == 0:
        x_min = float(pos[:, 0].min()) if len(pos) > 0 else 0.0
        x_max = float(pos[:, 0].max()) if len(pos) > 0 else 0.0
        y_min = float(pos[:, 1].min()) if len(pos) > 0 else 0.0
        y_max = float(pos[:, 1].max()) if len(pos) > 0 else 0.0
        return [{
            'word': text.strip(),
            'strokes': strokes.copy(),
            'pos': pos.copy(),
            'start_idx': 0,
            'end_idx': len(strokes),
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'width': max(0.01, x_max - x_min),
            'height': max(0.01, y_max - y_min),
            'trailing_gap': 0.0
        }]

    word_boundaries = find_word_boundaries(strokes, text, config)
    word_boxes = []

    for w_idx in range(num_words):
        start = word_boundaries[w_idx - 1] if w_idx > 0 else 0
        end = word_boundaries[w_idx] if w_idx < len(word_boundaries) else len(strokes)
        if end <= start:
            continue

        w_strokes = strokes[start:end].copy()
        w_pos = pos[start:end]

        x_min = float(w_pos[:, 0].min())
        x_max = float(w_pos[:, 0].max())
        y_min = float(w_pos[:, 1].min())
        y_max = float(w_pos[:, 1].max())
        width = max(0.01, x_max - x_min)
        height = max(0.01, y_max - y_min)

        if w_idx < len(word_boundaries) and word_boundaries[w_idx] < len(pos):
            next_start = word_boundaries[w_idx]
            trailing_gap = max(0.5, float(pos[next_start, 0] - x_max))
        else:
            trailing_gap = 0.0

        word_boxes.append({
            'word': words[w_idx],
            'strokes': w_strokes,
            'pos': w_pos,
            'start_idx': start,
            'end_idx': end,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'width': width,
            'height': height,
            'trailing_gap': trailing_gap
        })

    return word_boxes


def build_line_strokes_from_boxes(word_boxes: List[Dict[str, Any]]) -> np.ndarray:
    """
    Assembles a list of word stroke boxes into a continuous single-line stroke sequence.
    Preserves word internal relative coordinates, pen lifts, and horizontal word gaps.
    """
    if not word_boxes:
        dummy = np.zeros((1, 3), dtype=np.float32)
        dummy[0, 2] = 1.0
        return dummy

    if len(word_boxes) == 1:
        st = word_boxes[0]['strokes'].copy()
        st[-1, 2] = 1.0
        return st

    line_positions = []
    line_pen_lifts = []
    x_cursor = 0.0

    for i, w_info in enumerate(word_boxes):
        w_pos = w_info['pos'].copy()
        pl = w_info['strokes'][:, 2:3].copy()
        w_xmin = w_info['x_min']

        w_pos[:, 0] += (x_cursor - w_xmin)
        line_positions.append(w_pos)
        line_pen_lifts.append(pl)

        x_cursor += w_info['width'] + w_info['trailing_gap']

    combined_pos = np.vstack(line_positions)
    combined_pl = np.vstack(line_pen_lifts)
    combined_pl[-1] = 1.0

    return absolute_to_relative(
        combined_pos,
        combined_pl,
        orig_dtype=word_boxes[0]['strokes'].dtype
    )


def wrap_strokes_by_word(
    strokes: np.ndarray,
    text: str,
    available_width_coords: float,
    line_spacing: float = 0.0,
    config: Optional[StyleConfig] = None,
    x_margin: float = 0.0
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Automatically wraps words to subsequent lines when exceeding available horizontal canvas width.

    Key guarantees:
    - Words are never split across lines.
    - Explicit newlines ('\\n') are preserved as manual line breaks.
    - Consecutive newlines ('\\n\\n') preserve empty lines.
    - Word widths and line breaks are determined using actual stroke bounding boxes.
    - Single word wider than available canvas width triggers an explicit warning without distortion or shrinking.
    - Lines are laid out with layout_multiline_strokes without vertical overlap.

    Parameters:
        strokes: Array of shape (N, 3) representing (dx, dy, pen_lift).
        text: Text string corresponding to strokes (may contain '\\n').
        available_width_coords: Maximum horizontal space available for writing (in coordinate units).
        line_spacing: Configurable line spacing control [-1.0, 1.0].
        config: Optional StyleConfig instance.
        x_margin: Left margin offset in coordinate units.

    Returns:
        combined_strokes: Array of shape (total_N, 3) with wrapped multiline strokes.
        line_infos: List of line metadata dicts.
    """
    import warnings

    if config is None:
        config = StyleConfig()

    paragraphs = text.split('\n')
    lines_strokes: List[Optional[np.ndarray]] = []

    all_words = [w for w in text.replace('\n', ' ').split(' ') if w]
    all_word_boxes = calculate_word_stroke_boxes(strokes, ' '.join(all_words), config)

    box_cursor = 0
    for para in paragraphs:
        para_words = [w for w in para.split(' ') if w]
        if not para_words:
            lines_strokes.append(None)
            continue

        num_p_words = len(para_words)
        para_boxes = all_word_boxes[box_cursor:box_cursor + num_p_words]
        box_cursor += num_p_words

        if not para_boxes:
            continue

        current_line_boxes: List[Dict[str, Any]] = []
        current_line_width = 0.0

        for w_info in para_boxes:
            w_width = w_info['width']
            gap = current_line_boxes[-1]['trailing_gap'] if current_line_boxes else 0.0

            if w_width > available_width_coords:
                warnings.warn(
                    f"Word '{w_info['word']}' width ({w_width:.1f} units) exceeds available horizontal "
                    f"canvas width ({available_width_coords:.1f} units). Preserving word intact without splitting or distortion."
                )
                if current_line_boxes:
                    lines_strokes.append(build_line_strokes_from_boxes(current_line_boxes))
                    current_line_boxes = []
                    current_line_width = 0.0
                lines_strokes.append(build_line_strokes_from_boxes([w_info]))
                continue

            needed_width = (current_line_width + gap + w_width) if current_line_boxes else w_width
            if current_line_boxes and (needed_width > available_width_coords):
                lines_strokes.append(build_line_strokes_from_boxes(current_line_boxes))
                current_line_boxes = [w_info]
                current_line_width = w_width
            else:
                current_line_boxes.append(w_info)
                current_line_width = needed_width

        if current_line_boxes:
            lines_strokes.append(build_line_strokes_from_boxes(current_line_boxes))

    if not lines_strokes:
        return strokes.copy(), []

    return layout_multiline_strokes(
        lines_strokes,
        line_spacing=line_spacing,
        config=config,
        x_margin=x_margin
    )


def calculate_dynamic_canvas_dimensions(
    strokes: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    config: Optional[StyleConfig] = None,
    pixels_per_unit: float = 15.0,
    margin_ratio_x: float = 0.05,
    margin_ratio_y: float = 0.10
) -> Tuple[int, int]:
    """
    Computes canvas pixel dimensions under exactly three sizing modes (no --max_size CLI arg):

    Mode 1 (Neither width nor height specified: width is None and height is None):
        Smallest reasonable canvas calculated dynamically from stroke bounding boxes + margins,
        clamped to DEFAULT_MAX_CANVAS_WIDTH (1400) and DEFAULT_MAX_CANVAS_HEIGHT (1400).
        Short text is never enlarged to fill the maximum canvas.

    Mode 2 (Width specified, height unspecified: width is not None and height is None):
        Canvas width is exactly the specified width; height is dynamically calculated
        from wrapped lines and bounding box, clamped to DEFAULT_MAX_CANVAS_HEIGHT (1400).

    Mode 3 (Height specified, width unspecified: height is not None and width is None):
        Canvas height is exactly the specified height; width is dynamically calculated
        from widest line and bounding box, clamped to DEFAULT_MAX_CANVAS_WIDTH (1400).

    Fixed Mode (Both width and height specified):
        Canvas width and height are exactly the specified values.

    Parameters:
        strokes: Array of shape (N, 3) representing (dx, dy, pen_lift).
        width: Optional specified canvas width in pixels.
        height: Optional specified canvas height in pixels.
        config: Optional StyleConfig instance.
        pixels_per_unit: Coordinate scale factor (default: 15.0 px/unit).
        margin_ratio_x: Horizontal margin ratio (default: 0.05).
        margin_ratio_y: Vertical margin ratio (default: 0.10).

    Returns:
        (canvas_width, canvas_height): Integer pixel dimensions.
    """
    if config is None:
        config = StyleConfig()

    max_w = getattr(config, 'max_canvas_width', DEFAULT_MAX_CANVAS_WIDTH)
    max_h = getattr(config, 'max_canvas_height', DEFAULT_MAX_CANVAS_HEIGHT)

    pos = relative_to_absolute(strokes) if strokes is not None and len(strokes) > 0 else np.zeros((1, 2))
    if len(pos) > 0:
        w_stroke = float(np.max(pos[:, 0]) - np.min(pos[:, 0]))
        h_stroke = float(np.max(pos[:, 1]) - np.min(pos[:, 1]))
    else:
        w_stroke = 1.0
        h_stroke = 1.0

    w_needed_coords = max(1.0, w_stroke / max(0.1, 1.0 - 2.0 * margin_ratio_x))
    h_needed_coords = max(1.0, h_stroke / max(0.1, 1.0 - 2.0 * margin_ratio_y))

    dynamic_w_px = int(np.ceil(w_needed_coords * pixels_per_unit))
    dynamic_h_px = int(np.ceil(h_needed_coords * pixels_per_unit))

    dynamic_w_px = max(100, dynamic_w_px)
    dynamic_h_px = max(50, dynamic_h_px)

    if width is not None and height is not None:
        final_w = int(width)
        final_h = int(height)
    elif width is None and height is None:
        final_w = min(dynamic_w_px, max_w)
        final_h = min(dynamic_h_px, max_h)
    elif width is not None and height is None:
        final_w = int(width)
        final_h = min(dynamic_h_px, max_h)
    else:
        final_w = min(dynamic_w_px, max_w)
        final_h = int(height)

    return final_w, final_h


def generate_handwriting(
    text: str,
    character_spacing: float = 0.0,
    word_spacing: float = 0.0,
    italics: float = 0.0,
    boldness: float = 0.0,
    text_size: float = 0.0,
    line_spacing: float = 0.0,
    writersource: Optional[str] = None,
    output_path: Optional[str] = None,
    show_sample: bool = False,
    weights_path: str = './weights/model_weights.h5',
    diffmode: str = 'new',
    seqlen: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    autowrap: int = 0,
    fit_overflow: bool = False,
    style_config: Optional[StyleConfig] = None,
    model: Any = None,
    style_extractor: Any = None
) -> Tuple[np.ndarray, float]:
    """
    High-level end-to-end interface to generate stylized handwriting using the diffusion model.

    Workflow:
    1. User provides text (single line or multiline with '\n').
    2. Original diffusion model generates base handwriting strokes.
    3. User provides five style-control values (char_spacing, word_spacing, italics, boldness, text_size).
    4. Mathematical style-control module modifies the generated strokes.
    5. If multiline, arranges lines with automatic vertical positioning and line spacing.
    6. Modified strokes are rendered with the calculated boldness linewidth on a fixed canvas.
    7. Final handwriting image is displayed or saved.

    Parameters:
        text: Text string to generate (may contain newline characters).
        character_spacing: Normalized character spacing in [-1.0, 1.0].
        word_spacing: Normalized word spacing in [-1.0, 1.0].
        italics: Normalized italics (slant) in [-1.0, 1.0].
        boldness: Normalized stroke thickness in [-1.0, 1.0].
        text_size: Normalized text size in [-1.0, 1.0].
        line_spacing: Normalized line spacing in [-1.0, 1.0].
        writersource: Optional path to writer style image (random asset if None).
        output_path: Optional path to save image (e.g. './output.png').
        show_sample: Whether to display sample popup.
        weights_path: Path to diffusion model weights.
        diffmode: Diffusion sampling mode ('new' or 'standard').
        seqlen: Stroke sequence length (default 16 * length of text).
        canvas_width: Optional fixed canvas width in pixels.
        canvas_height: Optional fixed canvas height in pixels.
        multiline: Force multiline generation mode (also auto-enabled if '\n' in text).
        fit_overflow: Whether to uniformly scale down if text overflows canvas.
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
        text_size=text_size,
        line_spacing=line_spacing,
        width=width,
        height=height,
        autowrap=autowrap,
        fit_overflow=fit_overflow,
        style_config=style_config
    )

    lw = compute_boldness_linewidth(boldness, style_config)
    return strokes[0], lw
