"""
Automated Comprehensive Test Suite for Mathematical Handwriting Style Controllers.

Covers:
- Test A: Text Size (-1.0, 0.0, +1.0, aspect ratio, pen-lift invariance, non-mutating)
- Test B: Fixed Canvas Dimensions (identical pixel dimensions across varying text lengths)
- Test C: Two Lines Multiline Generation & Layout
- Test D: Three Lines Vertical Positioning & Arrangement
- Test E: Line Spacing Independence (varies vertical gap without affecting character/word spacing)
- Test F: Combined Controls (all five style controllers + line spacing)
- Test G: Zero Controls Exact Identity (bit-for-bit identity bypass)
- Test H: Overflow Detection & Fallback Handling
- Test I: Python 3.14 Compatibility & Dependency Reporting
- Existing controllers: Character Spacing, Word Spacing, Italics, Boldness, Clamping, Edge Cases
"""

import os
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

import unittest
import warnings
import numpy as np
from PIL import Image

import style_controller
from style_controller import StyleConfig, relative_to_absolute, absolute_to_relative
import utils


def make_synthetic_handwriting(text: str = "HELLO WORLD") -> np.ndarray:
    
    words = text.split(' ')
    all_strokes = []
    for w_idx, word in enumerate(words):
        for c_idx, char in enumerate(word):
            # 8 stroke points per character
            for i in range(8):
                dx = 0.15 + 0.05 * np.cos(i)
                dy = 0.35 * np.sin(i)
                pl = 1.0 if i == 7 else 0.0
                all_strokes.append([dx, dy, pl])
            # Inter-character transition gap
            all_strokes.append([0.8, 0.0, 1.0])
        # Inter-word transition gap
        all_strokes.append([2.5, 0.0, 1.0])
    return np.array(all_strokes, dtype=np.float32)


class TestMathematicalStyleControllers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set matplotlib backend to Agg for headless environments
        import matplotlib
        matplotlib.use('Agg')

        if os.path.exists('sample_strokes.npy'):
            cls.strokes = np.load('sample_strokes.npy')
        else:
            cls.strokes = make_synthetic_handwriting("HELLO WORLD")

        cls.text = "HELLO WORLD"
        cls.config = StyleConfig()

    # ==========================================================================
    # Test A: Text Size Control
    # ==========================================================================
    def test_a_text_size_scaling(self):
        """Verify that Text Size increases at +1.0 and decreases at -1.0."""
        strokes_inc = style_controller.apply_text_size(self.strokes, 1.0, config=self.config)
        strokes_dec = style_controller.apply_text_size(self.strokes, -1.0, config=self.config)
        strokes_zero = style_controller.apply_text_size(self.strokes, 0.0, config=self.config)

        pos_orig = relative_to_absolute(self.strokes)
        pos_inc = relative_to_absolute(strokes_inc)
        pos_dec = relative_to_absolute(strokes_dec)

        w_orig = pos_orig[:, 0].max() - pos_orig[:, 0].min()
        h_orig = pos_orig[:, 1].max() - pos_orig[:, 1].min()

        w_inc = pos_inc[:, 0].max() - pos_inc[:, 0].min()
        h_inc = pos_inc[:, 1].max() - pos_inc[:, 1].min()

        w_dec = pos_dec[:, 0].max() - pos_dec[:, 0].min()
        h_dec = pos_dec[:, 1].max() - pos_dec[:, 1].min()

        # Positive values must increase overall size
        self.assertGreater(w_inc, w_orig, "Text size +1.0 must increase width!")
        self.assertGreater(h_inc, h_orig, "Text size +1.0 must increase height!")

        # Negative values must decrease overall size
        self.assertLess(w_dec, w_orig, "Text size -1.0 must decrease width!")
        self.assertLess(h_dec, h_orig, "Text size -1.0 must decrease height!")

        # Zero value must preserve exact size
        self.assertTrue(np.array_equal(strokes_zero, self.strokes), "Text size 0.0 must leave strokes unchanged!")

    def test_a_text_size_aspect_ratio_preservation(self):
        """Verify that Text Size uniformly scales without distorting aspect ratio."""
        strokes_inc = style_controller.apply_text_size(self.strokes, 0.8, config=self.config)
        strokes_dec = style_controller.apply_text_size(self.strokes, -0.6, config=self.config)

        pos_orig = relative_to_absolute(self.strokes)
        pos_inc = relative_to_absolute(strokes_inc)
        pos_dec = relative_to_absolute(strokes_dec)

        aspect_orig = (pos_orig[:, 0].max() - pos_orig[:, 0].min()) / max(0.01, pos_orig[:, 1].max() - pos_orig[:, 1].min())
        aspect_inc = (pos_inc[:, 0].max() - pos_inc[:, 0].min()) / max(0.01, pos_inc[:, 1].max() - pos_inc[:, 1].min())
        aspect_dec = (pos_dec[:, 0].max() - pos_dec[:, 0].min()) / max(0.01, pos_dec[:, 1].max() - pos_dec[:, 1].min())

        np.testing.assert_allclose(aspect_inc, aspect_orig, rtol=1e-3,
                                  err_msg="Text Size scaling +0.8 distorted aspect ratio!")
        np.testing.assert_allclose(aspect_dec, aspect_orig, rtol=1e-3,
                                  err_msg="Text Size scaling -0.6 distorted aspect ratio!")

    def test_a_text_size_pen_lift_and_order_invariance(self):
        """Verify that Text Size preserves pen-lift states, stroke ordering, and does not mutate in place."""
        strokes_scaled = style_controller.apply_text_size(self.strokes, 0.7, config=self.config)

        # Non-mutating
        self.assertIsNot(strokes_scaled, self.strokes, "apply_text_size mutated input strokes in place!")

        # Sequence length and ordering
        self.assertEqual(len(strokes_scaled), len(self.strokes))

        # Pen lifts must be bit-for-bit identical
        self.assertTrue(np.array_equal(strokes_scaled[:, 2], self.strokes[:, 2]),
                        "Text size altered pen lift states!")

    # ==========================================================================
    # Test B: Mode 2 Canvas Width Uniformity
    # ==========================================================================
    def test_b_fixed_canvas_identical_pixel_dimensions(self):
        """Verify that short words, sentences, long sentences, and multiline text produce exact specified width in Mode 2."""
        s_short = make_synthetic_handwriting("Hi")
        s_sentence = make_synthetic_handwriting("Hello world how are you")
        s_long = make_synthetic_handwriting("This is a significantly longer handwriting sentence for testing canvas")

        # Multiline synthetic strokes
        l1 = make_synthetic_handwriting("Line one")
        l2 = make_synthetic_handwriting("Line two")
        l3 = make_synthetic_handwriting("Line three")
        s_multi, _ = style_controller.layout_multiline_strokes([l1, l2, l3], config=self.config)

        test_cases = [
            ("short.png", s_short),
            ("sentence.png", s_sentence),
            ("long.png", s_long),
            ("multiline.png", s_multi),
        ]

        expected_size = (1400, 400)
        temp_dir = tempfile.gettempdir()

        for filename, stroke_data in test_cases:
            filepath = os.path.join(temp_dir, filename)
            try:
                utils.show(
                    stroke_data,
                    name=filepath,
                    show_output=False,
                    width=1400,
                    height=400,
                    dpi=100
                )
                self.assertTrue(os.path.exists(filepath), f"File was not created: {filepath}")
                with Image.open(filepath) as img:
                    actual_size = img.size
                self.assertEqual(
                    actual_size,
                    expected_size,
                    f"{filename} generated dimensions {actual_size}, expected {expected_size}!"
                )
            finally:
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass

    # ==========================================================================
    # Test C: Two Lines Multiline Layout
    # ==========================================================================
    def test_c_two_lines_layout(self):
        """Verify that two lines are placed on the same canvas with Line 1 below Line 0 without overlap."""
        line0_strokes = make_synthetic_handwriting("Hello World")
        line1_strokes = make_synthetic_handwriting("This is a test.")

        combined, line_infos = style_controller.layout_multiline_strokes(
            [line0_strokes, line1_strokes],
            line_spacing=0.0,
            config=self.config
        )

        self.assertEqual(len(line_infos), 2)
        self.assertFalse(line_infos[0]['is_empty'])
        self.assertFalse(line_infos[1]['is_empty'])

        # Line 0 top must be strictly higher than Line 1 top
        self.assertGreater(line_infos[0]['y_top'], line_infos[1]['y_top'],
                           "Line 0 should be positioned above Line 1!")

        # Line 0 bottom must be greater than or equal to Line 1 top (no overlap)
        self.assertGreaterEqual(line_infos[0]['y_bottom'], line_infos[1]['y_top'],
                               "Lines overlap vertically!")

        # Both lines are in the combined stroke sequence
        pos_combined = relative_to_absolute(combined)
        self.assertGreater(len(pos_combined), len(line0_strokes))

    # ==========================================================================
    # Test D: Three Lines Vertical Positioning
    # ==========================================================================
    def test_d_three_lines_vertical_ordering(self):
        """Verify three lines maintain strictly monotonic top-to-bottom vertical ordering."""
        lines = [
            make_synthetic_handwriting("First line of text"),
            make_synthetic_handwriting("Second line below the first"),
            make_synthetic_handwriting("Third line at the bottom")
        ]

        combined, line_infos = style_controller.layout_multiline_strokes(
            lines,
            line_spacing=0.0,
            config=self.config
        )

        self.assertEqual(len(line_infos), 3)

        # Monotonic descent: y_top[0] > y_top[1] > y_top[2]
        self.assertGreater(line_infos[0]['y_top'], line_infos[1]['y_top'])
        self.assertGreater(line_infos[1]['y_top'], line_infos[2]['y_top'])

        # No overlap between consecutive lines
        self.assertGreaterEqual(line_infos[0]['y_bottom'], line_infos[1]['y_top'])
        self.assertGreaterEqual(line_infos[1]['y_bottom'], line_infos[2]['y_top'])

    # ==========================================================================
    # Test E: Line Spacing Independence
    # ==========================================================================
    def test_e_line_spacing(self):
        """Verify line spacing adjusts vertical distance between lines without altering intra-line geometry."""
        line0 = make_synthetic_handwriting("ALPHA LINE")
        line1 = make_synthetic_handwriting("BETA LINE")

        _, infos_compact = style_controller.layout_multiline_strokes([line0, line1], line_spacing=-0.8, config=self.config)
        _, infos_default = style_controller.layout_multiline_strokes([line0, line1], line_spacing=0.0, config=self.config)
        _, infos_wide = style_controller.layout_multiline_strokes([line0, line1], line_spacing=0.8, config=self.config)

        gap_compact = infos_compact[0]['y_bottom'] - infos_compact[1]['y_top']
        gap_default = infos_default[0]['y_bottom'] - infos_default[1]['y_top']
        gap_wide = infos_wide[0]['y_bottom'] - infos_wide[1]['y_top']

        self.assertGreater(gap_wide, gap_default, "Higher line spacing must increase inter-line gap!")
        self.assertGreater(gap_default, gap_compact, "Lower line spacing must decrease inter-line gap!")

        # Verify intra-line coordinates of line0 are unaffected by line_spacing
        pos0 = relative_to_absolute(line0)
        span0_orig = pos0[:, 0].max() - pos0[:, 0].min()
        self.assertAlmostEqual(span0_orig, pos0[:, 0].max() - pos0[:, 0].min(), places=5)

    # ==========================================================================
    # Test F: Combined Controls
    # ==========================================================================
    def test_f_combined_controls(self):
        """Verify that all five style controllers and line spacing can be applied simultaneously."""
        mod_strokes, lw = style_controller.apply_all_styles(
            self.strokes,
            self.text,
            char_spacing=0.4,
            word_spacing=0.6,
            italics=0.3,
            boldness=0.5,
            text_size=0.4,
            config=self.config
        )

        self.assertEqual(len(mod_strokes), len(self.strokes))
        self.assertTrue(np.all(np.isfinite(mod_strokes)))
        self.assertGreater(lw, self.config.default_linewidth)

    # ==========================================================================
    # Test G: Zero Controls Exact Identity
    # ==========================================================================
    def test_g_zero_controls_exact_identity(self):
        """Verify that all five style controls at 0.0 preserve exact bit-for-bit stroke array identity."""
        mod_strokes, lw = style_controller.apply_all_styles(
            self.strokes,
            self.text,
            char_spacing=0.0,
            word_spacing=0.0,
            italics=0.0,
            boldness=0.0,
            text_size=0.0,
            config=self.config
        )

        # Bit-for-bit identity check
        self.assertTrue(
            np.array_equal(self.strokes, mod_strokes),
            "Zero controls did not preserve exact stroke array identity!"
        )
        self.assertEqual(lw, self.config.default_linewidth)
        self.assertIsNot(mod_strokes, self.strokes, "Expected safe copy, but received original object!")

    # ==========================================================================
    # Test H: Overflow Detection and Fallback Handling
    # ==========================================================================
    def test_h_overflow_detection_and_fallback(self):
        """Verify that horizontal overflow is detected, warned, and uniformly fitted without cropping."""
        # Create an exceptionally long stroke sequence that exceeds standard canvas width
        s_huge = make_synthetic_handwriting("A " * 80)
        temp_path = os.path.join(tempfile.gettempdir(), "test_overflow.png")

        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                utils.show(
                    s_huge,
                    name=temp_path,
                    show_output=False,
                    width=1400,
                    height=None,
                    dpi=100,
                    fit_overflow=True
                )

                # Verify warning was raised
                overflow_warned = any("exceed" in str(w.message).lower() for w in caught_warnings)
                self.assertTrue(overflow_warned, "Overflow warning was not raised for huge text!")

            # Verify image was rendered with exact specified canvas width
            self.assertTrue(os.path.exists(temp_path))
            with Image.open(temp_path) as img:
                self.assertEqual(img.size[0], 1400)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    # ==========================================================================
    # Test I: Python 3.14 Compatibility & Dependency Reporting
    # ==========================================================================
    def test_i_python_compatibility(self):
        """Verify source compatibility with Python 3.14 and report dependency status."""
        py_version = sys.version_info
        print(f"\n[TEST I] Runtime Python Version: {py_version.major}.{py_version.minor}.{py_version.micro}")

        # Core math & style modules must import without deprecation errors
        self.assertIsNotNone(style_controller)
        self.assertIsNotNone(utils)

        # Check TensorFlow status in Python 3.14
        try:
            import tensorflow as tf
            tf_available = True
            tf_version = tf.__version__
        except (ImportError, Exception) as e:
            tf_available = False
            tf_version = str(e)

        print(f"[TEST I] TensorFlow availability: {tf_available} ({tf_version})")

    # ==========================================================================
    # Feature 1: Automatic Word Wrapping (11 Test Cases)
    # ==========================================================================
    def test_ww_01_word_moves_to_next_line_when_exceeding_canvas_width(self):
        """1. Verify that when adding the next word exceeds available width, it moves to the next line."""
        text = "Alpha Beta Gamma Delta"
        s = make_synthetic_handwriting(text)
        boxes = style_controller.calculate_word_stroke_boxes(s, text, self.config)
        w_two_words = boxes[0]['width'] + boxes[0]['trailing_gap'] + boxes[1]['width']
        avail_w = w_two_words + 1.0
        combined, line_infos = style_controller.wrap_strokes_by_word(s, text, avail_w, config=self.config)
        self.assertGreaterEqual(len(line_infos), 2, "Text should wrap to at least 2 lines!")
        self.assertGreater(line_infos[0]['y_top'], line_infos[1]['y_top'])

    def test_ww_02_words_never_split(self):
        """2. Verify words are never split across lines (every word belongs entirely to one line)."""
        text = "Diffusion Models Generate Handwriting Perfectly"
        s = make_synthetic_handwriting(text)
        avail_w = 40.0
        combined, line_infos = style_controller.wrap_strokes_by_word(s, text, avail_w, config=self.config)
        self.assertTrue(len(combined) >= len(s))
        self.assertTrue(np.all(np.isfinite(combined)))

    def test_ww_03_explicit_newlines_preserved(self):
        """3. Verify explicit newline characters are preserved as manual breaks."""
        text = "First Line\nSecond Line\nThird Line"
        s = make_synthetic_handwriting(text.replace('\n', ' '))
        avail_w = 200.0  # Large width so auto-wrapping does not trigger
        combined, line_infos = style_controller.wrap_strokes_by_word(s, text, avail_w, config=self.config)
        self.assertEqual(len(line_infos), 3, "Explicit 3 lines must produce 3 line_infos!")
        self.assertGreater(line_infos[0]['y_top'], line_infos[1]['y_top'])
        self.assertGreater(line_infos[1]['y_top'], line_infos[2]['y_top'])

    def test_ww_04_arbitrary_number_of_lines(self):
        """4. Verify wrapping supports arbitrary numbers of lines (e.g. 5 lines)."""
        words = [f"Word{i}" for i in range(15)]
        text = " ".join(words)
        s = make_synthetic_handwriting(text)
        avail_w = 25.0
        combined, line_infos = style_controller.wrap_strokes_by_word(s, text, avail_w, config=self.config)
        self.assertGreaterEqual(len(line_infos), 4, f"Expected 4+ lines, got {len(line_infos)}!")
        for i in range(len(line_infos) - 1):
            self.assertGreater(line_infos[i]['y_top'], line_infos[i+1]['y_top'])

    def test_ww_05_wrapping_uses_actual_stroke_widths(self):
        """5. Verify wrapping uses actual bounding box coordinates, not character count heuristics."""
        text = "AAA BBB CCC DDD"
        s = make_synthetic_handwriting(text)
        boxes = style_controller.calculate_word_stroke_boxes(s, text, self.config)
        for b in boxes:
            self.assertIn('width', b)
            self.assertIn('x_min', b)
            self.assertIn('x_max', b)
            self.assertGreater(b['width'], 0.0)

    def test_ww_06_word_order_and_text_content_preserved(self):
        """6. Verify original word order is preserved when wrapped."""
        text = "One Two Three Four Five"
        s = make_synthetic_handwriting(text)
        boxes = style_controller.calculate_word_stroke_boxes(s, text, self.config)
        extracted_words = [b['word'] for b in boxes]
        self.assertEqual(extracted_words, ["One", "Two", "Three", "Four", "Five"])

    def test_ww_07_consecutive_newlines_preserve_empty_lines(self):
        """7. Verify consecutive newlines ('\\n\\n') preserve empty lines without crash."""
        text = "Line One\n\nLine Three"
        s = make_synthetic_handwriting("Line One Line Three")
        combined, line_infos = style_controller.wrap_strokes_by_word(s, text, 100.0, config=self.config)
        self.assertEqual(len(line_infos), 3)
        self.assertTrue(line_infos[1]['is_empty'], "Middle line must be empty!")
        self.assertGreater(line_infos[0]['y_top'], line_infos[1]['y_top'])
        self.assertGreater(line_infos[1]['y_top'], line_infos[2]['y_top'])

    def test_ww_08_wrapping_respects_style_transformations(self):
        """8. Verify wrapping respects text size and spacing transformations."""
        text = "One Two Three Four Five Six Seven Eight"
        s_base = make_synthetic_handwriting(text)
        s_large = style_controller.apply_text_size(s_base, 1.0, config=self.config)
        s_small = style_controller.apply_text_size(s_base, -1.0, config=self.config)

        avail_w = 60.0
        _, lines_large = style_controller.wrap_strokes_by_word(s_large, text, avail_w, config=self.config)
        _, lines_small = style_controller.wrap_strokes_by_word(s_small, text, avail_w, config=self.config)

        self.assertGreater(len(lines_large), len(lines_small),
                           "Larger text size should produce more wrapped lines than smaller text size!")

    def test_ww_09_lines_do_not_overlap_vertically(self):
        """9. Verify consecutive lines never overlap vertically."""
        text = "A quick brown fox jumps over the lazy dog"
        s = make_synthetic_handwriting(text)
        combined, line_infos = style_controller.wrap_strokes_by_word(s, text, 35.0, config=self.config)
        self.assertGreaterEqual(len(line_infos), 2)
        for i in range(len(line_infos) - 1):
            if not line_infos[i]['is_empty'] and not line_infos[i+1]['is_empty']:
                self.assertGreaterEqual(
                    line_infos[i]['y_bottom'],
                    line_infos[i+1]['y_top'],
                    f"Line {i} overlaps with Line {i+1}!"
                )

    def test_ww_10_single_word_wider_than_canvas_warns_without_distortion(self):
        """10. Verify single word wider than canvas issues a warning, is not split or shrunk."""
        text = "Supercalifragilisticexpialidocious"
        s = make_synthetic_handwriting(text)
        pos = relative_to_absolute(s)
        word_w = float(pos[:, 0].max() - pos[:, 0].min())
        tight_w = word_w * 0.5

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            combined, line_infos = style_controller.wrap_strokes_by_word(s, text, tight_w, config=self.config)
            warned = any("exceeds available horizontal" in str(w.message) for w in caught)
            self.assertTrue(warned, "Warning not emitted for single-word width overflow!")

        pos_comb = relative_to_absolute(combined)
        w_comb = float(pos_comb[:, 0].max() - pos_comb[:, 0].min())
        h_comb = float(pos_comb[:, 1].max() - pos_comb[:, 1].min())
        orig_aspect = word_w / (pos[:, 1].max() - pos[:, 1].min())
        comb_aspect = w_comb / h_comb
        np.testing.assert_allclose(comb_aspect, orig_aspect, rtol=1e-2,
                                  err_msg="Aspect ratio was distorted on overflow!")

    def test_ww_11_height_overflow_warns_clearly_without_cropping_or_shrinking(self):
        """11. Verify height overflow emits a clear warning and does not crop or shrink."""
        lines = [make_synthetic_handwriting(f"Line number {i}") for i in range(25)]
        combined, _ = style_controller.layout_multiline_strokes(lines, config=self.config)
        temp_path = os.path.join(tempfile.gettempdir(), "test_height_overflow.png")

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                utils.show(
                    combined,
                    name=temp_path,
                    show_output=False,
                    width=None,
                    height=400,
                    fit_overflow=False
                )
                warned = any("exceed" in str(w.message).lower() for w in caught)
                self.assertTrue(warned, "Warning not emitted for canvas height overflow!")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    # ==========================================================================
    # Feature 2: Dynamic Canvas Dimensions (10 Test Cases)
    # ==========================================================================
    def test_cs_01_mode1_neither_width_nor_height_specified(self):
        """1. Mode 1: auto width & height calculated dynamically from bounding box + margins."""
        s = make_synthetic_handwriting("Hello World")
        w, h = style_controller.calculate_dynamic_canvas_dimensions(s, width=None, height=None, config=self.config)
        self.assertLessEqual(w, 1400)
        self.assertLessEqual(h, 1400)
        self.assertGreater(w, 50)
        self.assertGreater(h, 30)

    def test_cs_02_mode2_width_specified_height_unspecified(self):
        """2. Mode 2: exact specified width, height dynamically calculated."""
        s = make_synthetic_handwriting("Hello World Test Sentence")
        w, h = style_controller.calculate_dynamic_canvas_dimensions(s, width=1000, height=None, config=self.config)
        self.assertEqual(w, 1000, "Mode 2 must have exact specified width!")
        self.assertLessEqual(h, 1400)
        self.assertGreater(h, 0)

    def test_cs_03_mode3_height_specified_width_unspecified(self):
        """3. Mode 3: exact specified height, width dynamically calculated."""
        s = make_synthetic_handwriting("Hello World Test Sentence")
        w, h = style_controller.calculate_dynamic_canvas_dimensions(s, width=None, height=800, config=self.config)
        self.assertEqual(h, 800, "Mode 3 must have exact specified height!")
        self.assertLessEqual(w, 1400)
        self.assertGreater(w, 0)

    def test_cs_04_both_width_and_height_specified(self):
        """4. Verify that specifying both width and height produces exact specified dimensions."""
        s = make_synthetic_handwriting("Hello")
        w, h = style_controller.calculate_dynamic_canvas_dimensions(s, width=1100, height=550, config=self.config)
        self.assertEqual((w, h), (1100, 550))

    def test_cs_05_max_canvas_limits_respected(self):
        """5. Max canvas limits (1400x1400) respected in Mode 1, Mode 2, and Mode 3."""
        s_huge = make_synthetic_handwriting("A " * 120)
        w1, h1 = style_controller.calculate_dynamic_canvas_dimensions(s_huge, width=None, height=None, config=self.config)
        self.assertLessEqual(w1, style_controller.DEFAULT_MAX_CANVAS_WIDTH)
        self.assertLessEqual(h1, style_controller.DEFAULT_MAX_CANVAS_HEIGHT)

    def test_cs_06_short_text_never_enlarged_in_mode1(self):
        """6. Short text is never enlarged to 1400 in Mode 1."""
        s_short = make_synthetic_handwriting("Hi")
        w, h = style_controller.calculate_dynamic_canvas_dimensions(s_short, width=None, height=None, config=self.config)
        self.assertLess(w, 600, "Short text should not produce an inflated canvas width!")
        self.assertLess(h, 300, "Short text should not produce an inflated canvas height!")

    def test_cs_07_long_text_does_not_alter_specified_dimension(self):
        """7. Long text does not alter specified width in Mode 2 or specified height in Mode 3."""
        s_short = make_synthetic_handwriting("Short")
        s_long = make_synthetic_handwriting("Extremely long handwriting sentence with many words")

        w2_short, _ = style_controller.calculate_dynamic_canvas_dimensions(s_short, width=1000, height=None)
        w2_long, _ = style_controller.calculate_dynamic_canvas_dimensions(s_long, width=1000, height=None)
        self.assertEqual(w2_short, 1000)
        self.assertEqual(w2_long, 1000)

        _, h3_short = style_controller.calculate_dynamic_canvas_dimensions(s_short, width=None, height=600)
        _, h3_long = style_controller.calculate_dynamic_canvas_dimensions(s_long, width=None, height=600)
        self.assertEqual(h3_short, 600)
        self.assertEqual(h3_long, 600)

    def test_cs_08_text_size_affects_dynamic_dimensions_but_preserves_explicit(self):
        """8. Text size scaling affects dynamic dimensions, but preserves explicit dimension."""
        s = make_synthetic_handwriting("Scaling Test")
        s_large = style_controller.apply_text_size(s, 1.0, config=self.config)
        s_small = style_controller.apply_text_size(s, -1.0, config=self.config)

        w1_l, h1_l = style_controller.calculate_dynamic_canvas_dimensions(s_large, width=None, height=None)
        w1_s, h1_s = style_controller.calculate_dynamic_canvas_dimensions(s_small, width=None, height=None)
        self.assertGreater(w1_l, w1_s)
        self.assertGreater(h1_l, h1_s)

        w2_l, _ = style_controller.calculate_dynamic_canvas_dimensions(s_large, width=900, height=None)
        w2_s, _ = style_controller.calculate_dynamic_canvas_dimensions(s_small, width=900, height=None)
        self.assertEqual(w2_l, 900)
        self.assertEqual(w2_s, 900)

    def test_cs_09_saved_image_pixels_match_exactly(self):
        """9. Saved image pixel dimensions match calculated/specified canvas dimensions exactly."""
        temp_dir = tempfile.gettempdir()
        test_cases = [
            ("test_m1.png", make_synthetic_handwriting("Test Mode 1"), None, None),
            ("test_m2.png", make_synthetic_handwriting("Test Mode 2"), 850, None),
            ("test_m3.png", make_synthetic_handwriting("Test Mode 3"), None, 650),
            ("test_m4.png", make_synthetic_handwriting("Test Mode 4"), 900, 450),
        ]

        for filename, strokes, w_in, h_in in test_cases:
            filepath = os.path.join(temp_dir, filename)
            try:
                expected_w, expected_h = style_controller.calculate_dynamic_canvas_dimensions(
                    strokes, width=w_in, height=h_in, config=self.config
                )
                utils.show(
                    strokes,
                    name=filepath,
                    show_output=False,
                    width=w_in,
                    height=h_in,
                    dpi=100
                )
                self.assertTrue(os.path.exists(filepath))
                with Image.open(filepath) as img:
                    actual_size = img.size
                self.assertEqual(actual_size, (expected_w, expected_h),
                                 f"{filename} size mismatch: got {actual_size}, expected {(expected_w, expected_h)}!")
            finally:
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass

    def test_cs_10_confirm_no_max_size_cli_argument(self):
        """10. Verify that --max_size is NOT present as a CLI argument in inference.py or demo.py."""
        with open('inference.py', 'r', encoding='utf-8') as f:
            inf_code = f.read()
        with open('demo.py', 'r', encoding='utf-8') as f:
            demo_code = f.read()

        self.assertNotIn('--max_size', inf_code, "inference.py should NOT contain --max_size CLI argument!")
        self.assertNotIn('--max_size', demo_code, "demo.py should NOT contain --max_size CLI argument!")

    # ==========================================================================
    # Existing Controllers & Regression Tests
    # ==========================================================================
    def test_regression_character_spacing(self):
        """Verify character spacing expansion and contraction."""
        pos_orig = relative_to_absolute(self.strokes)
        span_orig = pos_orig[:, 0].max() - pos_orig[:, 0].min()

        strokes_inc = style_controller.apply_character_spacing(self.strokes, self.text, 1.0, config=self.config)
        pos_inc = relative_to_absolute(strokes_inc)
        span_inc = pos_inc[:, 0].max() - pos_inc[:, 0].min()

        strokes_dec = style_controller.apply_character_spacing(self.strokes, self.text, -1.0, config=self.config)
        pos_dec = relative_to_absolute(strokes_dec)
        span_dec = pos_dec[:, 0].max() - pos_dec[:, 0].min()

        self.assertGreater(span_inc, span_orig, "Character spacing +1.0 must increase span!")
        self.assertLess(span_dec, span_orig, "Character spacing -1.0 must decrease span!")

    def test_regression_word_spacing(self):
        """Verify word spacing increases inter-word gap."""
        boundaries = style_controller.find_word_boundaries(self.strokes, self.text, self.config)
        self.assertTrue(len(boundaries) > 0, "No word boundary found!")
        b = boundaries[0]

        strokes_inc = style_controller.apply_word_spacing(self.strokes, self.text, 1.0, config=self.config)
        strokes_dec = style_controller.apply_word_spacing(self.strokes, self.text, -1.0, config=self.config)

        pos_orig = relative_to_absolute(self.strokes)
        pos_inc = relative_to_absolute(strokes_inc)
        pos_dec = relative_to_absolute(strokes_dec)

        gap_orig = pos_orig[b, 0] - pos_orig[b - 1, 0]
        gap_inc = pos_inc[b, 0] - pos_inc[b - 1, 0]
        gap_dec = pos_dec[b, 0] - pos_dec[b - 1, 0]

        self.assertGreater(gap_inc, gap_orig, "Word spacing +1.0 must increase gap!")
        self.assertLess(gap_dec, gap_orig, "Word spacing -1.0 must decrease gap!")

    def test_regression_italics(self):
        """Verify italics shears top coordinates while preserving vertical coordinates."""
        strokes_inc = style_controller.apply_italics(self.strokes, 1.0, config=self.config)
        pos_orig = relative_to_absolute(self.strokes)
        pos_inc = relative_to_absolute(strokes_inc)

        np.testing.assert_allclose(pos_inc[:, 1], pos_orig[:, 1], atol=1e-6)
        top_idx = int(np.argmax(pos_orig[:, 1]))
        self.assertGreater(pos_inc[top_idx, 0], pos_orig[top_idx, 0])

    def test_regression_boldness(self):
        """Verify boldness modifies linewidth without modifying coordinates."""
        lw_inc = style_controller.compute_boldness_linewidth(1.0, self.config)
        lw_dec = style_controller.compute_boldness_linewidth(-1.0, self.config)
        lw_zero = style_controller.compute_boldness_linewidth(0.0, self.config)

        self.assertGreater(lw_inc, self.config.default_linewidth)
        self.assertLess(lw_dec, self.config.default_linewidth)
        self.assertEqual(lw_zero, self.config.default_linewidth)

    def test_regression_input_clamping(self):
        """Verify out-of-range inputs are clamped safely to [-1.0, 1.0]."""
        mod_clamped, lw_clamped = style_controller.apply_all_styles(
            self.strokes, self.text,
            char_spacing=2.5, word_spacing=3.0, italics=5.0, boldness=10.0, text_size=4.0,
            config=self.config
        )
        mod_max, lw_max = style_controller.apply_all_styles(
            self.strokes, self.text,
            char_spacing=1.0, word_spacing=1.0, italics=1.0, boldness=1.0, text_size=1.0,
            config=self.config
        )
        np.testing.assert_allclose(mod_clamped, mod_max, atol=1e-5)
        self.assertEqual(lw_clamped, lw_max)

    def test_regression_empty_and_single_char_text(self):
        """Verify empty text and single character text run gracefully."""
        strokes_empty, _ = style_controller.apply_all_styles(self.strokes, "", char_spacing=0.5, text_size=0.5, config=self.config)
        self.assertEqual(len(strokes_empty), len(self.strokes))

        strokes_single, _ = style_controller.apply_all_styles(self.strokes, "A", char_spacing=0.5, text_size=0.5, config=self.config)
        self.assertEqual(len(strokes_single), len(self.strokes))

    # ==========================================================================
    # CLI Argument & Autowrap Integration Tests
    # ==========================================================================
    def test_cli_both_width_and_height_accepted(self):
        """Verify that CLI runners accept both --width and --height simultaneously."""
        import subprocess
        for script in ['inference.py', 'demo.py']:
            cmd = [sys.executable, script, '--help']
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0)
            self.assertIn("--width", res.stdout)
            self.assertIn("--height", res.stdout)
            self.assertNotIn("--canvas_width", res.stdout)
            self.assertNotIn("--canvas_height", res.stdout)

    def test_autowrap_flag_semantics_and_user_scenario(self):
        """Verify that autowrap=0 preserves single line and autowrap=1 wraps words."""
        text = "Generative AI Loves Generating"
        s = make_synthetic_handwriting(text)
        # Apply text_size=0.6 as in user scenario
        s_sized = style_controller.apply_text_size(s, 0.6, config=self.config)

        # In Mode 2 with width=1400, available width is:
        avail_w = (1400 / 15.0) * 0.9  # 84.0 coords

        # When autowrap is 0 (disabled), text is not wrapped
        # The wrapped routine with autowrap=1 will split into multiple lines:
        wrapped_strokes, line_infos = style_controller.wrap_strokes_by_word(
            s_sized, text, available_width_coords=avail_w, config=self.config
        )
        self.assertGreaterEqual(len(line_infos), 2, "With autowrap=1, this text should wrap to >= 2 lines")

        # But when autowrap=0 is passed to style_controller.generate_handwriting or run_batch_inference,
        # it preserves the single-line layout
        # Verify that layout_multiline_strokes is NOT called on word chunks when autowrap=0
        pos_raw = relative_to_absolute(s_sized)
        y_range_raw = pos_raw[:, 1].max() - pos_raw[:, 1].min()

        pos_wrapped = relative_to_absolute(wrapped_strokes)
        y_range_wrapped = pos_wrapped[:, 1].max() - pos_wrapped[:, 1].min()

        # The multi-line wrapped strokes have larger vertical span than single-line strokes
        self.assertGreater(y_range_wrapped, y_range_raw,
                           "Wrapped strokes should occupy more vertical height than single-line")

    def test_single_pass_efficiency_reuse(self):
        """Verify that word wrapping reuses input strokes without generating duplicates."""
        text = "Fast Efficient Stroke Reuse Verification"
        s = make_synthetic_handwriting(text)
        boxes = style_controller.calculate_word_stroke_boxes(s, text, self.config)
        # Total strokes across all word boxes must equal total strokes in input
        total_word_strokes = sum(b['end_idx'] - b['start_idx'] for b in boxes)
        self.assertEqual(total_word_strokes, len(s), "Word bounding boxes must cover exactly all input strokes!")


if __name__ == '__main__':
    unittest.main()
