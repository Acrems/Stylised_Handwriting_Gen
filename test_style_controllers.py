"""
Automated Comprehensive Test Suite for Mathematical Handwriting Style Controllers.

Covers:
- Test A: Zero Controls (Identity & Bypass)
- Test B: Character Spacing (-1.0, 0.0, +1.0)
- Test C: Word Spacing (-1.0, 0.0, +1.0)
- Test D: Italics (-1.0, 0.0, +1.0)
- Test E: Boldness (-1.0, 0.0, +1.0)
- Test F: Combined Controls & Edge Cases (clamping, empty text, single char, independence)
"""

import os
import unittest
import numpy as np

import style_controller
from style_controller import StyleConfig, relative_to_absolute, absolute_to_relative


class TestMathematicalStyleControllers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load sample strokes if available, else generate synthetic realistic strokes
        if os.path.exists('sample_strokes.npy'):
            cls.strokes = np.load('sample_strokes.npy')
        else:
            # Fallback synthetic stroke sequence
            N = 120
            dx = np.random.normal(0.2, 0.5, (N, 1))
            dy = np.random.normal(0.0, 0.4, (N, 1))
            p = np.zeros((N, 1))
            p[[15, 35, 60, 85, 110]] = 1.0
            cls.strokes = np.concatenate([dx, dy, p], axis=-1).astype(np.float32)

        cls.text = "HELLO WORLD"
        cls.config = StyleConfig()

    # ==========================================================================
    # Test A: Zero Controls
    # ==========================================================================
    def test_a_zero_controls_exact_identity(self):
        """Verify that zero controls produce an EXACT bit-for-bit identical copy."""
        mod_strokes, lw = style_controller.apply_all_styles(
            self.strokes,
            self.text,
            char_spacing=0.0,
            word_spacing=0.0,
            italics=0.0,
            boldness=0.0,
            config=self.config
        )

        # Bit-for-bit identity check
        self.assertTrue(
            np.array_equal(self.strokes, mod_strokes),
            "Zero controls did not preserve exact stroke array identity!"
        )

        # Stroke thickness check
        self.assertEqual(
            lw,
            self.config.default_linewidth,
            "Zero control linewidth does not match default renderer linewidth!"
        )

        # Ensure no in-place mutation of original array
        self.assertIsNot(
            mod_strokes,
            self.strokes,
            "Expected a safe copy of strokes, but received the original object!"
        )

    # ==========================================================================
    # Test B: Character Spacing
    # ==========================================================================
    def test_b_character_spacing(self):
        """Verify that character spacing expands at +1 and contracts at -1."""
        pos_orig = relative_to_absolute(self.strokes)
        span_orig = pos_orig[:, 0].max() - pos_orig[:, 0].min()

        strokes_inc = style_controller.apply_character_spacing(
            self.strokes, self.text, 1.0, config=self.config
        )
        pos_inc = relative_to_absolute(strokes_inc)
        span_inc = pos_inc[:, 0].max() - pos_inc[:, 0].min()

        strokes_dec = style_controller.apply_character_spacing(
            self.strokes, self.text, -1.0, config=self.config
        )
        pos_dec = relative_to_absolute(strokes_dec)
        span_dec = pos_dec[:, 0].max() - pos_dec[:, 0].min()

        strokes_zero = style_controller.apply_character_spacing(
            self.strokes, self.text, 0.0, config=self.config
        )

        # Verification of expansion and contraction
        self.assertGreater(
            span_inc,
            span_orig,
            "Character spacing +1.0 should increase the overall horizontal span!"
        )
        self.assertLess(
            span_dec,
            span_orig,
            "Character spacing -1.0 should decrease the overall horizontal span!"
        )
        self.assertTrue(
            np.array_equal(strokes_zero, self.strokes),
            "Character spacing 0.0 must leave strokes unchanged!"
        )

        # Verify pen lift states are unchanged
        self.assertTrue(
            np.array_equal(strokes_inc[:, 2], self.strokes[:, 2]),
            "Pen lifts were corrupted by character spacing!"
        )

    # ==========================================================================
    # Test C: Word Spacing
    # ==========================================================================
    def test_c_word_spacing(self):
        """Verify that word spacing changes distance between words without altering intra-word geometry."""
        boundaries = style_controller.find_word_boundaries(self.strokes, self.text, self.config)
        self.assertTrue(len(boundaries) > 0, "No word boundary found for multi-word text!")
        b = boundaries[0]

        strokes_inc = style_controller.apply_word_spacing(
            self.strokes, self.text, 1.0, config=self.config
        )
        strokes_dec = style_controller.apply_word_spacing(
            self.strokes, self.text, -1.0, config=self.config
        )
        strokes_zero = style_controller.apply_word_spacing(
            self.strokes, self.text, 0.0, config=self.config
        )

        # 1. Internal geometry of Word 0 must be 100% untouched
        np.testing.assert_allclose(
            strokes_inc[:b, :2],
            self.strokes[:b, :2],
            atol=1e-5,
            err_msg="Word 0 internal coordinates were modified by word spacing!"
        )

        # 2. Distance to Word 1 must increase for +1.0 and decrease for -1.0
        pos_orig = relative_to_absolute(self.strokes)
        pos_inc = relative_to_absolute(strokes_inc)
        pos_dec = relative_to_absolute(strokes_dec)

        gap_orig = pos_orig[b, 0] - pos_orig[b - 1, 0]
        gap_inc = pos_inc[b, 0] - pos_inc[b - 1, 0]
        gap_dec = pos_dec[b, 0] - pos_dec[b - 1, 0]

        self.assertGreater(gap_inc, gap_orig, "Word spacing +1.0 must increase inter-word gap!")
        self.assertLess(gap_dec, gap_orig, "Word spacing -1.0 must decrease inter-word gap!")
        self.assertTrue(
            np.array_equal(strokes_zero, self.strokes),
            "Word spacing 0.0 must leave strokes unchanged!"
        )

    # ==========================================================================
    # Test D: Italics
    # ==========================================================================
    def test_d_italics(self):
        """Verify that italics slants top coordinates forward for +1 and backward for -1."""
        strokes_inc = style_controller.apply_italics(self.strokes, 1.0, config=self.config)
        strokes_dec = style_controller.apply_italics(self.strokes, -1.0, config=self.config)
        strokes_zero = style_controller.apply_italics(self.strokes, 0.0, config=self.config)

        pos_orig = relative_to_absolute(self.strokes)
        pos_inc = relative_to_absolute(strokes_inc)
        pos_dec = relative_to_absolute(strokes_dec)

        # Vertical coordinates y must be completely unchanged
        np.testing.assert_allclose(
            pos_inc[:, 1],
            pos_orig[:, 1],
            atol=1e-6,
            err_msg="Italics must preserve vertical dimensions y!"
        )

        # Top point (maximum y) must slant right for +1 and left for -1
        top_idx = int(np.argmax(pos_orig[:, 1]))
        self.assertGreater(
            pos_inc[top_idx, 0],
            pos_orig[top_idx, 0],
            "Positive italics should slant top points to the right!"
        )
        self.assertLess(
            pos_dec[top_idx, 0],
            pos_orig[top_idx, 0],
            "Negative italics should slant top points to the left!"
        )

        # Zero control identity
        self.assertTrue(
            np.array_equal(strokes_zero, self.strokes),
            "Italics 0.0 must preserve exact strokes!"
        )

    # ==========================================================================
    # Test E: Boldness
    # ==========================================================================
    def test_e_boldness(self):
        """Verify boldness modifies rendering linewidth without modifying stroke geometry."""
        lw_inc = style_controller.compute_boldness_linewidth(1.0, self.config)
        lw_dec = style_controller.compute_boldness_linewidth(-1.0, self.config)
        lw_zero = style_controller.compute_boldness_linewidth(0.0, self.config)

        self.assertGreater(lw_inc, self.config.default_linewidth)
        self.assertLess(lw_dec, self.config.default_linewidth)
        self.assertEqual(lw_zero, self.config.default_linewidth)
        self.assertEqual(lw_inc, self.config.max_linewidth)
        self.assertEqual(lw_dec, self.config.min_linewidth)

        # Applying boldness only via apply_all_styles must leave geometry untouched
        mod_strokes, lw = style_controller.apply_all_styles(
            self.strokes, self.text, boldness=1.0, config=self.config
        )
        self.assertTrue(
            np.array_equal(mod_strokes, self.strokes),
            "Boldness must NOT modify stroke coordinates!"
        )
        self.assertEqual(lw, lw_inc)

    # ==========================================================================
    # Test F: Combined Controls & Edge Cases
    # ==========================================================================
    def test_f_combined_controls(self):
        """Verify that all four style controllers can be applied simultaneously."""
        mod_strokes, lw = style_controller.apply_all_styles(
            self.strokes,
            self.text,
            char_spacing=0.5,
            word_spacing=0.7,
            italics=0.4,
            boldness=0.6,
            config=self.config
        )

        self.assertEqual(len(mod_strokes), len(self.strokes))
        self.assertTrue(np.all(np.isfinite(mod_strokes)))
        self.assertGreater(lw, self.config.default_linewidth)

    def test_f_input_clamping(self):
        """Verify that out-of-range values are clamped to [-1.0, 1.0]."""
        mod_clamped, lw_clamped = style_controller.apply_all_styles(
            self.strokes,
            self.text,
            char_spacing=2.5,
            word_spacing=3.0,
            italics=5.0,
            boldness=10.0,
            config=self.config
        )
        mod_max, lw_max = style_controller.apply_all_styles(
            self.strokes,
            self.text,
            char_spacing=1.0,
            word_spacing=1.0,
            italics=1.0,
            boldness=1.0,
            config=self.config
        )
        np.testing.assert_allclose(mod_clamped, mod_max, atol=1e-6)
        self.assertEqual(lw_clamped, lw_max)

    def test_f_single_word_text(self):
        """Verify single-word text works without error and leaves word spacing untouched."""
        mod_strokes, _ = style_controller.apply_all_styles(
            self.strokes,
            "HELLO",
            char_spacing=0.5,
            word_spacing=0.5,
            config=self.config
        )
        self.assertEqual(len(mod_strokes), len(self.strokes))

    def test_f_single_char_text(self):
        """Verify single character text works gracefully."""
        mod_strokes, _ = style_controller.apply_all_styles(
            self.strokes,
            "A",
            char_spacing=0.8,
            word_spacing=0.8,
            italics=0.5,
            boldness=0.5,
            config=self.config
        )
        self.assertEqual(len(mod_strokes), len(self.strokes))

    def test_f_empty_text(self):
        """Verify empty text does not fail and preserves strokes."""
        mod_strokes, _ = style_controller.apply_all_styles(
            self.strokes,
            "",
            char_spacing=0.5,
            word_spacing=0.5,
            italics=0.5,
            config=self.config
        )
        self.assertEqual(len(mod_strokes), len(self.strokes))

    def test_f_coordinate_roundtrip_precision(self):
        """Verify that relative <-> absolute coordinate roundtrip has 0.0 drift."""
        pos = relative_to_absolute(self.strokes)
        dx_reconstructed = absolute_to_relative(pos, self.strokes[:, 2:3], orig_dtype=self.strokes.dtype)
        max_diff = np.max(np.abs(self.strokes[:, :2] - dx_reconstructed[:, :2]))
        self.assertEqual(max_diff, 0.0, "Relative <-> Absolute conversion introduced numerical drift!")

    def test_f_pen_lift_invariance(self):
        """Verify that pen lift column [:, 2] is invariant under all transformations."""
        mod_strokes, _ = style_controller.apply_all_styles(
            self.strokes,
            self.text,
            char_spacing=0.8,
            word_spacing=-0.6,
            italics=0.7,
            boldness=-0.4,
            config=self.config
        )
        self.assertTrue(
            np.array_equal(self.strokes[:, 2], mod_strokes[:, 2]),
            "Pen lift flags were altered by style controller!"
        )


if __name__ == '__main__':
    unittest.main()
