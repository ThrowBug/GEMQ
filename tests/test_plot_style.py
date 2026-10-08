import sys
import types
import unittest
from unittest.mock import patch

from gemq.plot_style import configure_plot_font


class PlotFontStyleTest(unittest.TestCase):
    def fake_matplotlib(self):
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.rcParams = {}
        return matplotlib

    def test_prefers_times_new_roman_and_sets_base_size(self):
        matplotlib = self.fake_matplotlib()
        with patch.dict(sys.modules, {"matplotlib": matplotlib}):
            configure_plot_font(14)
        self.assertEqual(matplotlib.rcParams["font.family"],
                         ["Times New Roman", "Liberation Serif", "DejaVu Serif"])
        self.assertEqual(matplotlib.rcParams["font.size"], 14)
        self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
        self.assertEqual(matplotlib.rcParams["ps.fonttype"], 42)

    def test_missing_times_new_roman_does_not_block_plotting(self):
        matplotlib = self.fake_matplotlib()
        with patch.dict(sys.modules, {"matplotlib": matplotlib}):
            configure_plot_font(10)
        self.assertIn("DejaVu Serif", matplotlib.rcParams["font.family"])

    def test_rejects_nonpositive_font_size(self):
        matplotlib = self.fake_matplotlib()
        with patch.dict(sys.modules, {"matplotlib": matplotlib}):
            with self.assertRaisesRegex(ValueError, "font_size must be positive"):
                configure_plot_font(0)


if __name__ == "__main__":
    unittest.main()
