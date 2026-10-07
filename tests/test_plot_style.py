import sys
import types
import unittest
from unittest.mock import patch

from gemq.plot_style import configure_arial


class ArialPlotStyleTest(unittest.TestCase):
    def fake_matplotlib(self):
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.rcParams = {}
        return matplotlib

    def test_sets_arial_and_embeddable_pdf_fonts(self):
        matplotlib = self.fake_matplotlib()
        with patch.dict(sys.modules, {"matplotlib": matplotlib}):
            configure_arial()
        self.assertEqual(matplotlib.rcParams["font.family"],
                         ["Arial", "Liberation Sans", "DejaVu Sans"])
        self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
        self.assertEqual(matplotlib.rcParams["ps.fonttype"], 42)

    def test_missing_arial_does_not_block_plotting(self):
        matplotlib = self.fake_matplotlib()
        with patch.dict(sys.modules, {"matplotlib": matplotlib}):
            configure_arial()
        self.assertIn("DejaVu Sans", matplotlib.rcParams["font.family"])


if __name__ == "__main__":
    unittest.main()
