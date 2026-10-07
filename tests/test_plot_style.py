import sys
import types
import unittest
from unittest.mock import patch

from gemq.plot_style import configure_arial


class ArialPlotStyleTest(unittest.TestCase):
    def fake_modules(self, font_names):
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.rcParams = {}
        font_manager = types.ModuleType("matplotlib.font_manager")
        font_manager.fontManager = types.SimpleNamespace(
            ttflist=[types.SimpleNamespace(name=name) for name in font_names]
        )
        matplotlib.font_manager = font_manager
        return matplotlib, font_manager

    def test_sets_arial_and_embeddable_pdf_fonts(self):
        matplotlib, font_manager = self.fake_modules(["DejaVu Sans", "Arial"])
        with patch.dict(sys.modules, {"matplotlib": matplotlib,
                                      "matplotlib.font_manager": font_manager}):
            configure_arial()
        self.assertEqual(matplotlib.rcParams["font.family"], "Arial")
        self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
        self.assertEqual(matplotlib.rcParams["ps.fonttype"], 42)

    def test_missing_arial_is_not_silently_replaced(self):
        matplotlib, font_manager = self.fake_modules(["DejaVu Sans"])
        with patch.dict(sys.modules, {"matplotlib": matplotlib,
                                      "matplotlib.font_manager": font_manager}):
            with self.assertRaisesRegex(RuntimeError, "Arial is not available"):
                configure_arial()
        self.assertEqual(matplotlib.rcParams, {})


if __name__ == "__main__":
    unittest.main()
