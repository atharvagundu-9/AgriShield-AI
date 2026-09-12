import tomllib
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def contrast_ratio(foreground: str, background: str) -> float:
    def luminance(hex_color: str) -> float:
        channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    first, second = luminance(foreground), luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


class ThemeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (PROJECT_ROOT / ".streamlit" / "config.toml").open("rb") as config_file:
            cls.theme = tomllib.load(config_file)["theme"]

    def test_light_and_dark_body_text_meet_wcag_aa(self):
        for mode in ("light", "dark"):
            colors = self.theme[mode]
            self.assertGreaterEqual(contrast_ratio(colors["textColor"], colors["backgroundColor"]), 4.5)
            sidebar = colors["sidebar"]
            self.assertGreaterEqual(contrast_ratio(sidebar["textColor"], sidebar["backgroundColor"]), 4.5)

    def test_primary_buttons_keep_white_text_readable(self):
        for mode in ("light", "dark"):
            self.assertGreaterEqual(contrast_ratio("#FFFFFF", self.theme[mode]["primaryColor"]), 4.5)
            self.assertGreaterEqual(contrast_ratio("#FFFFFF", self.theme[mode]["sidebar"]["primaryColor"]), 4.5)

    def test_no_forced_light_plot_or_page_theme_remains(self):
        ui_source = (PROJECT_ROOT / "src" / "ui.py").read_text(encoding="utf-8")
        css_source = (PROJECT_ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
        self.assertNotIn("plotly_white", ui_source)
        self.assertNotIn(".stApp", css_source)
        self.assertNotIn("[data-testid=\"stSidebar\"]", css_source)


if __name__ == "__main__":
    unittest.main()
