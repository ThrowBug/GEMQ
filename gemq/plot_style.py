"""Shared publication figure typography."""


def configure_plot_font(font_size):
    import matplotlib

    if font_size <= 0:
        raise ValueError("font_size must be positive")
    matplotlib.rcParams.update({
        "font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": font_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
