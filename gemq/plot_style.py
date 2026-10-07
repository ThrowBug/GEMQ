"""Shared publication figure typography."""


def configure_arial():
    import matplotlib
    from matplotlib import font_manager

    if not any(font.name == "Arial" for font in font_manager.fontManager.ttflist):
        raise RuntimeError(
            "Arial is not available to Matplotlib. Install/register Arial before plotting; "
            "a silent fallback would produce a figure in a different font."
        )
    matplotlib.rcParams.update({
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
