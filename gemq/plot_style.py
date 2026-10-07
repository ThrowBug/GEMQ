"""Shared publication figure typography."""


def configure_arial():
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
