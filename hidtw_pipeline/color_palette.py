"""Shared colour palette used by every figure produced by this package.

Combines Okabe-Ito (colour-blind safe) with Paul Tol's bright series. All
figures use the same hex codes so multi-panel comparisons stay coherent
without per-figure styling.
"""

PALETTE = {
    "primary":   "#332288",  # HiDTW signature
    "secondary": "#4477AA",
    "accent":    "#EE6677",
    "ok_blue":   "#0072B2",
    "ok_orange": "#E69F00",
    "ok_green":  "#009E73",
    "ok_pink":   "#CC79A7",
    "ok_yellow": "#F0E442",
    "neutral":   "#999999",
}


def finalize_notebook_figure(fig) -> None:
    """Embed ``fig`` in Jupyter when IPython is active; then close to free memory."""
    import matplotlib.pyplot as plt

    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
            plt.close(fig)
            return
    except ImportError:
        pass
    plt.show()
    plt.close(fig)
