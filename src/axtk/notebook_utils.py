import sys


def is_in_notebook() -> bool:
    """Returns True if running in a Jupyter Notebook."""
    try:
        get_ipython = sys.modules['IPython'].get_ipython
        return 'IPKernelApp' in get_ipython().config
    except:
        return False


def is_in_colab() -> bool:
    """Returns True if running in Google Colab."""
    try:
        import google.colab  # type: ignore
        return True
    except:
        return False
