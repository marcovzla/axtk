import sys


def running_in_notebook() -> bool:
    """
    Determine if the current environment is a Jupyter Notebook.

    Returns:
        bool: True if running in a Jupyter Notebook, False otherwise.

    Notes:
        The function checks for the existence of the 'IPython' module 
        and the presence of 'IPKernelApp' in its configuration to infer
        if the code is being executed within a Jupyter Notebook.
    """
    try:
        get_ipython = sys.modules['IPython'].get_ipython
        return 'IPKernelApp' in get_ipython().config
    except KeyError:
        # 'IPython' not in sys.modules
        return False
    except AttributeError:
        # get_ipython or config not available
        return False


def running_in_colab() -> bool:
    """
    Determine if the current environment is Google Colab.

    Returns:
        bool: True if running in Google Colab, False otherwise.
    """
    return 'google.colab' in sys.modules
