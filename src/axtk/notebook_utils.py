from collections.abc import Iterable
from ftfy import fix_text
from IPython.display import display, Markdown
from axtk import running_in_notebook, running_in_colab


def display_markdown(text: str):
    """
    Display Markdown-formatted text in a Jupyter notebook.

    Args:
        text (str): The text to be displayed as Markdown.
    """
    display(Markdown(fix_text(text)))


def display_markdown_stream(stream: Iterable[str]):
    """
    Display a stream of Markdown-formatted text chunks in a Jupyter notebook.

    Args:
        stream (Iterable[str]): An iterable providing chunks of text to display.
    """
    text = ''
    handle = display(Markdown(text), display_id=True)

    for chunk in stream:
        text += chunk
        handle.update(Markdown(fix_text(text)))
