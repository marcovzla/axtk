import textwrap
import jinja2
import jinja2.meta
from natsort import natsorted


class Template:
    """
    A class for creating and rendering Jinja2 templates with optional preprocessing and variable checking.

    Args:
        source (str): The template source code.
        strict (bool, optional): If True, enforces checking for missing variables during rendering. Defaults to True.
        rstrip (bool, optional): Remove trailing whitespace from each line. Defaults to True.
        dedent (bool, optional): Dedent the template code. Defaults to True.
        trim_blocks (bool, optional): If True, the first newline after a block is removed (block, not variable tag!). Defaults to True.
        lstrip_blocks (bool, optional): If True, leading spaces and tabs are stripped from the start of a line to a block. Defaults to True.
        newline_sequence (str, optional): The sequence used for newline characters. Defaults to '\\n'.
        keep_trailing_newline (bool, optional): Keep the trailing newline in the template. Defaults to False.
        autoescape (bool, optional): Enable XML/HTML autoescaping. Defaults to False.

    Attributes:
        source (str): The processed template source code.
        strict (bool): Whether strict mode is enabled or not.
        template (jinja2.Template): The Jinja2 template object.
        input_variables (set[str]): A set of undeclared variables in the template.
    """

    def __init__(
            self,
            source: str,
            *,
            strict: bool = True,
            rstrip: bool = True,
            dedent: bool = True,
            trim_blocks: bool = True,
            lstrip_blocks: bool = True,
            newline_sequence: str = '\n',
            keep_trailing_newline: bool = False,
            autoescape: bool = False,
    ):
        """
        Initialize the Template object.

        Args:
            source (str): The template source code.
            strict (bool, optional): If True, enforces checking for missing variables during rendering. Defaults to True.
            rstrip (bool, optional): Remove trailing whitespace from each line. Defaults to True.
            dedent (bool, optional): Dedent the template code. Defaults to True.
            trim_blocks (bool, optional): If True, the first newline after a block is removed (block, not variable tag!). Defaults to True.
            lstrip_blocks (bool, optional): If True, leading spaces and tabs are stripped from the start of a line to a block. Defaults to True.
            newline_sequence (str, optional): The sequence used for newline characters. Defaults to '\\n'.
            keep_trailing_newline (bool, optional): Keep the trailing newline in the template. Defaults to False.
            autoescape (bool, optional): Enable XML/HTML autoescaping. Defaults to False.
        """
        self.source = source
        self.strict = strict

        # Remove spaces from the end of lines
        if rstrip:
            self.source = newline_sequence.join(
                line.rstrip()
                for line in self.source.splitlines()
            )
            if source.endswith(newline_sequence):
                self.source += newline_sequence

        # Remove indentation
        if dedent:
            self.source = textwrap.dedent(self.source)

        # Create template object
        self.template = jinja2.Template(
            source=self.source,
            trim_blocks=trim_blocks,
            lstrip_blocks=lstrip_blocks,
            newline_sequence=newline_sequence,
            keep_trailing_newline=keep_trailing_newline,
            autoescape=autoescape,
        )

        # Get input variables
        ast = self.template.environment.parse(self.source)
        self.input_variables = jinja2.meta.find_undeclared_variables(ast)

    def __call__(self, *args, **kwargs) -> str:
        """
        Render the template with provided variables.

        This method accepts the same arguments as the dict constructor: A dict, a dict subclass, or some keyword arguments.
        If self.strict is set to True and some variables are missing, a ValueError will be raised.

        Args:
            *args: Positional arguments as dictionaries with input variables.
            **kwargs: Keyword arguments corresponding to input variables.

        Returns:
            str: The rendered template as a string.

        Raises:
            ValueError: If self.strict is set to True and some variables are missing in the input.
        """
        # Ensure all input variables were provided
        if self.strict:
            if missing := self.missing_variables(*args, **kwargs):
                missing = ', '.join(missing)
                raise ValueError(f'Missing variables: {missing}')

        # Render the template
        text = self.template.render(*args, **kwargs)
        return text

    def missing_variables(self, *args, **kwargs) -> list[str]:
        """
        Get a list of undeclared variables that are missing in the provided variables.

        Args:
            *args: Positional arguments as dictionaries with input variables.
            **kwargs: Keyword arguments corresponding to input variables.

        Returns:
            list[str]: A list of variable names that are undeclared in the template and missing in the provided variables.
        """
        variables = dict(*args, **kwargs)
        return natsorted(self.input_variables - variables.keys())
