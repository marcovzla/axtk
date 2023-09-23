import re
from typing import Optional



def delimited_string(
        delimiters: str,
        close_delimiters: Optional[str] = None,
        escape_chars: Optional[str] = '\\',
        return_string: bool = False,
) -> str | re.Pattern[str]:
    """
    Create a regex pattern to capture strings between matching delimiters.

    Parameters:
        delimiters (str): Opening delimiters.
        close_delimiters (Optional[str]): Closing delimiters. Defaults to the same as delimiters.
        escape_chars (Optional[str]): Characters to treat as escape characters.
        return_string (bool): If True, returns the regex pattern as a string; else, a compiled regex.

    Returns:
        str | regex.Pattern: The regex pattern.
    """
    
    # Ensure we have at least one delimiter
    if not delimiters:
        raise ValueError('At least one delimiter is required.')

    # Default to using the same delimiter for closing if none is provided
    close_delimiters = close_delimiters or delimiters

    # Check for matching lengths of delimiters
    if len(delimiters) != len(close_delimiters):
        raise ValueError('Mismatch between open and close delimiters.')

    # Duplicate a single escape character for all delimiters if necessary
    if escape_chars and len(escape_chars) == 1:
        escape_chars *= len(delimiters)

    # Check for matching lengths of delimiters and escape characters
    if escape_chars and len(escape_chars) != len(delimiters):
        raise ValueError('Mismatch between delimiters and escape characters.')

    # Construct patterns
    patterns = []
    for i in range(len(delimiters)):
        open_delim = re.escape(delimiters[i])
        close_delim = re.escape(close_delimiters[i])
        if escape_chars is None:
            pattern = f'{open_delim}[^{close_delim}]*{close_delim}'
        else:
            esc_char = re.escape(escape_chars[i])
            if esc_char == close_delim:
                pattern = f'{open_delim}[^{close_delim}]*(?:{close_delim}{close_delim}[^{close_delim}]*)*{close_delim}'
            else:
                pattern = f'{open_delim}[^{esc_char}{close_delim}]*(?:{esc_char}.[^{esc_char}{close_delim}]*)*{close_delim}'
        patterns.append(pattern)

    # Combine patterns using a non-capturing group
    pattern = '(?:' + '|'.join(patterns) + ')'

    # Return the pattern either as a string or as a compiled regex
    return pattern if return_string else re.compile(pattern)



def bracketed_string(
        brackets: str,
        escape_chars: Optional[str] = '\\',
        return_string: bool = False,
) -> str | re.Pattern[str]:
    """
    Create a regex pattern to capture strings between matching brackets.

    Parameters:
        brackets (str): String containing pairs of brackets. e.g., "{}[]".
        escape_chars (Optional[str]): Characters to treat as escape characters.
        return_string (bool): If True, returns the regex pattern as a string; else, a compiled regex.

    Returns:
        str | regex.Pattern: The regex pattern.
    """
    
    # Ensure the provided brackets string contains even number of characters (pairs)
    if len(brackets) % 2 != 0:
        raise ValueError('The brackets string should contain pairs of opening and closing brackets.')

    return delimited_string(
        delimiters=brackets[::2],
        close_delimiters=brackets[1::2],
        escape_chars=escape_chars,
        return_string=return_string,
    )



def anything_except(
        *args: str,
        escape_args: bool = True,
        allow_spaces: bool = True,
        allow_newline: bool = False,
        allow_empty_match: bool = False,
        return_string: bool = False,
) -> str | re.Pattern[str]:
    """
    Returns a regex pattern that matches any string except the provided arguments.

    Parameters:
        *args (str): Strings to exclude.
        escape_args (bool): If True, escapes regex metacharacters in provided args.
        allow_spaces (bool): If True, pattern can match spaces.
        allow_newline (bool): If True, pattern can match newlines.
        allow_empty_match (bool): If True, pattern allows for an empty match.
        return_string (bool): If True, returns the regex pattern as a string; else, a compiled regex.

    Returns:
        str | regex.Pattern: The regex pattern.
    """

    # Check for invalid combination of arguments
    if allow_newline and not allow_spaces:
        raise ValueError("Cannot allow newlines without allowing spaces.")

    # Escape regex metacharacters if needed
    if escape_args:
        args = [re.escape(arg) for arg in args]

    # Create negative lookahead for arguments to exclude
    negative_lookahead = '|'.join(f'(?:{arg})' for arg in args)

    # Determine the character set to match
    if allow_spaces:
        char_match = '(?s:.)' if allow_newline else '.'
    else:
        char_match = r'\S'

    # Construct the pattern using negative lookahead and the char_match
    pattern = f'(?:(?!{negative_lookahead}){char_match})'

    # Add quantifier based on the allow_empty_match flag
    pattern += '*' if allow_empty_match else '+'

    # Wrap in non-capturing parenthesis
    pattern = f'(?:{pattern})'

    # Return the pattern either as a string or as a compiled regex
    return pattern if return_string else re.compile(pattern)
