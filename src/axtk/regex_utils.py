from typing import Optional, Union
import re
import regex



def delimited_string(
        delimiters: str,
        close_delimiters: Optional[str] = None,
        escape_chars: Optional[str] = '\\',
        return_string: bool = False,
) -> str | regex.Pattern[str]:
    """
    Create a regex pattern to capture strings between matching delimiters.

    Parameters:
        delimiters (str): Opening delimiters.
        close_delimiters (Optional[str]): Closing delimiters. Defaults to the same as delimiters.
        escape_chars (Optional[str]): Characters to treat as escape characters.
        return_string (bool): If True, returns the regex pattern as a string; else, a compiled regex.

    Returns:
        str | regex.Pattern[str]: The regex pattern.
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
        open_delim = regex.escape(delimiters[i])
        close_delim = regex.escape(close_delimiters[i])
        if escape_chars is None:
            pattern = f'{open_delim}[^{close_delim}]*{close_delim}'
        else:
            esc_char = regex.escape(escape_chars[i])
            if esc_char == close_delim:
                pattern = f'{open_delim}[^{close_delim}]*(?:{close_delim}{close_delim}[^{close_delim}]*)*{close_delim}'
            else:
                pattern = f'{open_delim}[^{esc_char}{close_delim}]*(?:{esc_char}.[^{esc_char}{close_delim}]*)*{close_delim}'
        patterns.append(pattern)

    # Combine patterns using a non-capturing group
    pattern = '(?:' + '|'.join(patterns) + ')'

    # Return the pattern either as a string or as a compiled regex
    return pattern if return_string else regex.compile(pattern)



def bracketed_string(
        brackets: str,
        escape_chars: Optional[str] = '\\',
        return_string: bool = False,
) -> str | regex.Pattern[str]:
    """
    Create a regex pattern to capture strings between matching brackets.

    Parameters:
        brackets (str): String containing pairs of brackets. e.g., "{}[]".
        escape_chars (Optional[str]): Characters to treat as escape characters.
        return_string (bool): If True, returns the regex pattern as a string; else, a compiled regex.

    Returns:
        str | regex.Pattern[str]: The regex pattern.
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
) -> str | regex.Pattern[str]:
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
        str | regex.Pattern[str]: The regex pattern.
    """

    # Check for invalid combination of arguments
    if allow_newline and not allow_spaces:
        raise ValueError("Cannot allow newlines without allowing spaces.")

    # Escape regex metacharacters if needed
    if escape_args:
        args = [regex.escape(arg) for arg in args]

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
    return pattern if return_string else regex.compile(pattern)



def integer(
        base: int = 10,
        sep: Optional[str] = None,
        group: Optional[Union[int, tuple[int, int]]] = 3,
        places: Optional[Union[int, tuple[int, int]]] = None,
        sign: Optional[str] = '[-+]?',
        return_string: bool = False,
) -> str | regex.Pattern[str]:
    """
    Returns a regex pattern that matches integers in a specific numeral base
    with optional formatting.

    Args:
        base (int, optional): The numeral base for the integer. Must be between 2 and 36.
            Default is 10.
        sep (str, optional): The character used to separate groups of digits. Default is None.
        group (int or tuple[int, int], optional): The number of digits in each group, 
            either as a fixed number or a range. This is ignored unless `sep` is provided.
            Default is 3.
        places (int or tuple[int, int], optional): Specifies the total number of places/digits 
            the integer should have, either as a fixed number or a range. This is ignored if `sep` 
            is provided. Default is None.
        sign (str, optional): Regular expression pattern to match the sign of the integer. 
            Default is '[-+]?' which matches optional minus or plus signs.
        return_string (bool, optional): If True, returns the pattern as a string. 
            If False, returns a compiled regex pattern. Default is False.

    Returns:
        str | regex.Pattern[str]: The regex pattern.
    """
    # Get valid characters for the base
    valid_chars = ''.join(valid_digits_for_base(base))

    # Separator handling
    if sep:
        if group is None:
            core_pattern = f'[{valid_chars}]+(?:[{valid_chars}{sep}]*[{valid_chars}])?'
        elif isinstance(group, tuple):
            grp_pattern = f'(?:{sep}([{valid_chars}]{{{group[0]},{group[1]}}})'
            core_pattern = f'[{valid_chars}]{{1,{group[1]}}}{grp_pattern}*'
        else:
            grp_pattern = f'(?:{sep}[{valid_chars}]{{{group}}})'
            core_pattern = f'[{valid_chars}]{{1,{group}}}{grp_pattern}*'
    else:
        if places:
            if isinstance(places, tuple):
                core_pattern = f'[{valid_chars}]{{{places[0]},{places[1]}}}'
            else:
                core_pattern = f'[{valid_chars}]{{{places}}}'
        else:
            core_pattern = f'[{valid_chars}]+'

    # Add sign pattern if required
    pattern = f'{sign}{core_pattern}' if sign else core_pattern

    # Wrap in non-capturing parenthesis
    pattern = f'(?:{pattern})'

    # Return the pattern either as a string or as a compiled regex
    return pattern if return_string else regex.compile(pattern)



def floating_point(
        base: int = 10,
        radix: str = '[.]',
        places: Optional[Union[int, tuple[int, int]]] = None,
        sep: Optional[str] = None,
        group: Optional[Union[int, tuple[int, int]]] = 3,
        expon: Optional[str] = '[Ee]',
        sign: Optional[str] = '[-+]?',
        return_string: bool = False,
) -> str | regex.Pattern[str]:
    """
    Returns a regex pattern that matches floating-point numbers in a specific
    numeral base with optional formatting.

    Args:
        base (int, optional): The numeral base for the number. Must be between 2 and 36.
            Default is 10.
        radix (str, optional): The character or pattern representing the radix point.
            Default is '[.]', which matches a dot.
        places (int or tuple[int, int], optional): Specifies the number of digits after 
            the radix point. It can be a fixed number or a range. If omitted, it allows 
            any number of digits after the radix point. Default is None.
        sep (str, optional): The character used to separate groups of digits in the 
            pre-radix section of the number. By default, no grouping is assumed. Default is None.
        group (int or tuple[int, int], optional): The number of digits in each group in the 
            pre-radix section, either as a fixed number or a range. This is relevant only if `sep`
            is provided. Default is 3.
        expon (str or None, optional): The character or pattern used to represent the exponential 
            part. If set to `None`, no exponential part is allowed in the pattern. Default is '[Ee]', 
            which matches 'E' or 'e'.
        sign (str, optional): Regular expression pattern to match the sign of the number or 
            its exponent. Default is '[-+]?', which matches optional minus or plus signs.
        return_string (bool, optional): If True, returns the pattern as a string. 
            If False, returns a compiled regex pattern. Default is False.

    Returns:
        str | regex.Pattern[str]: The regex pattern.
    """

    # Integral part
    int_pattern = integer(base=base, sep=sep, group=group, places=None, sign=sign, return_string=True)

    # Get valid characters for the base
    valid_chars = ''.join(valid_digits_for_base(base))

    # Fractional part
    if places:
        if isinstance(places, tuple):
            frac_pattern = f'{radix}[{valid_chars}]{{{places[0]},{places[1]}}}'
        else:
            frac_pattern = f'{radix}[{valid_chars}]{{{places}}}'
    else:
        frac_pattern = f'{radix}[{valid_chars}]*'

    # Decimal pattern
    pattern = f'{sign}(?:{int_pattern}(?:{frac_pattern})?|{frac_pattern})'

    # Exponential part
    if expon is not None:
        exponent_pattern = f'{expon}{sign}[{valid_chars}]+'
        pattern += f'(?:{exponent_pattern})?'

    # Wrap in non-capturing parenthesis
    pattern = f'(?:{pattern})'

    # Return the pattern either as a string or as a compiled regex
    return pattern if return_string else regex.compile(pattern)



def valid_digits_for_base(base: int) -> list[str]:
    """
    Generates a list of valid digit characters for a given numeral system base.

    For bases 2 to 10, the list will contain numerical digits ('0' to '9').
    For bases greater than 10, the list will include additional lowercase alphabetic
    characters ('a' to 'z'), starting from 'a' for base 11, 'b' for base 12, and so on,
    up to 'z' for base 36.

    Args:
        base (int): The base of the numeral system. Must be between 2 and 36, inclusive.

    Returns:
        list[str]: A list of valid digit characters for the given base.

    Raises:
        ValueError: If the base is not between 2 and 36.
    """
    if not (2 <= base <= 36):
        raise ValueError('Base should be between 2 and 36')

    # Valid digits for bases 2 to 10
    digits = [str(i) for i in range(base) if i < 10]

    # For bases greater than 10
    if base > 10:
        # Using ASCII values for lowercase letters to get the additional valid characters
        digits += [chr(i) for i in range(ord('a'), ord('a') + (base-10))]

    return digits



REGEX_LITERAL_PATTERN: regex.Pattern[str] = delimited_string('/')
"""Pattern that matches strings delimited by forward slashes."""



def get_pattern_from_regex_literal(
        string: str,
        return_string: bool = False,
) -> str | regex.Pattern[str]:
    """
    Extracts the inner pattern from a regex literal.
    
    This function assumes that a regex literal starts and ends with a forward slash (/).
    It extracts the inner pattern and unescapes forward slashes.
    
    Args:
        string (str): The regex literal to be processed.
        return_string (bool, optional): If True, returns the pattern as a string. 
            If False, returns a compiled regex pattern. Default is False.
        
    Returns:
        str: The extracted regex pattern with unescaped forward slashes.
        
    Raises:
        ValueError: If the input string doesn't meet the criteria for being a regex literal.
    """
    if not REGEX_LITERAL_PATTERN.fullmatch(string):
        raise ValueError(f'Invalid Regex literal: {string}')
    pattern = unescape_forward_slashes(string[1:-1])
    return pattern if return_string else regex.compile(pattern)



def get_regex_literal_from_pattern(pattern: str | re.Pattern[str] | regex.Pattern[str]) -> str:
    """
    Converts a regex pattern to its regex literal representation.

    Args:
        pattern (str | re.Pattern[str] | regex.Pattern[str]): The pattern to be converted to a regex literal.

    Returns:
        str: The regex literal representation of the pattern.
    """
    if not isinstance(pattern, str):
        pattern = pattern.pattern
    pattern = escape_forward_slashes(pattern)
    return f'/{pattern}/'



def escape_forward_slashes(string: str) -> str:
    """
    Escapes forward slashes in a given string.

    This function escapes each forward slash (/) in the given string, 
    unless it is already escaped by an odd number of backslashes.

    Args:
        string (str): The input string to be processed.

    Returns:
        str | regex.Pattern[str]: The string with forward slashes appropriately escaped.
    """
    # Use regex to find non-escaped forward slashes and escape them
    return regex.sub(r'(?<!(?:\\\\)*\\)/', r'\/', string)



def unescape_forward_slashes(string: str) -> str:
    """
    Unescapes forward slashes in a given string.

    This function unescapes each escaped forward slash (\/) in the given string, 
    considering if they are preceded by an even number of backslashes.

    Args:
        string (str): The input string to be processed.

    Returns:
        str: The string with escaped forward slashes appropriately unescaped.
    """
    # Use regex to find escaped forward slashes and unescape them
    return regex.sub(r'(?<=(?:\\\\)*)\\/', '/', string)
