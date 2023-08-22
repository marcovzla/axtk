import re
from typing import Optional


DECIMAL = re.compile(r'[-+]?(?:0(?:[0_]*0)?|[1-9](?:[0-9_]*[0-9])?)')
"""Matches a decimal number."""

BINARY = re.compile(r'[-+]?0[bB][01_]*[01]')
"""Matches a binary number."""

OCTAL = re.compile(r'[-+]?0[oO][0-7_]*[0-7]')
"""Matches an octal number."""

HEXADECIMAL = re.compile(r'[-+]?0[xX][a-fA-F0-9_]*[a-fA-F0-9]')
"""Matches a hexadecimal number."""

FLOAT = re.compile(r'[-+]?(?:[0-9](?:[0-9_]*[0-9])?)?\.?(?:[0-9](?:[0-9_]*[0-9])?)(?:[eE][-+]?[0-9](?:[0-9_]*[0-9])?)?')
"""Matches a floating point number."""


def bracketed_string(
        brackets: str,
        escape_chars: Optional[str] = '\\',
        return_string: bool = False,
) -> str | re.Pattern[str]:
    return delimited_string(
        delimiters=brackets[::2],
        close_delimiters=brackets[1::2],
        escape_chars=escape_chars,
        return_string=return_string,
    )


def delimited_string(
        delimiters: str,
        close_delimiters: Optional[str] = None,
        escape_chars: Optional[str] = '\\',
        return_string: bool = False,
) -> str | re.Pattern[str]:

    # ensure we have delimiters
    if not delimiters:
        raise ValueError('at least one delimiter is required')

    # if close delimiters were not provided, default to open delimiters
    if close_delimiters is None:
        close_delimiters = delimiters

    # ensure we have a close delimiter for each open delimiter
    if len(delimiters) != len(close_delimiters):
        raise ValueError('delimiters and close_delimiters do not match')

    # if only one escape character was provided, reuse it for all delimiters
    if escape_chars is not None and len(escape_chars) == 1:
        escape_chars *= len(delimiters)

    # ensure we have an escape character for each delimiter
    if escape_chars is not None and len(escape_chars) != len(delimiters):
        raise ValueError('delimiters and escape_chars do not match')

    # make pattern for each delimiter
    patterns = []
    for i in range(len(delimiters)):
        delim = re.escape(delimiters[i])
        cdelim = re.escape(close_delimiters[i])
        if escape_chars is None:
            pattern = f'{delim}[^{cdelim}]*{cdelim}'
        else:
            esc = re.escape(escape_chars[i])
            if esc == cdelim:
                pattern = f'{delim}[^{cdelim}]*(?:{cdelim}{cdelim}[^{cdelim}]*)*{cdelim}'
            else:
                pattern = f'{delim}[^{esc}{cdelim}]*(?:{esc}.[^{esc}{cdelim}]*)*{cdelim}'
        patterns.append(pattern)

    # merge patterns
    if len(patterns) == 1:
        pattern = patterns[0]
    else:
        pattern = '(?:' + '|'.join(patterns) + ')'

    # maybe return string
    if return_string:
        return pattern

    # return compiled pattern
    return re.compile(pattern)
