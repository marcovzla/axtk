import re


def split_snakecase(identifier: str) -> list[str]:
    """
    Split a snake case string into its components.
    
    Args:
        identifier (str): A string in snake case format.
        
    Returns:
        list[str]: List of components extracted from the identifier.
    """
    return identifier.split('_')

def join_snakecase(parts: list[str], *, preserve_case: bool = False) -> str:
    """
    Join a list of strings into a snake case identifier.
    
    Args:
        parts (list[str]): List of string components.
        preserve_case (bool, optional): If set to True, the original case of parts is preserved.
            Otherwise, all parts are converted to lowercase. Default is False.
            
    Returns:
        str: A string in snake case format.
    """
    if not preserve_case:
        parts = [p.lower() for p in parts]
    return '_'.join(parts)

def abbreviate_snakecase(identifier: str) -> str:
    """
    Abbreviate a snake case string by taking the first character of each component.
    
    Args:
        identifier (str): A string in snake case format.
        
    Returns:
        str: Abbreviated string.
    """
    parts = split_snakecase(identifier)
    # Ensure that each part has at least one character
    parts = [p[0].lower() if p else '' for p in parts]
    return ''.join(parts)


def split_kebab_case(identifier: str) -> list[str]:
    return identifier.split('-')

def join_kebab_case(parts: str, *, preserve_case: bool = False) -> list[str]:
    if not preserve_case:
        parts = [p.lower() for p in parts]
    return '-'.join(parts)

def abbreviate_kebab_case(identifier: str) -> str:
    parts = split_kebab_case(identifier)
    parts = [p[0].lower() for p in parts]
    return ''.join(parts)


def split_camel_case(identifier: str) -> list[str]:
    matches = re.finditer(r'.+?(?:(?<=[0-9])(?=[A-Z])|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def join_camel_case(parts: list[str], *, first_lowercase: bool = False) -> str:
    parts = [p.capitalize() for p in parts]
    if first_lowercase:
        parts[0] = parts[0].lower()
    return ''.join(parts)

def abbreviate_camel_case(identifier: str) -> str:
    parts = split_camel_case(identifier)
    parts = [p[0].lower() for p in parts]
    return ''.join(parts)


def snakecase_to_camel_case(identifier: str, *, first_lowercase: bool = False) -> str:
    return join_camel_case(split_snakecase(identifier), first_lowercase=first_lowercase)

def snakecase_to_kebab_case(identifier: str, *, preserve_case: bool = False) -> str:
    return join_kebab_case(split_snakecase(identifier), preserve_case=preserve_case)

def kebab_case_to_camel_case(identifier: str, *, first_lowercase: bool = False) -> str:
    return join_camel_case(split_kebab_case(identifier), first_lowercase=first_lowercase)

def kebab_case_to_snakecase(identifier: str, *, preserve_case: bool = False) -> str:
    return join_snakecase(split_kebab_case(identifier), preserve_case=preserve_case)

def camel_case_to_snakecase(identifier: str, *, preserve_case: bool = False) -> str:
    return join_snakecase(split_camel_case(identifier), preserve_case=preserve_case)

def camel_case_to_kebab_case(identifier: str, *, preserve_case: bool = False) -> str:
    return join_kebab_case(split_camel_case(identifier), preserve_case=preserve_case)
