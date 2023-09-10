import re


def split_snakecase(identifier: str) -> list[str]:
    """
    Split a snake_case string into its components.
    
    Args:
        identifier (str): A string in snake_case format.
        
    Returns:
        list[str]: List of components extracted from the identifier.
    """
    return identifier.split('_')

def join_snakecase(parts: list[str], *, preserve_case: bool = False) -> str:
    """
    Join a list of strings into a snake_case identifier.
    
    Args:
        parts (list[str]): List of string components.
        preserve_case (bool, optional): If set to True, the original case of parts is preserved.
            Otherwise, all parts are converted to lowercase. Default is False.
            
    Returns:
        str: A string in snake_case format.
    """
    if not preserve_case:
        parts = [p.lower() for p in parts]
    return '_'.join(parts)

def abbreviate_snakecase(identifier: str) -> str:
    """
    Abbreviate a snake_case string by taking the first character of each component.
    
    Args:
        identifier (str): A string in snake_case format.
        
    Returns:
        str: Abbreviated string.
    """
    parts = split_snakecase(identifier)
    # Ensure that each part has at least one character
    parts = [p[0].lower() if p else '' for p in parts]
    return ''.join(parts)


def split_kebabcase(identifier: str) -> list[str]:
    """
    Split a kebab-case string into its components.
    
    Args:
        identifier (str): A string in kebab-case format.
        
    Returns:
        list[str]: List of components extracted from the identifier.
    """
    return identifier.split('-')

def join_kebabcase(parts: list[str], *, preserve_case: bool = False) -> str:
    """
    Join a list of strings into a kebab-case identifier.
    
    Args:
        parts (list[str]): List of string components.
        preserve_case (bool, optional): If set to True, the original case of parts is preserved.
            Otherwise, all parts are converted to lowercase. Default is False.
            
    Returns:
        str: A string in kebab-case format.
    """
    if not preserve_case:
        parts = [p.lower() for p in parts]
    return '-'.join(parts)

def abbreviate_kebabcase(identifier: str) -> str:
    """
    Abbreviate a kebab-case string by taking the first character of each component.
    
    Args:
        identifier (str): A string in kebab-case format.
        
    Returns:
        str: Abbreviated string.
    """
    parts = split_kebabcase(identifier)
    # Ensure that each part has at least one character
    parts = [p[0].lower() if p else '' for p in parts]
    return ''.join(parts)


def split_camelcase(identifier: str) -> list[str]:
    """
    Split a camelCase string into its components.
    
    Args:
        identifier (str): A string in camelCase format.
        
    Returns:
        list[str]: List of components extracted from the identifier.
    """
    matches = re.finditer(r'.+?(?:(?<=[a-zA-Z])(?=[0-9])|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def join_camelcase(parts: list[str], *, first_lowercase: bool = False) -> str:
    """
    Join a list of strings into a camelCase identifier.
    
    Args:
        parts (list[str]): List of string components.
        first_lowercase (bool, optional): If set to True, the first part of the identifier is in lowercase.
            Otherwise, every part starts with an uppercase letter. Default is False.
            
    Returns:
        str: A string in camelCase format.
    """
    parts = [p.capitalize() for p in parts]
    if first_lowercase:
        parts[0] = parts[0].lower()
    return ''.join(parts)

def abbreviate_camelcase(identifier: str) -> str:
    """
    Abbreviate a camelCase string by taking the first character of each component.
    
    Args:
        identifier (str): A string in camelCase format.
        
    Returns:
        str: Abbreviated string.
    """
    parts = split_camelcase(identifier)
    # Ensure that each part has at least one character
    parts = [p[0].lower() if p else '' for p in parts]
    return ''.join(parts)


def snakecase_to_camelcase(identifier: str, *, first_lowercase: bool = False) -> str:
    """
    Convert a snake_case identifier to camelCase format.

    Args:
        identifier (str): A string in snake_case format.
        first_lowercase (bool, optional): If set to True, the resulting identifier starts with a lowercase letter.
            Default is False.

    Returns:
        str: A string in camelCase format.
    """
    return join_camelcase(split_snakecase(identifier), first_lowercase=first_lowercase)

def snakecase_to_kebabcase(identifier: str, *, preserve_case: bool = False) -> str:
    """
    Convert a snake_case identifier to kebab-case format.

    Args:
        identifier (str): A string in snake_case format.
        preserve_case (bool, optional): If set to True, the original case of identifier is preserved.
            Default is False.

    Returns:
        str: A string in kebab-case format.
    """
    return join_kebabcase(split_snakecase(identifier), preserve_case=preserve_case)

def kebabcase_to_camelcase(identifier: str, *, first_lowercase: bool = False) -> str:
    """
    Convert a kebab-case identifier to camelCase format.

    Args:
        identifier (str): A string in kebab-case format.
        first_lowercase (bool, optional): If set to True, the resulting identifier starts with a lowercase letter.
            Default is False.

    Returns:
        str: A string in camelCase format.
    """
    return join_camelcase(split_kebabcase(identifier), first_lowercase=first_lowercase)

def kebabcase_to_snakecase(identifier: str, *, preserve_case: bool = False) -> str:
    """
    Convert a kebab-case identifier to snake_case format.

    Args:
        identifier (str): A string in kebab-case format.
        preserve_case (bool, optional): If set to True, the original case of identifier is preserved.
            Default is False.

    Returns:
        str: A string in snake_case format.
    """
    return join_snakecase(split_kebabcase(identifier), preserve_case=preserve_case)

def camelcase_to_snakecase(identifier: str, *, preserve_case: bool = False) -> str:
    """
    Convert a camelCase identifier to snake_case format.

    Args:
        identifier (str): A string in camelCase format.
        preserve_case (bool, optional): If set to True, the original case of identifier is preserved.
            Default is False.

    Returns:
        str: A string in snake_case format.
    """
    return join_snakecase(split_camelcase(identifier), preserve_case=preserve_case)

def camelcase_to_kebabcase(identifier: str, *, preserve_case: bool = False) -> str:
    """
    Convert a camelCase identifier to kebab-case format.

    Args:
        identifier (str): A string in camelCase format.
        preserve_case (bool, optional): If set to True, the original case of identifier is preserved.
            Default is False.

    Returns:
        str: A string in kebab-case format.
    """
    return join_kebabcase(split_camelcase(identifier), preserve_case=preserve_case)
