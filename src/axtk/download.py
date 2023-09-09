import re
import uuid
import mimetypes
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
import requests
from tqdm.auto import tqdm
from axtk.typing import PathLike



def download(
        url: str,
        file_path: Optional[PathLike] = None,
        *,
        directory: Optional[PathLike] = None,
        overwrite: bool = True,
        show_progress: bool = True,
        chunk_size: int = 1024,
) -> Path:
    """
    Download a file from a given URL.

    Args:
        url (str): URL of the file to be downloaded.
        file_path (Optional[PathLike]): Specific path where the file should be saved (takes priority over `directory`).
        directory (Optional[PathLike]): Directory where the file should be saved. Ignored if `path` is provided.
        overwrite (bool): If True and file already exists, it will be overwritten.
        show_progress (bool): If true, show a download progress bar.
        chunk_size (int): The size of chunks to use when downloading the file.

    Returns:
        Path: The path to the downloaded file.
    """
    with requests.get(url, stream=True) as response:
        # raise HTTPError, if one occurred
        response.raise_for_status()

        # get path to store downloaded file
        file_path = get_file_path(file_path, directory, url, response)

        # validate file path
        if file_path.is_file() and not overwrite:
            raise FileExistsError(f'File {file_path} already exists.')
        elif file_path.is_dir():
            raise IsADirectoryError(f'{file_path} is a directory. Expected a file path.')

        # ensure parent directories exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # create progress bar
        if show_progress:
            total = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total, unit='iB', unit_scale=True, unit_divisor=1024, leave=False)

        # download file
        with file_path.open('wb') as file:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                if show_progress:
                    progress_bar.update(size)

        # close progress bar
        if show_progress:
            progress_bar.close()

        # return path to downloaded file
        return file_path


def get_file_path(file_path: Optional[PathLike], directory: Optional[PathLike], url: str, response: requests.Response) -> Path:
    """Determine the correct file path for a given URL and HTTP response."""
    # ensure file path is a Path object
    file_path = Path(file_path or get_filename(url, response))

    # ensure file path is absolute
    if not file_path.is_absolute():
        directory = get_directory_path(directory)
        file_path = directory / file_path

    # return normalized file path
    return file_path.resolve()


def get_directory_path(directory: Optional[PathLike]) -> Path:
    """Ensure the directory exists or create it."""
    # ensure directory is a Path object
    directory = Path.cwd() if directory is None else Path(directory)

    # ensure directory exists
    if not directory.exists():
        directory.mkdir(parents=True)
    elif not directory.is_dir():
        raise NotADirectoryError(f'Not a directory: {directory}')

    # return normalized directory path
    return directory.resolve()


def get_filename(url: str, response: requests.Response) -> str:
    """Determine the filename for a given URL and HTTP response."""
    return (
        get_filename_from_response(response)
        or get_filename_from_url(url, response)
        or generate_fallback_filename(response)
    )


def get_filename_from_response(response: requests.Response) -> Optional[str]:
    """Extract filename from the content-disposition header of a response."""
    # patterns to find filenames in content-disposition header
    filename_regexes = [
        r'filename="((?:[^"\\]|\\.)+)"',
        r'filename=([^;" ]+)',
    ]

    # search for filename in headers
    if content_disposition := response.headers.get('content-disposition'):
        for regex in filename_regexes:
            if m := re.search(regex, content_disposition):
                return m.group(1)


def get_filename_from_url(url: str, response: Optional[requests.Response] = None) -> str:
    """Generate filename from a URL."""
    # get path from url
    path = Path(urlparse(url).path)

    # get extension from response, if available
    if path.name and not path.suffix and response is not None:
        suffix = get_file_extension_from_response(response)
        path = path.with_suffix(suffix)

    return path.name


def generate_fallback_filename(response: requests.Response) -> str:
    """Generate a fallback filename based on the content type."""
    extension = get_file_extension_from_response(response)
    return f'{uuid.uuid4()}{extension}'


def get_file_extension_from_response(response: requests.Response) -> str:
    """Extract the file extension based on the content type from an HTTP response."""
    if content_type := response.headers.get('content-type'):
        content_type, *_ = content_type.split(';')
        return mimetypes.guess_extension(content_type, strict=False)
    return ''
