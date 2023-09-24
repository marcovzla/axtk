from asyncio import AbstractEventLoop
from collections.abc import Iterable, Iterator, AsyncIterator
from typing import Any


def async_to_sync_iterator(ait: AsyncIterator[Any], loop: AbstractEventLoop) -> Iterator[Any]:
    """
    Convert an asynchronous iterator to a synchronous one using the provided event loop.
    
    Args:
        ait (AsyncIterator[Any]): The asynchronous iterator to be converted.
        loop (asyncio.AbstractEventLoop): The event loop to run the asynchronous tasks.
        
    Yields:
        Any: Values yielded by the asynchronous iterator.
    """
    async def get_next():
        try:
            obj = await anext(ait)
            return False, obj
        except StopAsyncIteration:
            return True, None
    while True:
        done, obj = loop.run_until_complete(get_next())
        if done: break
        yield obj


class AsyncIterableWrapper:
    """
    Wraps a regular iterable to be used as an asynchronous iterator.
    
    This class allows for iterating over the provided items using the async 
    for loop syntax, even if the underlying iterable is not natively asynchronous.

    Args:
        items (Iterable[T]): The underlying iterable to wrap.

    Example:
        async for item in AsyncIterableWrapper([1, 2, 3]):
            print(item)
    """

    def __init__(self, items: Iterable[Any]):
        self._items = iter(items)  # Convert items to an iterator

    def __aiter__(self):
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration
