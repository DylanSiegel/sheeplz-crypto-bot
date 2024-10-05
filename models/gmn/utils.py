import functools
from logger import logger

def retry(exceptions, tries=3, delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(tries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < tries - 1:
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {tries} attempts failed for function {func.__name__}")
                        raise
        return wrapper
    return decorator