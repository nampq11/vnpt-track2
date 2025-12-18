"""Retry utilities for API calls with exponential backoff"""
import asyncio
import time
import random
from functools import wraps
from typing import Callable, Type, Tuple
from loguru import logger


def retry_sync(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    max_backoff: float = 8.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for retrying synchronous functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_base: Base for exponential backoff in seconds (default: 2.0)
        max_backoff: Maximum backoff time in seconds (default: 8.0)
        jitter: Add randomization to backoff to prevent thundering herd (default: True)
        exceptions: Tuple of exceptions to catch and retry on
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {str(e)}"
                        )
                        raise
                    
                    # Calculate backoff with exponential increase
                    backoff = min(backoff_base ** attempt, max_backoff)
                    
                    # Add jitter (±20% randomization)
                    if jitter:
                        backoff = backoff * (0.8 + random.random() * 0.4)
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                        f"Retrying in {backoff:.2f}s..."
                    )
                    
                    time.sleep(backoff)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def retry_async(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    max_backoff: float = 8.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for retrying asynchronous functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_base: Base for exponential backoff in seconds (default: 2.0)
        max_backoff: Maximum backoff time in seconds (default: 8.0)
        jitter: Add randomization to backoff to prevent thundering herd (default: True)
        exceptions: Tuple of exceptions to catch and retry on
    
    Returns:
        Decorated async function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {str(e)}"
                        )
                        raise
                    
                    # Calculate backoff with exponential increase
                    backoff = min(backoff_base ** attempt, max_backoff)
                    
                    # Add jitter (±20% randomization)
                    if jitter:
                        backoff = backoff * (0.8 + random.random() * 0.4)
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                        f"Retrying in {backoff:.2f}s..."
                    )
                    
                    await asyncio.sleep(backoff)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

