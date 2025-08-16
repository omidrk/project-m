import functools
import json
import pickle
import hashlib
from typing import Any, Callable, Optional, Tuple, Dict

import redis

client = redis.Redis(host="localhost", port=6379, db=0)


def get_client() -> redis.Redis:
    if client:
        print("reuse client")
        return client
    print("making new client")
    return redis.Redis(host="localhost", port=6379, db=0)


def _make_cache_key(
    func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> str:
    """
    Create a cache key based on function's module, name, and a hash of its arguments.
    This key will be used to store/retrieve values in Redis.

    Args:
        func: The function being cached.
        args: Positional arguments passed to the function.
        kwargs: Keyword arguments passed to the function.

    Returns:
        A string that uniquely identifies this call, hashed for safe key length.
    """
    prefix = f"{func.__module__}.{func.__name__}"
    # Serialize args and kwargs deterministically
    try:
        serialized = json.dumps(
            {"args": args, "kwargs": kwargs},
            default=repr,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        # Fallback: pickle the args
        serialized = pickle.dumps((args, kwargs))

    # Compute SHA256 hash of the serialized args
    arg_hash = hashlib.sha256(serialized).hexdigest()
    return f"{prefix}:{arg_hash}"


def redis_cache(
    redis_client: redis.Redis = client, ttl: Optional[int] = None
) -> Callable:
    """
    Decorator factory to cache function results in Redis.

    Args:
        redis_client: A Redis client instance.
        ttl: Optional time-to-live for cache entries, in seconds.

    Returns:
        A decorator that caches function calls.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _make_cache_key(func, args, kwargs)
            # Attempt to fetch from cache
            try:
                cached = redis_client.get(key)
            except redis.RedisError:
                cached = None

            if cached is not None:
                print("cache found")
                try:
                    # Unpickle the result
                    return pickle.loads(cached)
                except (pickle.PickleError, TypeError):
                    # If unpickling fails, delete corrupted cache
                    try:
                        redis_client.delete(key)
                    except redis.RedisError:
                        pass
            print("no cache found")
            # Call the actual function and cache the result
            result = func(*args, **kwargs)
            try:
                redis_client.set(name=key, value=pickle.dumps(result), ex=ttl)
            except redis.RedisError:
                # If caching fails, ignore and return the result
                pass
            return result

        return wrapper

    return decorator


def empty_redis_cache(redis_client: redis.Redis = client) -> None:
    """
    Clear all cache entries from Redis.

    Args:
        redis_client: Redis client instance to connect to the cache.
    """
    # Get all keys in the Redis database
    for key in redis_client.keys():
        print(key, redis_client.type(key))
        # if redis_client.type(key) == "string":  # Ensure we're only deleting string keys
        redis_client.delete(key)
        # redis_client.
