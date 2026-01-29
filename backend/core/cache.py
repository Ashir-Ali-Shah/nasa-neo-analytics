"""
Redis Look-Aside Caching Module for NASA NEO API

This module implements a robust caching layer using Redis to minimize
external NASA API calls and respect rate limits.

Architecture Decisions:
1. Fail-Open Logic: If Redis is unavailable, the system falls back to
   calling the API directly - no cache failure should break the app.
2. Arg Hashing: Complex arguments are hashed to keep Redis keys short.
3. Variable TTLs: Different endpoints get different cache durations
   based on data volatility.
"""

import os
import json
import logging
import hashlib
from functools import wraps
from typing import Callable, Optional, Any
from datetime import datetime

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

# Configure structured logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nasa_cache")

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_TTL_SECONDS = int(os.getenv("REDIS_CACHE_TTL_FEED", 86400))  # 24 hours

# Global connection pool for efficient connection reuse
_connection_pool: Optional[ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None

# Cache statistics tracking
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "errors": 0,
    "last_reset": datetime.utcnow().isoformat()
}


async def get_redis_client() -> Optional[redis.Redis]:
    """
    Get or create a Redis client with connection pooling.
    
    Uses a singleton pattern for the connection pool to ensure
    efficient connection reuse across all requests.
    """
    global _connection_pool, _redis_client
    
    if _redis_client is not None:
        try:
            # Test if connection is still alive
            await _redis_client.ping()
            return _redis_client
        except Exception:
            # Connection lost, recreate
            _redis_client = None
            _connection_pool = None
    
    try:
        if _connection_pool is None:
            _connection_pool = ConnectionPool.from_url(
                REDIS_URL,
                max_connections=10,
                decode_responses=True
            )
        
        _redis_client = redis.Redis(connection_pool=_connection_pool)
        await _redis_client.ping()
        logger.info(f"✓ Redis connected at {REDIS_URL}")
        return _redis_client
        
    except Exception as e:
        logger.warning(f"⚠ Redis connection failed: {e}. Operating in pass-through mode.")
        return None


# Convenience accessor for the singleton client
redis_client = get_redis_client


def generate_cache_key(prefix: str, func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Generate a deterministic, compact cache key.
    
    Uses MD5 hashing to keep keys short while maintaining uniqueness.
    Sorts kwargs to ensure {date: A, id: B} equals {id: B, date: A}.
    
    Format: {prefix}:{function_name}:{hash}
    """
    # Create a stable string representation of arguments
    # Sort kwargs to ensure deterministic ordering
    sorted_kwargs = sorted(kwargs.items())
    arg_string = f"{args}-{sorted_kwargs}"
    
    # Use MD5 for compact keys (not for security, just uniqueness)
    arg_hash = hashlib.md5(arg_string.encode()).hexdigest()[:16]
    
    return f"{prefix}:{func_name}:{arg_hash}"


def cache_response(
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    key_prefix: str = "nasa"
) -> Callable:
    """
    Async decorator factory for Look-Aside Caching.
    
    This decorator wraps async functions to:
    1. Check Redis cache first (cache-aside pattern)
    2. On hit: Return cached JSON data immediately
    3. On miss: Execute function, cache result, then return
    4. On Redis failure: Fall back to direct function execution
    
    Args:
        ttl_seconds: Time to live in seconds (default 24h for feed, 7d for lookups)
        key_prefix: Namespace prefix for cache keys (e.g., 'neo_feed', 'neo_lookup')
    
    Returns:
        Decorated async function with caching behavior
    
    Example:
        @cache_response(ttl_seconds=86400, key_prefix="neo_feed")
        async def fetch_asteroid_feed(start_date: str, end_date: str):
            # ... fetch from NASA API
            return data
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            global _cache_stats
            
            # Generate unique cache key
            cache_key = generate_cache_key(key_prefix, func.__name__, args, kwargs)
            
            # Attempt to get Redis client
            client = await get_redis_client()
            
            # STEP 1: Try to fetch from cache (non-blocking on Redis failure)
            if client is not None:
                try:
                    cached_data = await client.get(cache_key)
                    
                    if cached_data is not None:
                        _cache_stats["hits"] += 1
                        logger.info(
                            f"⚡ CACHE HIT | key={cache_key} | "
                            f"func={func.__name__} | stats={_cache_stats['hits']}H/{_cache_stats['misses']}M"
                        )
                        return json.loads(cached_data)
                        
                except redis.RedisError as e:
                    _cache_stats["errors"] += 1
                    logger.error(f"❌ Redis read error: {e}")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error for key {cache_key}: {e}")
                    # Invalid cache entry, will refresh
            
            # STEP 2: Cache miss - execute the actual function
            _cache_stats["misses"] += 1
            logger.info(
                f"🌍 CACHE MISS | key={cache_key} | "
                f"func={func.__name__} | fetching from source..."
            )
            
            result = await func(*args, **kwargs)
            
            # STEP 3: Store result in cache (non-blocking on failure)
            if client is not None and result is not None:
                try:
                    await client.setex(
                        cache_key,
                        ttl_seconds,
                        json.dumps(result)
                    )
                    logger.info(
                        f"💾 CACHED | key={cache_key} | "
                        f"ttl={ttl_seconds}s ({ttl_seconds//3600}h)"
                    )
                except redis.RedisError as e:
                    _cache_stats["errors"] += 1
                    logger.error(f"❌ Redis write error: {e}")
                except (TypeError, ValueError) as e:
                    logger.error(f"❌ JSON serialize error: {e}")
            
            return result
        
        return wrapper
    return decorator


async def get_cache_stats() -> dict:
    """
    Get current cache statistics for observability.
    
    Returns:
        Dictionary containing hit/miss counts, hit ratio, and connection status
    """
    global _cache_stats
    
    client = await get_redis_client()
    
    total = _cache_stats["hits"] + _cache_stats["misses"]
    hit_ratio = (_cache_stats["hits"] / total * 100) if total > 0 else 0
    
    stats = {
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "errors": _cache_stats["errors"],
        "total_requests": total,
        "hit_ratio_percent": round(hit_ratio, 2),
        "redis_connected": client is not None,
        "redis_url": REDIS_URL,
        "last_reset": _cache_stats["last_reset"]
    }
    
    # Get Redis memory info if connected
    if client is not None:
        try:
            info = await client.info("memory")
            stats["redis_memory_used"] = info.get("used_memory_human", "N/A")
            stats["redis_memory_peak"] = info.get("used_memory_peak_human", "N/A")
            
            # Get key count for all cache namespaces
            total_keys = 0
            patterns = ["neo_feed:*", "neo_lookup:*", "neo_stats:*", "neo_browse:*", "nasa:*"]
            for pattern in patterns:
                keys = await client.keys(pattern)
                total_keys += len(keys)
            stats["cached_keys_count"] = total_keys
        except Exception as e:
            logger.warning(f"Could not get Redis info: {e}")
    
    return stats


async def clear_cache(pattern: str = "nasa:*") -> dict:
    """
    Clear cached entries matching a pattern.
    
    Args:
        pattern: Redis key pattern to match (default: all NASA cache keys)
    
    Returns:
        Dictionary with deletion count and status
    """
    global _cache_stats
    
    client = await get_redis_client()
    
    if client is None:
        return {
            "status": "error",
            "message": "Redis not connected",
            "deleted_count": 0
        }
    
    try:
        keys = await client.keys(pattern)
        deleted_count = 0
        
        if keys:
            deleted_count = await client.delete(*keys)
        
        # Reset stats
        _cache_stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "last_reset": datetime.utcnow().isoformat()
        }
        
        logger.info(f"🗑️ Cache cleared | pattern={pattern} | deleted={deleted_count} keys")
        
        return {
            "status": "success",
            "message": f"Cleared {deleted_count} cached entries",
            "deleted_count": deleted_count,
            "pattern": pattern
        }
        
    except redis.RedisError as e:
        logger.error(f"❌ Cache clear error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "deleted_count": 0
        }


async def invalidate_cache_key(key: str) -> bool:
    """
    Invalidate a specific cache key.
    
    Useful for manual cache invalidation when data is known to have changed.
    
    Args:
        key: The exact cache key to invalidate
    
    Returns:
        True if key was deleted, False otherwise
    """
    client = await get_redis_client()
    
    if client is None:
        return False
    
    try:
        result = await client.delete(key)
        if result:
            logger.info(f"🗑️ Invalidated cache key: {key}")
        return result > 0
    except redis.RedisError as e:
        logger.error(f"❌ Cache invalidation error: {e}")
        return False


async def get_cached_keys_info(pattern: str = "nasa:*") -> list:
    """
    Get information about cached keys for debugging.
    
    Args:
        pattern: Redis key pattern to match
    
    Returns:
        List of dicts with key name and TTL
    """
    client = await get_redis_client()
    
    if client is None:
        return []
    
    try:
        keys = await client.keys(pattern)
        result = []
        
        for key in keys[:50]:  # Limit to first 50 keys
            ttl = await client.ttl(key)
            result.append({
                "key": key,
                "ttl_seconds": ttl,
                "ttl_hours": round(ttl / 3600, 2) if ttl > 0 else 0
            })
        
        return sorted(result, key=lambda x: x["ttl_seconds"], reverse=True)
        
    except redis.RedisError as e:
        logger.error(f"❌ Get keys error: {e}")
        return []
