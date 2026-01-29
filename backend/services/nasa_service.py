"""
NASA NeoWs API Service with Redis Caching

This module provides a clean service layer for interacting with the NASA
Near-Earth Object Web Service (NeoWs) API. All methods are decorated with
the caching layer to minimize external API calls and respect rate limits.

TTL Strategy:
- Feed data (7-day asteroid list): 24h cache - data changes daily but not hourly
- Asteroid details (physical properties): 7 days - mass/diameter rarely change
- Statistics/counts: 1 hour - may need more frequent updates
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

import httpx

from core.cache import cache_response

# Configure logging
logger = logging.getLogger("nasa_service")

# Configuration from environment
NASA_API_KEY = os.getenv("NASA_API_KEY")
# Use a base URL without the endpoint - the .env may contain the full feed URL
_nasa_url = os.getenv("NASA_API_URL", "https://api.nasa.gov/neo/rest/v1/feed")
# Extract base URL by removing /feed if present
NASA_BASE_URL = _nasa_url.replace("/feed", "") if _nasa_url.endswith("/feed") else _nasa_url

# Validate API key on module load
if not NASA_API_KEY:
    logger.warning("⚠ NASA_API_KEY not set! API calls will fail.")


class NasaService:
    """
    Service class for NASA NeoWs API interactions.
    
    All methods are static and async for use with FastAPI.
    Caching is applied via decorators - no manual cache management needed.
    """
    
    @staticmethod
    @cache_response(ttl_seconds=86400, key_prefix="neo_feed")  # 24 hours
    async def fetch_asteroid_feed(start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Fetch asteroid feed for a date range from NASA NeoWs API.
        
        NASA API limits: Maximum 7-day range per request.
        Cached for 24 hours because:
        - Historical data doesn't change
        - Future predictions are stable on a daily basis
        - NASA has hourly rate limits we need to respect
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (max 7 days from start)
        
        Returns:
            Raw JSON response from NASA API containing:
            - element_count: Total NEOs in response
            - near_earth_objects: Dict keyed by date with list of NEO objects
        
        Raises:
            httpx.HTTPStatusError: On API error responses
            httpx.RequestError: On network failures
        """
        logger.info(f"🚀 Fetching NASA feed: {start_date} to {end_date}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{NASA_BASE_URL}/feed",
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "api_key": NASA_API_KEY
                }
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(
                f"✓ NASA feed received: {data.get('element_count', 0)} NEOs | "
                f"range: {start_date} to {end_date}"
            )
            return data
    
    @staticmethod
    @cache_response(ttl_seconds=604800, key_prefix="neo_lookup")  # 7 days
    async def fetch_asteroid_details(asteroid_id: str) -> Dict[str, Any]:
        """
        Fetch detailed information for a specific asteroid.
        
        Cached for 7 days because physical properties (mass, diameter,
        orbital parameters) rarely change and lookups are expensive.
        
        Args:
            asteroid_id: NASA SPK-ID or NEO reference ID
        
        Returns:
            Detailed asteroid object containing:
            - name, id, absolute_magnitude_h
            - estimated_diameter (multiple units)
            - orbital_data (semi-major axis, eccentricity, etc.)
            - close_approach_data (historical and predicted)
            - is_potentially_hazardous_asteroid
        
        Raises:
            httpx.HTTPStatusError: On API error responses (404 if ID not found)
            httpx.RequestError: On network failures
        """
        logger.info(f"🔍 Fetching asteroid details: {asteroid_id}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{NASA_BASE_URL}/neo/{asteroid_id}",
                params={"api_key": NASA_API_KEY}
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✓ Asteroid details received: {data.get('name', 'Unknown')}")
            return data
    
    @staticmethod
    @cache_response(ttl_seconds=3600, key_prefix="neo_stats")  # 1 hour
    async def fetch_neo_statistics() -> Dict[str, Any]:
        """
        Fetch general NEO database statistics from NASA.
        
        Cached for 1 hour - statistics may update more frequently.
        
        Returns:
            Statistics object containing counts and metadata
        """
        logger.info("📊 Fetching NEO statistics")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{NASA_BASE_URL}/stats",
                params={"api_key": NASA_API_KEY}
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✓ NEO statistics received")
            return data
    
    @staticmethod
    @cache_response(ttl_seconds=86400, key_prefix="neo_browse")  # 24 hours
    async def browse_asteroids(page: int = 0, size: int = 20) -> Dict[str, Any]:
        """
        Browse the overall asteroid dataset with pagination.
        
        Cached for 24 hours - the browse endpoint returns stable data.
        
        Args:
            page: Page number (0-indexed)
            size: Items per page (max 20)
        
        Returns:
            Paginated response containing:
            - near_earth_objects: List of NEO objects
            - page: Pagination metadata
        """
        logger.info(f"📖 Browsing asteroids: page={page}, size={size}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{NASA_BASE_URL}/neo/browse",
                params={
                    "page": page,
                    "size": min(size, 20),  # API max is 20
                    "api_key": NASA_API_KEY
                }
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✓ Browse results: {len(data.get('near_earth_objects', []))} NEOs")
            return data
    
    @staticmethod
    async def fetch_asteroid_feed_no_cache(start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Fetch asteroid feed WITHOUT caching.
        
        Use this method when you need guaranteed fresh data,
        such as for real-time monitoring or cache warming.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Raw JSON response from NASA API
        """
        logger.info(f"🚀 [NO CACHE] Fetching NASA feed: {start_date} to {end_date}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{NASA_BASE_URL}/feed",
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "api_key": NASA_API_KEY
                }
            )
            response.raise_for_status()
            return response.json()
    
    @staticmethod
    def normalize_neo_data(raw_neo: Dict[str, Any], approach_date: str) -> Dict[str, Any]:
        """
        Normalize raw NASA NEO data into a clean, consistent format.
        
        This is useful for preprocessing data before caching or storing
        in the vector database, ensuring consistent field names and types.
        
        Args:
            raw_neo: Raw NEO object from NASA API
            approach_date: The close approach date for this record
        
        Returns:
            Normalized dictionary with consistent field names
        """
        # Get close approach data for the specific date
        close_approach = None
        for approach in raw_neo.get('close_approach_data', []):
            if approach.get('close_approach_date') == approach_date:
                close_approach = approach
                break
        
        if close_approach is None and raw_neo.get('close_approach_data'):
            close_approach = raw_neo['close_approach_data'][0]
        
        return {
            "neo_id": int(raw_neo.get('id', 0)),
            "name": raw_neo.get('name', 'Unknown'),
            "nasa_jpl_url": raw_neo.get('nasa_jpl_url', ''),
            "absolute_magnitude": float(raw_neo.get('absolute_magnitude_h', 0)),
            "is_potentially_hazardous": raw_neo.get('is_potentially_hazardous_asteroid', False),
            "estimated_diameter_km_min": float(
                raw_neo.get('estimated_diameter', {})
                .get('kilometers', {})
                .get('estimated_diameter_min', 0)
            ),
            "estimated_diameter_km_max": float(
                raw_neo.get('estimated_diameter', {})
                .get('kilometers', {})
                .get('estimated_diameter_max', 0)
            ),
            "close_approach_date": approach_date,
            "relative_velocity_kph": float(
                close_approach.get('relative_velocity', {})
                .get('kilometers_per_hour', 0)
            ) if close_approach else 0,
            "miss_distance_km": float(
                close_approach.get('miss_distance', {})
                .get('kilometers', 0)
            ) if close_approach else 0,
            "miss_distance_lunar": float(
                close_approach.get('miss_distance', {})
                .get('lunar', 0)
            ) if close_approach else 0,
            "orbiting_body": close_approach.get('orbiting_body', 'Earth') if close_approach else 'Earth'
        }
    
    @staticmethod
    async def get_api_status() -> Dict[str, Any]:
        """
        Check NASA API availability and rate limit status.
        
        Makes a minimal request to verify the API is reachable.
        Not cached - always makes a fresh request.
        
        Returns:
            Status dict with availability and response time
        """
        import time
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{NASA_BASE_URL}/stats",
                    params={"api_key": NASA_API_KEY}
                )
                
                elapsed = time.time() - start_time
                
                return {
                    "status": "available" if response.status_code == 200 else "degraded",
                    "status_code": response.status_code,
                    "response_time_ms": round(elapsed * 1000, 2),
                    "api_key_valid": response.status_code != 403,
                    "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining", "unknown"),
                    "rate_limit_limit": response.headers.get("X-RateLimit-Limit", "unknown"),
                    "checked_at": datetime.utcnow().isoformat()
                }
                
        except httpx.TimeoutException:
            return {
                "status": "timeout",
                "response_time_ms": 10000,
                "api_key_valid": None,
                "checked_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
