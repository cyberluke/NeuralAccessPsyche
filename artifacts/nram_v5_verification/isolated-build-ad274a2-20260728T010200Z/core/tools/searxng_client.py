"""SearXNG client for market research and company analysis."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


class SearXNGClient:
    """Client for SearXNG search engine at https://ds.nanotrik.ai"""
    
    def __init__(self, base_url: str = "https://ds.nanotrik.ai"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"SearXNG client initialized: {base_url}")
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def search(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: str = "auto",
        time_range: Optional[str] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Perform a search query.
        
        Args:
            query: Search query string
            categories: List of categories (general, news, images, videos, etc.)
            engines: List of specific engines to use
            language: Language code (auto, en, cs, etc.)
            time_range: Time filter (day, week, month, year)
            max_results: Maximum number of results to return
        
        Returns:
            Dict with search results
        """
        params = {
            "q": query,
            "format": "json",
            "language": language,
        }
        
        if categories:
            params["categories"] = ",".join(categories)
        
        if engines:
            params["engines"] = ",".join(engines)
        
        if time_range:
            params["time_range"] = time_range
        
        try:
            response = await self._client.get(f"{self.base_url}/search", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Limit results
            results = data.get("results", [])[:max_results]
            
            return {
                "query": query,
                "number_of_results": data.get("number_of_results", len(results)),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "engine": r.get("engine", ""),
                        "score": r.get("score", 0),
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"SearXNG search failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "results": [],
            }
    
    async def market_research(self, topic: str, max_results: int = 15) -> Dict[str, Any]:
        """
        Perform market research search.
        
        Args:
            topic: Market or industry topic
            max_results: Maximum results
        
        Returns:
            Dict with market research results
        """
        query = f"{topic} market analysis trends 2024 2025"
        return await self.search(
            query=query,
            categories=["general", "news"],
            language="en",
            max_results=max_results,
        )
    
    async def company_analysis(self, company_name: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Perform company analysis search.
        
        Args:
            company_name: Name of the company
            max_results: Maximum results
        
        Returns:
            Dict with company analysis results
        """
        query = f"{company_name} company profile products strategy"
        return await self.search(
            query=query,
            categories=["general"],
            language="en",
            max_results=max_results,
        )
    
    async def technology_trends(self, technology: str, max_results: int = 12) -> Dict[str, Any]:
        """
        Search for technology trends and developments.
        
        Args:
            technology: Technology or field
            max_results: Maximum results
        
        Returns:
            Dict with technology trend results
        """
        query = f"{technology} technology trends innovations 2024 2025"
        return await self.search(
            query=query,
            categories=["general", "news"],
            language="en",
            time_range="year",
            max_results=max_results,
        )


# Global client instance (initialized on demand)
_searxng_client: Optional[SearXNGClient] = None


def get_searxng_client() -> SearXNGClient:
    """Get or create the global SearXNG client."""
    global _searxng_client
    if _searxng_client is None:
        _searxng_client = SearXNGClient()
    return _searxng_client
