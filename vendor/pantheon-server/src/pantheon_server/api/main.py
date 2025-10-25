"""
Pantheon Server API

This module provides the main FastAPI application for the Pantheon Server,
exposing REST endpoints for cryptocurrency analysis and market data.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from legends import LegendType, Pantheon
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from ..services import CoinbaseService, PantheonMarketAnalyzer, RedisService

# Initialize services
coinbase_service = CoinbaseService()
market_analyzer = PantheonMarketAnalyzer(coinbase_service)
redis_service = RedisService()

app = FastAPI(
    title="Pantheon Server",
    description="Cryptocurrency analysis server using Pantheon Legends framework",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    product_id: str
    legend_type: LegendType = LegendType.TRADITIONAL
    timeframes: Optional[List[str]] = ["5m", "15m", "1h"]
    max_candles: int = 200
    force_refresh: bool = False  # New field for cache bypass


class ScanRequest(BaseModel):
    product_ids: List[str]
    legend_type: LegendType = LegendType.SCANNER
    timeframe: str = "5m"
    max_candles: int = 100
    force_refresh: bool = False  # New field for cache bypass


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning basic server information"""
    try:
        return {
            "service": "Pantheon Server",
            "version": "0.1.0",
            "description": "Cryptocurrency analysis using Pantheon Legends",
            "timestamp": datetime.utcnow().isoformat(),
            "docs": "/docs",
            "endpoints": {
                "health": "/health",
                "engines": "/engines",
                "products": "/products",
                "analyze": "/analyze",
                "scan": "/scan",
                "ema9": "/ema9/{product_id}",
                "overview": "/overview"
            }
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for monitoring including Redis status"""
    try:
        redis_health = redis_service.health_check()
        redis_status = redis_health.get("status", "unknown")
    except Exception:
        redis_status = "unavailable"
    
    return {
        "status": "healthy" if redis_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "pantheon-server",
        "pantheon_legends": "connected",
        "coinbase_api": "available",
        "redis_cache": redis_status
    }


@app.get("/test")
async def test_endpoint() -> Dict[str, str]:
    """Simple test endpoint for debugging"""
    return {"message": "Server is working!", "status": "ok"}


@app.get("/engines")
async def list_engines() -> Dict[str, List[str]]:
    """List available analysis engines"""
    try:
        pantheon = Pantheon.create_default()
        engines = pantheon.available_engines
        return {
            "available_engines": list(engines),
            "timestamp": datetime.utcnow().isoformat(),
            "descriptions": {
                "traditional": "Classic technical analysis with traditional indicators",
                "scanner": "Advanced scanning engine for pattern detection"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving engines: {str(e)}")


@app.get("/products")
async def get_products() -> Dict:
    """Get available cryptocurrency trading pairs"""
    try:
        products = await coinbase_service.get_products()
        popular_pairs = coinbase_service.get_popular_crypto_pairs()
        
        return {
            "total_products": len(products),
            "popular_pairs": popular_pairs,
            "all_products": [p.get("id") for p in products if p.get("id")],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching products: {str(e)}")


@app.post("/analyze")
async def analyze_crypto(request: AnalysisRequest) -> Dict:
    """Analyze a cryptocurrency pair using specified engine and timeframes with Redis caching"""
    try:
        cache_status = "miss"
        cache_age_seconds = 0
        results = {}
        
        # Check if force refresh is requested
        if not request.force_refresh:
            # Try to get cached results for each timeframe
            for timeframe in request.timeframes:
                cached_result = redis_service.get_cached_analysis(
                    product_id=request.product_id,
                    timeframe=timeframe,
                    legend_type=request.legend_type.value
                )
                
                if cached_result:
                    # Calculate cache age
                    cache_time = datetime.fromisoformat(cached_result['cached_at'])
                    cache_age = datetime.utcnow() - cache_time
                    cache_age_seconds = cache_age.total_seconds()
                    
                    # Remove cache metadata from result
                    clean_result = {k: v for k, v in cached_result.items() 
                                  if k not in ['cached_at', 'cache_key']}
                    results[timeframe] = clean_result
                    cache_status = "hit"
        
        # If we have cached results for all timeframes, return them
        if len(results) == len(request.timeframes) and not request.force_refresh:
            return {
                "success": True,
                "request": request.dict(),
                "results": results,
                "cache_status": cache_status,
                "cache_age_seconds": int(cache_age_seconds),
                "data_freshness": "cached",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Otherwise, fetch fresh data
        fresh_results = await market_analyzer.analyze_crypto_pair(
            product_id=request.product_id,
            legend_type=request.legend_type,
            timeframes=request.timeframes,
            max_candles=request.max_candles
        )
        
        # Cache the fresh results
        for timeframe, result in fresh_results.items():
            redis_service.cache_analysis_result(
                product_id=request.product_id,
                timeframe=timeframe,
                legend_type=request.legend_type.value,
                result=result
            )
        
        cache_status = "refreshed" if request.force_refresh else "miss"
        
        return {
            "success": True,
            "request": request.dict(),
            "results": fresh_results,
            "cache_status": cache_status,
            "cache_age_seconds": 0,
            "data_freshness": "live",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/scan")
async def scan_multiple_pairs(request: ScanRequest) -> Dict:
    """Scan multiple cryptocurrency pairs for trading opportunities"""
    try:
        results = await market_analyzer.scan_multiple_pairs(
            product_ids=request.product_ids,
            engine_type=request.engine_type,
            timeframe=request.timeframe,
            max_candles=request.max_candles
        )
        
        # Count successful vs failed scans
        successful = sum(1 for r in results.values() if "error" not in r)
        failed = len(results) - successful
        
        return {
            "success": True,
            "request": request.dict(),
            "summary": {
                "total_pairs": len(request.product_ids),
                "successful_scans": successful,
                "failed_scans": failed,
                "success_rate": (successful / len(request.product_ids)) * 100
            },
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")


@app.get("/ema9/{product_id}")
async def ema9_fakeout_analysis(product_id: str, max_candles: int = 200) -> Dict:
    """Run EMA(9) fakeout analysis on a specific cryptocurrency pair"""
    try:
        signals = await market_analyzer.get_ema9_fakeout_signals(
            product_id=product_id,
            max_candles=max_candles
        )
        
        return {
            "success": True,
            "product_id": product_id,
            "strategy": "EMA(9) Multi-timeframe Fakeout Detection",
            "signals": signals,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EMA(9) analysis failed: {str(e)}")


@app.get("/overview")
async def market_overview(
    popular_only: bool = True,
    legend_type: LegendType = LegendType.TRADITIONAL,
    force_refresh: bool = False
) -> Dict:
    """Get a comprehensive market overview with Redis caching"""
    try:
        cache_key = f"overview:{popular_only}:{legend_type.value}"
        cache_status = "miss"
        cache_age_seconds = 0
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_overview = redis_service.get(f"pantheon:cache:{cache_key}")
            
            if cached_overview:
                cache_time = datetime.fromisoformat(cached_overview['cached_at'])
                cache_age = datetime.utcnow() - cache_time
                cache_age_seconds = cache_age.total_seconds()
                
                # Return cached if still fresh (10 minutes TTL)
                if cache_age_seconds < 600:  # 10 minutes
                    return {
                        "success": True,
                        "overview": cached_overview['data'],
                        "cache_status": "hit",
                        "cache_age_seconds": int(cache_age_seconds),
                        "data_freshness": "cached",
                        "timestamp": datetime.utcnow().isoformat()
                    }
        
        # Fetch fresh overview data
        overview = await market_analyzer.get_market_overview(
            popular_pairs_only=popular_only,
            legend_type=legend_type
        )
        
        # Cache the fresh overview
        cache_data = {
            "data": overview,
            "cached_at": datetime.utcnow().isoformat()
        }
        redis_service.set(f"pantheon:cache:{cache_key}", cache_data, ttl=600)  # 10 minutes
        
        cache_status = "refreshed" if force_refresh else "miss"
        
        return {
            "success": True,
            "overview": overview,
            "cache_status": cache_status,
            "cache_age_seconds": 0,
            "data_freshness": "live",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market overview failed: {str(e)}")


@app.get("/ticker/{product_id}")
async def get_ticker(product_id: str) -> Dict:
    """Get current ticker information for a cryptocurrency pair"""
    try:
        ticker = await coinbase_service.get_product_ticker(product_id)
        
        return {
            "success": True,
            "product_id": product_id,
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ticker fetch failed: {str(e)}")


@app.get("/candles/{product_id}")
async def get_candles(
    product_id: str,
    timeframe: str = "300",
    max_candles: int = 100
) -> Dict:
    """Get historical candle data for a cryptocurrency pair"""
    try:
        df = await coinbase_service.get_product_candles(
            product_id=product_id,
            timeframe=timeframe,
            max_candles=max_candles
        )
        
        # Convert DataFrame to JSON-serializable format
        candles_data = []
        for timestamp, row in df.iterrows():
            candles_data.append({
                "timestamp": timestamp.isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        
        return {
            "success": True,
            "product_id": product_id,
            "timeframe": timeframe,
            "candle_count": len(candles_data),
            "candles": candles_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Candles fetch failed: {str(e)}")


# === Cache Management Endpoints ===

@app.get("/cache/health")
async def cache_health() -> Dict:
    """Get Redis cache health and statistics"""
    try:
        health = redis_service.health_check()
        stats = redis_service.get_cache_stats()
        
        return {
            "success": True,
            "redis_health": health,
            "cache_statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.delete("/cache/analysis/{product_id}")
async def clear_analysis_cache(product_id: str) -> Dict:
    """Clear analysis cache for a specific product"""
    try:
        deleted_count = redis_service.clear_analysis_cache(product_id)
        
        return {
            "success": True,
            "message": f"Cleared analysis cache for {product_id}",
            "deleted_keys": deleted_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


@app.delete("/cache/overview")
async def clear_overview_cache() -> Dict:
    """Clear market overview cache"""
    try:
        # Clear overview cache keys
        overview_keys = [
            "pantheon:cache:overview:True:traditional",
            "pantheon:cache:overview:True:scanner", 
            "pantheon:cache:overview:False:traditional",
            "pantheon:cache:overview:False:scanner"
        ]
        
        deleted_count = 0
        for key in overview_keys:
            if redis_service.delete(key):
                deleted_count += 1
        
        return {
            "success": True,
            "message": "Cleared market overview cache",
            "deleted_keys": deleted_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


@app.delete("/cache/all")
async def clear_all_cache() -> Dict:
    """Clear all pantheon cache (use with caution)"""
    try:
        analysis_deleted = redis_service.clear_analysis_cache()
        market_deleted = redis_service.clear_market_cache()
        
        # Clear general cache
        general_keys = redis_service.redis_client.keys("pantheon:cache:*")
        general_deleted = 0
        if general_keys:
            general_deleted = redis_service.redis_client.delete(*general_keys)
        
        total_deleted = analysis_deleted + market_deleted + general_deleted
        
        return {
            "success": True,
            "message": "Cleared all pantheon cache",
            "deleted_breakdown": {
                "analysis_keys": analysis_deleted,
                "market_keys": market_deleted, 
                "general_keys": general_deleted,
                "total": total_deleted
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


@app.post("/analyze/{product_id}/refresh")
async def force_analyze_refresh(product_id: str, request: AnalysisRequest) -> Dict:
    """Force fresh analysis, bypassing cache"""
    # Override product_id and force_refresh
    request.product_id = product_id
    request.force_refresh = True
    return await analyze_crypto(request)


@app.post("/overview/refresh")
async def force_overview_refresh(
    popular_only: bool = True,
    legend_type: LegendType = LegendType.TRADITIONAL
) -> Dict:
    """Force fresh market overview, bypassing cache"""
    return await market_overview(popular_only=popular_only, legend_type=legend_type, force_refresh=True)


if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload
    )
