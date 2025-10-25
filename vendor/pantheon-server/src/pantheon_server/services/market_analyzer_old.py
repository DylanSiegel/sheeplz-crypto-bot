"""
Pantheon Market Analyzer

This module orchestrates the pantheon-legends engines with real market data
to provide comprehensive cryptocurrency analysis and alerts.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from legends import (
    Pantheon,
    LegendRequest,
    LegendType,
    DowLegendEngine,
    WyckoffLegendEngine,
    VolumeBreakoutScanner
)

from .coinbase_service import CoinbaseService


class PantheonMarketAnalyzer:
    """
    Main analyzer that combines real market data with pantheon-legends engines
    
    This class orchestrates multiple analysis engines to provide comprehensive
    market insights for cryptocurrency trading pairs.
    """
    
    def __init__(self, coinbase_service: Optional[CoinbaseService] = None):
        """
        Initialize the market analyzer
        
        Args:
            coinbase_service: Optional pre-configured Coinbase service instance
        """
        self.coinbase = coinbase_service or CoinbaseService()
        
        # Initialize Pantheon with default engines
        self.pantheon = Pantheon.create_default()
        
        # Get available engines
        self.available_engines = self.pantheon.available_engines
        
        # Standard timeframes for multi-timeframe analysis
        self.timeframes = {
            "1m": "60",     # 1 minute
            "5m": "300",    # 5 minutes  
            "15m": "900",   # 15 minutes
            "1h": "3600",   # 1 hour
            "4h": "14400",  # 4 hours
            "1d": "86400"   # 1 day
        }
    
    async def analyze_crypto_pair(
        self,
        product_id: str,
        engine_type: EngineType = EngineType.TRADITIONAL,
        timeframes: Optional[List[str]] = None,
        max_candles: int = 300
    ) -> Dict:
        """
        Analyze a cryptocurrency pair using the specified engine
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            engine_type: Which analysis engine to use
            timeframes: List of timeframes to analyze (default: 1m, 5m, 15m)
            max_candles: Maximum candles per timeframe
            
        Returns:
            Dictionary containing analysis results for each timeframe
        """
        if timeframes is None:
            timeframes = ["1m", "5m", "15m"]
        
        logger.info(f"Starting {engine_type.value} analysis for {product_id}")
        
        try:
            # Get multi-timeframe data from Coinbase
            timeframe_seconds = [self.timeframes[tf] for tf in timeframes]
            market_data = await self.coinbase.get_multi_timeframe_data(
                product_id=product_id,
                timeframes=timeframe_seconds,
                max_candles=max_candles
            )
            
            # Select the appropriate engine
            engine = self.traditional_engine if engine_type == EngineType.TRADITIONAL else self.scanner_engine
            
            analysis_results = {}
            
            for timeframe in timeframes:
                tf_seconds = self.timeframes[timeframe]
                df = market_data.get(tf_seconds)
                
                if df is None or df.empty:
                    logger.warning(f"No data available for {product_id} on {timeframe} timeframe")
                    analysis_results[timeframe] = {
                        "error": "No data available",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    continue
                
                # Run the analysis
                try:
                    result = engine.analyze(df)
                    analysis_results[timeframe] = {
                        "engine_type": engine_type.value,
                        "timeframe": timeframe,
                        "product_id": product_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "candles_analyzed": len(df),
                        "latest_price": float(df['close'].iloc[-1]),
                        "analysis": result,
                        "data_range": {
                            "start": df.index[0].isoformat(),
                            "end": df.index[-1].isoformat()
                        }
                    }
                    
                    logger.debug(f"Completed {engine_type.value} analysis for {product_id} {timeframe}")
                    
                except Exception as e:
                    logger.error(f"Analysis failed for {product_id} {timeframe}: {e}")
                    analysis_results[timeframe] = {
                        "error": str(e),
                        "engine_type": engine_type.value,
                        "timeframe": timeframe,
                        "product_id": product_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze {product_id}: {e}")
            raise
    
    async def scan_multiple_pairs(
        self,
        product_ids: List[str],
        engine_type: EngineType = EngineType.SCANNER,
        timeframe: str = "5m",
        max_candles: int = 100
    ) -> Dict[str, Dict]:
        """
        Scan multiple cryptocurrency pairs for opportunities
        
        Args:
            product_ids: List of trading pair IDs to scan
            engine_type: Which analysis engine to use
            timeframe: Single timeframe to analyze
            max_candles: Maximum candles per pair
            
        Returns:
            Dictionary mapping product_id to analysis results
        """
        logger.info(f"Scanning {len(product_ids)} pairs using {engine_type.value} engine on {timeframe}")
        
        tasks = []
        for product_id in product_ids:
            task = self.analyze_crypto_pair(
                product_id=product_id,
                engine_type=engine_type,
                timeframes=[timeframe],
                max_candles=max_candles
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scan_results = {}
        for product_id, result in zip(product_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Scan failed for {product_id}: {result}")
                scan_results[product_id] = {
                    "error": str(result),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Extract the single timeframe result
                scan_results[product_id] = result.get(timeframe, {})
        
        return scan_results
    
    async def get_ema9_fakeout_signals(
        self,
        product_id: str,
        max_candles: int = 200
    ) -> Dict:
        """
        Specialized EMA(9) fakeout detection across multiple timeframes
        
        This method implements the EMA(9) scalper strategy discussed,
        looking for fakeouts across 1m, 5m, and 15m timeframes.
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            max_candles: Maximum candles per timeframe
            
        Returns:
            Dictionary with fakeout analysis results
        """
        logger.info(f"Running EMA(9) fakeout analysis for {product_id}")
        
        # Get data for scalping timeframes
        timeframes = ["1m", "5m", "15m"]
        analysis_results = await self.analyze_crypto_pair(
            product_id=product_id,
            engine_type=EngineType.SCANNER,  # Scanner engine for fakeout detection
            timeframes=timeframes,
            max_candles=max_candles
        )
        
        # Extract EMA and fakeout signals
        fakeout_signals = {
            "product_id": product_id,
            "timestamp": datetime.utcnow().isoformat(),
            "timeframes": {},
            "consensus": "neutral"
        }
        
        bullish_count = 0
        bearish_count = 0
        
        for timeframe in timeframes:
            tf_result = analysis_results.get(timeframe, {})
            analysis = tf_result.get("analysis", {})
            
            # Look for EMA and momentum indicators
            ema_signal = "neutral"
            if "ema" in analysis or "moving_average" in analysis:
                # Extract EMA-based signals from the analysis
                # This would depend on the specific output format of the scanner engine
                ema_signal = analysis.get("trend", "neutral")
            
            fakeout_signals["timeframes"][timeframe] = {
                "signal": ema_signal,
                "price": tf_result.get("latest_price"),
                "analysis": analysis
            }
            
            if ema_signal == "bullish":
                bullish_count += 1
            elif ema_signal == "bearish":
                bearish_count += 1
        
        # Determine consensus across timeframes
        if bullish_count >= 2:
            fakeout_signals["consensus"] = "bullish"
        elif bearish_count >= 2:
            fakeout_signals["consensus"] = "bearish"
        
        return fakeout_signals
    
    async def get_market_overview(
        self,
        popular_pairs_only: bool = True,
        engine_type: EngineType = EngineType.TRADITIONAL
    ) -> Dict:
        """
        Get a broad market overview across multiple cryptocurrency pairs
        
        Args:
            popular_pairs_only: Whether to only analyze popular trading pairs
            engine_type: Which analysis engine to use
            
        Returns:
            Dictionary with market overview and top opportunities
        """
        if popular_pairs_only:
            pairs = self.coinbase.get_popular_crypto_pairs()
        else:
            # Get all available pairs (would need to implement)
            pairs = self.coinbase.get_popular_crypto_pairs()
        
        logger.info(f"Generating market overview for {len(pairs)} pairs")
        
        # Quick scan on 5m timeframe
        scan_results = await self.scan_multiple_pairs(
            product_ids=pairs,
            engine_type=engine_type,
            timeframe="5m",
            max_candles=50  # Lighter load for overview
        )
        
        # Process and rank results
        overview = {
            "timestamp": datetime.utcnow().isoformat(),
            "engine_type": engine_type.value,
            "pairs_analyzed": len(pairs),
            "successful_analyses": 0,
            "failed_analyses": 0,
            "top_opportunities": [],
            "market_sentiment": "neutral",
            "results": scan_results
        }
        
        opportunities = []
        
        for product_id, result in scan_results.items():
            if "error" in result:
                overview["failed_analyses"] += 1
                continue
                
            overview["successful_analyses"] += 1
            
            # Extract opportunity score (this would depend on the engine output)
            analysis = result.get("analysis", {})
            score = analysis.get("score", 0)  # Assuming engines provide a score
            
            if score > 0.5:  # Threshold for interesting opportunities
                opportunities.append({
                    "product_id": product_id,
                    "score": score,
                    "price": result.get("latest_price"),
                    "analysis": analysis
                })
        
        # Sort by score and take top 10
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        overview["top_opportunities"] = opportunities[:10]
        
        return overview
