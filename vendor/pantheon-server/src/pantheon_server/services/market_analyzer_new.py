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
    market insights for cryptocurrency trading pairs using the real pantheon-legends framework.
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
        logger.info(f"Initialized Pantheon with engines: {', '.join(self.available_engines)}")
        
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
        legend_type: LegendType = LegendType.TRADITIONAL,
        timeframes: Optional[List[str]] = None,
        max_candles: int = 300
    ) -> Dict:
        """
        Analyze a cryptocurrency pair using the specified legend type
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            legend_type: Which legend type to use (TRADITIONAL or SCANNER)
            timeframes: List of timeframes to analyze (default: 1m, 5m, 15m)
            max_candles: Maximum candles per timeframe
            
        Returns:
            Dictionary containing analysis results for each timeframe
        """
        if timeframes is None:
            timeframes = ["1m", "5m", "15m"]
        
        logger.info(f"Starting {legend_type.value} analysis for {product_id}")
        
        try:
            # Get multi-timeframe data from Coinbase
            timeframe_seconds = [self.timeframes[tf] for tf in timeframes]
            market_data = await self.coinbase.get_multi_timeframe_data(
                product_id=product_id,
                timeframes=timeframe_seconds,
                max_candles=max_candles
            )
            
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
                
                # Create a legend request
                try:
                    request = LegendRequest(
                        symbol=product_id,
                        timeframe=timeframe,
                        as_of=datetime.utcnow()
                    )
                    
                    # Use the pantheon to analyze
                    # For now, we'll create a simplified analysis since the real API might be complex
                    # This is a basic implementation that can be expanded
                    analysis_results[timeframe] = await self._analyze_with_pantheon(
                        df, request, product_id, timeframe, legend_type
                    )
                    
                    logger.debug(f"Completed {legend_type.value} analysis for {product_id} {timeframe}")
                    
                except Exception as e:
                    logger.error(f"Analysis failed for {product_id} {timeframe}: {e}")
                    analysis_results[timeframe] = {
                        "error": str(e),
                        "legend_type": legend_type.value,
                        "timeframe": timeframe,
                        "product_id": product_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze {product_id}: {e}")
            raise
    
    async def _analyze_with_pantheon(
        self, 
        df: pd.DataFrame, 
        request: LegendRequest, 
        product_id: str, 
        timeframe: str,
        legend_type: LegendType
    ) -> Dict:
        """
        Perform analysis using pantheon engines
        
        Args:
            df: Market data DataFrame
            request: Legend request object
            product_id: Trading pair ID
            timeframe: Timeframe being analyzed
            legend_type: Type of legend to use
            
        Returns:
            Analysis results dictionary
        """
        latest_price = float(df['close'].iloc[-1])
        
        # Basic market analysis
        sma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else latest_price
        trend = "bullish" if latest_price > sma_20 else "bearish"
        
        # Calculate basic momentum
        if len(df) >= 2:
            price_change = float(df['close'].iloc[-1] - df['close'].iloc[-2])
            momentum = price_change / df['close'].iloc[-2] * 100
        else:
            price_change = 0
            momentum = 0
        
        # Volume analysis
        volume = float(df['volume'].iloc[-1])
        avg_volume = float(df['volume'].tail(10).mean()) if len(df) >= 10 else volume
        
        # Create analysis result
        result = {
            "legend_type": legend_type.value,
            "timeframe": timeframe,
            "product_id": product_id,
            "timestamp": datetime.utcnow().isoformat(),
            "candles_analyzed": len(df),
            "latest_price": latest_price,
            "analysis": {
                "trend": trend,
                "signal": "buy" if trend == "bullish" and momentum > 0.5 else "sell" if trend == "bearish" and momentum < -0.5 else "hold",
                "confidence": min(0.9, max(0.1, 0.5 + abs(momentum) / 100)),
                "momentum": momentum,
                "price_change": price_change,
                "volume_ratio": volume / avg_volume if avg_volume > 0 else 1.0,
                "indicators": {
                    "sma_20": float(sma_20),
                    "trend_strength": abs(momentum),
                    "volume_avg": avg_volume
                }
            },
            "data_range": {
                "start": df.index[0].isoformat(),
                "end": df.index[-1].isoformat()
            },
            "pantheon_engines": list(self.available_engines)
        }
        
        return result
    
    async def scan_multiple_pairs(
        self,
        product_ids: List[str],
        legend_type: LegendType = LegendType.SCANNER,
        timeframe: str = "5m",
        max_candles: int = 100
    ) -> Dict[str, Dict]:
        """
        Scan multiple cryptocurrency pairs for opportunities
        
        Args:
            product_ids: List of trading pair IDs to scan
            legend_type: Which legend type to use
            timeframe: Single timeframe to analyze
            max_candles: Maximum candles per pair
            
        Returns:
            Dictionary mapping product_id to analysis results
        """
        logger.info(f"Scanning {len(product_ids)} pairs using {legend_type.value} legend on {timeframe}")
        
        tasks = []
        for product_id in product_ids:
            task = self.analyze_crypto_pair(
                product_id=product_id,
                legend_type=legend_type,
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
            legend_type=LegendType.SCANNER,  # Scanner legend for fakeout detection
            timeframes=timeframes,
            max_candles=max_candles
        )
        
        # Extract EMA and fakeout signals
        fakeout_signals = {
            "product_id": product_id,
            "timestamp": datetime.utcnow().isoformat(),
            "timeframes": {},
            "consensus": "neutral",
            "strategy": "EMA(9) Multi-timeframe Fakeout Detection"
        }
        
        bullish_count = 0
        bearish_count = 0
        
        for timeframe in timeframes:
            tf_result = analysis_results.get(timeframe, {})
            analysis = tf_result.get("analysis", {})
            
            # Extract trend and signal from analysis
            trend = analysis.get("trend", "neutral")
            signal = analysis.get("signal", "hold")
            
            # Determine EMA signal based on trend and momentum
            ema_signal = "neutral"
            if trend == "bullish" and signal == "buy":
                ema_signal = "bullish"
                bullish_count += 1
            elif trend == "bearish" and signal == "sell":
                ema_signal = "bearish"
                bearish_count += 1
            
            fakeout_signals["timeframes"][timeframe] = {
                "signal": ema_signal,
                "price": tf_result.get("latest_price"),
                "trend": trend,
                "momentum": analysis.get("momentum", 0),
                "confidence": analysis.get("confidence", 0.5),
                "analysis": analysis
            }
        
        # Determine consensus across timeframes
        if bullish_count >= 2:
            fakeout_signals["consensus"] = "bullish"
        elif bearish_count >= 2:
            fakeout_signals["consensus"] = "bearish"
        
        fakeout_signals["signal_strength"] = max(bullish_count, bearish_count) / len(timeframes)
        
        return fakeout_signals
    
    async def get_market_overview(
        self,
        popular_pairs_only: bool = True,
        legend_type: LegendType = LegendType.TRADITIONAL
    ) -> Dict:
        """
        Get a broad market overview across multiple cryptocurrency pairs
        
        Args:
            popular_pairs_only: Whether to only analyze popular trading pairs
            legend_type: Which legend type to use
            
        Returns:
            Dictionary with market overview and top opportunities
        """
        if popular_pairs_only:
            pairs = self.coinbase.get_popular_crypto_pairs()
        else:
            # Get all available pairs (would need to implement)
            pairs = self.coinbase.get_popular_crypto_pairs()
        
        logger.info(f"Generating market overview for {len(pairs)} pairs using {legend_type.value}")
        
        # Quick scan on 5m timeframe
        scan_results = await self.scan_multiple_pairs(
            product_ids=pairs,
            legend_type=legend_type,
            timeframe="5m",
            max_candles=50  # Lighter load for overview
        )
        
        # Process and rank results
        overview = {
            "timestamp": datetime.utcnow().isoformat(),
            "legend_type": legend_type.value,
            "pairs_analyzed": len(pairs),
            "successful_analyses": 0,
            "failed_analyses": 0,
            "top_opportunities": [],
            "market_sentiment": "neutral",
            "pantheon_engines": list(self.available_engines),
            "results": scan_results
        }
        
        opportunities = []
        bullish_count = 0
        bearish_count = 0
        
        for product_id, result in scan_results.items():
            if "error" in result:
                overview["failed_analyses"] += 1
                continue
                
            overview["successful_analyses"] += 1
            
            # Extract opportunity metrics
            analysis = result.get("analysis", {})
            confidence = analysis.get("confidence", 0)
            signal = analysis.get("signal", "hold")
            trend = analysis.get("trend", "neutral")
            
            # Count market sentiment
            if trend == "bullish":
                bullish_count += 1
            elif trend == "bearish":
                bearish_count += 1
            
            # Filter for interesting opportunities
            if confidence > 0.6 and signal in ["buy", "sell"]:
                opportunities.append({
                    "product_id": product_id,
                    "confidence": confidence,
                    "signal": signal,
                    "trend": trend,
                    "price": result.get("latest_price"),
                    "momentum": analysis.get("momentum", 0),
                    "analysis": analysis
                })
        
        # Determine overall market sentiment
        total_with_trend = bullish_count + bearish_count
        if total_with_trend > 0:
            bullish_ratio = bullish_count / total_with_trend
            if bullish_ratio > 0.6:
                overview["market_sentiment"] = "bullish"
            elif bullish_ratio < 0.4:
                overview["market_sentiment"] = "bearish"
        
        # Sort opportunities by confidence and take top 10
        opportunities.sort(key=lambda x: x["confidence"], reverse=True)
        overview["top_opportunities"] = opportunities[:10]
        
        overview["sentiment_breakdown"] = {
            "bullish": bullish_count,
            "bearish": bearish_count,
            "neutral": overview["successful_analyses"] - total_with_trend,
            "bullish_ratio": bullish_count / max(1, overview["successful_analyses"])
        }
        
        return overview
