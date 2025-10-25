"""
Pantheon Server Streamlit UI

This module provides the main Streamlit web interface for the Pantheon Server,
including crypto analysis panels, test interfaces, and visualization components.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1
from plotly.subplots import make_subplots

from legends import LegendType
from pantheon_server.services import CoinbaseService, PantheonMarketAnalyzer


class PantheonUI:
    """Main Streamlit UI application for Pantheon Server"""
    
    def __init__(self):
        """Initialize the UI with required services"""
        if 'coinbase_service' not in st.session_state:
            st.session_state.coinbase_service = CoinbaseService()
        
        if 'market_analyzer' not in st.session_state:
            st.session_state.market_analyzer = PantheonMarketAnalyzer(
                coinbase_service=st.session_state.coinbase_service
            )
    
    def run(self):
        """Main entry point for the Streamlit app"""
        st.set_page_config(
            page_title="Pantheon Server",
            page_icon="🏛️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar navigation
        st.sidebar.title("🏛️ Pantheon Server")
        st.sidebar.markdown("*Legendary cryptocurrency analysis*")
        
        page = st.sidebar.selectbox(
            "Navigate to:",
            [
                "📊 Market Overview",
                "🔍 Crypto Analysis",
                "🧪 Test Panel", 
                "📈 EMA(9) Scalper",
                "⚙️ Settings"
            ]
        )
        
        # Route to appropriate page
        if page == "📊 Market Overview":
            self.market_overview_page()
        elif page == "🔍 Crypto Analysis":
            self.crypto_analysis_page()
        elif page == "🧪 Test Panel":
            self.test_panel_page()
        elif page == "📈 EMA(9) Scalper":
            self.ema_scalper_page()
        elif page == "⚙️ Settings":
            self.settings_page()
    
    def market_overview_page(self):
        """Market overview dashboard"""
        st.title("📊 Market Overview")
        st.markdown("Real-time cryptocurrency market analysis using Pantheon Legends")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            legend_type = st.selectbox(
                "Analysis Engine",
                options=[LegendType.TRADITIONAL, LegendType.SCANNER],
                format_func=lambda x: x.value.title()
            )
        
        with col2:
            popular_only = st.checkbox("Popular pairs only", value=True)
        
        with col3:
            auto_refresh = st.checkbox("Auto-refresh (30s)")
        
        if st.button("🔄 Generate Overview") or auto_refresh:
            with st.spinner("Analyzing market..."):
                try:
                    # Run async function
                    overview = asyncio.run(
                        st.session_state.market_analyzer.get_market_overview(
                            popular_pairs_only=popular_only,
                            legend_type=legend_type
                        )
                    )
                    
                    self._display_market_overview(overview)
                    
                    # Add timestamp
                    from datetime import datetime
                    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
                    
                except Exception as e:
                    st.error(f"Error generating overview: {e}")
        
        # Quick action buttons
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Analyze Top Pair"):
                st.info("Navigate to 'Crypto Analysis' to analyze individual pairs")
        
        with col2:
            if st.button("🧪 Run Batch Test"):
                st.info("Navigate to 'Test Panel' to run batch analysis")
        
        with col3:
            if st.button("📊 EMA Scalper"):
                st.info("Navigate to 'EMA(9) Scalper' for scalping signals")
        
        if auto_refresh:
            # Auto-refresh every 30 seconds
            import time
            time.sleep(30)
            st.experimental_rerun()
    
    def crypto_analysis_page(self):
        """Individual cryptocurrency analysis page"""
        st.title("🔍 Cryptocurrency Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Product selection options
            st.subheader("Trading Pair Selection")
            
            # Selection method
            selection_method = st.radio(
                "Choose pair selection method:",
                ["Popular Pairs", "Custom Pair"],
                horizontal=True
            )
            
            if selection_method == "Popular Pairs":
                # Original popular pairs dropdown
                popular_pairs = st.session_state.coinbase_service.get_popular_crypto_pairs()
                
                product_id = st.selectbox(
                    "Select Trading Pair",
                    options=popular_pairs,
                    index=0,
                    key="popular_pair_select"
                )
            
            else:  # Custom Pair
                # User-defined pair entry
                st.markdown("**Enter Custom Trading Pair:**")
                
                col_symbol, col_quote = st.columns([2, 1])
                
                with col_symbol:
                    symbol = st.text_input(
                        "Symbol (e.g., BTC, ETH, DOGE)",
                        value="BTC",
                        placeholder="Enter cryptocurrency symbol",
                        help="Enter the cryptocurrency symbol (base currency)"
                    ).upper().strip()
                
                with col_quote:
                    quote_currency = st.selectbox(
                        "Quote Currency",
                        options=["USD", "USDC"],
                        index=0,
                        help="Select the quote currency"
                    )
                
                # Construct the product_id
                if symbol:
                    product_id = f"{symbol}-{quote_currency}"
                    st.info(f"🎯 **Trading Pair:** {product_id}")
                    
                    # Validation message
                    if len(symbol) < 2:
                        st.warning("⚠️ Please enter a valid cryptocurrency symbol (at least 2 characters)")
                    elif not symbol.isalpha():
                        st.warning("⚠️ Symbol should contain only letters")
                    else:
                        st.success(f"✅ Ready to analyze {product_id}")
                else:
                    product_id = "BTC-USD"  # Default fallback
                    st.warning("⚠️ Please enter a cryptocurrency symbol")
            
            # Analysis engine selection
            legend_type = st.selectbox(
                "Analysis Engine",
                options=[LegendType.TRADITIONAL, LegendType.SCANNER],
                format_func=lambda x: x.value.title()
            )
        
        with col2:
            timeframes = st.multiselect(
                "Timeframes",
                options=["1m", "5m", "15m", "1h", "4h", "1d"],
                default=["5m", "15m", "1h"]
            )
            
            max_candles = st.slider(
                "Max Candles",
                min_value=50,
                max_value=500,
                value=200,
                step=50
            )
        
        if st.button("🚀 Analyze"):
            if not timeframes:
                st.warning("Please select at least one timeframe")
                return
            
            # Additional validation for custom pairs
            if selection_method == "Custom Pair":
                if len(symbol) < 2:
                    st.error("❌ Please enter a valid cryptocurrency symbol (at least 2 characters)")
                    return
                elif not symbol.isalpha():
                    st.error("❌ Symbol should contain only letters")
                    return
            
            with st.spinner(f"Analyzing {product_id}..."):
                try:
                    results = asyncio.run(
                        st.session_state.market_analyzer.analyze_crypto_pair(
                            product_id=product_id,
                            legend_type=legend_type,
                            timeframes=timeframes,
                            max_candles=max_candles
                        )
                    )
                    
                    # Store results in session state so they persist across button clicks
                    st.session_state.analysis_results = results
                    st.session_state.analysis_product_id = product_id
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "not found" in error_msg or "404" in error_msg:
                        st.error(f"❌ **Trading pair '{product_id}' not found on Coinbase**")
                        st.info("💡 **Suggestions:**")
                        st.markdown("""
                        - Check if the symbol is correct (e.g., BTC, ETH, DOGE)
                        - Try a different quote currency (USD vs USDC)
                        - Use 'Popular Pairs' to see available options
                        - Some newer or less popular tokens may not be available
                        """)
                    elif "rate limit" in error_msg or "429" in error_msg:
                        st.error("⏰ **Rate limit exceeded** - please wait a moment and try again")
                    else:
                        st.error(f"❌ **Analysis failed:** {e}")
                        st.info("💡 Try using a different trading pair or check your internet connection")
        
        # Display analysis results if they exist in session state
        if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
            self._display_analysis_results(st.session_state.analysis_product_id, st.session_state.analysis_results)
    
    def test_panel_page(self):
        """Interactive test panel for testing crypto pairs"""
        st.title("🧪 Test Panel")
        st.markdown("Interactive testing interface for cryptocurrency analysis")
        
        # Test configuration
        st.subheader("Test Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_pairs = st.text_area(
                "Test Pairs (one per line)",
                value="BTC-USD\nETH-USD\nSOL-USD",
                height=100
            )
        
        with col2:
            test_engine = st.selectbox(
                "Test Engine",
                options=[LegendType.TRADITIONAL, LegendType.SCANNER],
                format_func=lambda x: x.value.title()
            )
        
        with col3:
            test_timeframe = st.selectbox(
                "Test Timeframe",
                options=["1m", "5m", "15m", "1h"],
                index=1
            )
        
        # Batch test button
        if st.button("🧪 Run Batch Test"):
            pairs = [pair.strip() for pair in test_pairs.split('\n') if pair.strip()]
            
            if not pairs:
                st.warning("Please enter at least one trading pair")
                return
            
            with st.spinner(f"Testing {len(pairs)} pairs..."):
                try:
                    results = asyncio.run(
                        st.session_state.market_analyzer.scan_multiple_pairs(
                            product_ids=pairs,
                            legend_type=test_engine,
                            timeframe=test_timeframe,
                            max_candles=100
                        )
                    )
                    
                    self._display_batch_test_results(results, test_engine, test_timeframe)
                    
                except Exception as e:
                    st.error(f"Batch test failed: {e}")
        
        # Real-time test section
        st.subheader("Real-time Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rt_pair = st.selectbox(
                "Real-time Pair",
                options=st.session_state.coinbase_service.get_popular_crypto_pairs()
            )
        
        with col2:
            rt_interval = st.slider(
                "Update Interval (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                step=10
            )
        
        rt_placeholder = st.empty()
        
        if st.button("▶️ Start Real-time Test"):
            for i in range(10):  # Run for 10 iterations
                with rt_placeholder.container():
                    st.info(f"Real-time update {i+1}/10 for {rt_pair}")
                    
                    try:
                        result = asyncio.run(
                            st.session_state.market_analyzer.analyze_crypto_pair(
                                product_id=rt_pair,
                                legend_type=test_engine,
                                timeframes=[test_timeframe],
                                max_candles=50
                            )
                        )
                        
                        # Display compact results
                        if result and test_timeframe in result:
                            tf_result = result[test_timeframe]
                            if "error" not in tf_result:
                                st.success(f"✅ Price: ${tf_result.get('latest_price', 'N/A')}")
                                st.json(tf_result.get('analysis', {}))
                        
                    except Exception as e:
                        st.error(f"Real-time test error: {e}")
                
                import time
                time.sleep(rt_interval)
    
    def ema_scalper_page(self):
        """EMA(9) scalper strategy page"""
        st.title("📈 EMA(9) Scalper")
        st.markdown("Multi-timeframe EMA(9) fakeout detection system")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scalp_pair = st.selectbox(
                "Scalping Pair",
                options=st.session_state.coinbase_service.get_popular_crypto_pairs(),
                index=0
            )
        
        with col2:
            candle_count = st.slider(
                "Historical Candles",
                min_value=100,
                max_value=500,
                value=200,
                step=50
            )
        
        if st.button("🎯 Run EMA(9) Analysis"):
            with st.spinner(f"Running EMA(9) fakeout analysis for {scalp_pair}..."):
                try:
                    signals = asyncio.run(
                        st.session_state.market_analyzer.get_ema9_fakeout_signals(
                            product_id=scalp_pair,
                            max_candles=candle_count
                        )
                    )
                    
                    self._display_ema_signals(signals)
                    
                except Exception as e:
                    st.error(f"EMA(9) analysis failed: {e}")
        
        # Explanation section
        with st.expander("📚 About EMA(9) Scalping Strategy"):
            st.markdown("""
            **EMA(9) Multi-Timeframe Fakeout Detection**
            
            This strategy analyzes price action across multiple timeframes (1m, 5m, 15m) 
            to identify potential fakeout scenarios around the 9-period Exponential Moving Average.
            
            **Key Concepts:**
            - **Fakeouts**: False breakouts that quickly reverse direction
            - **Multi-timeframe confluence**: Signals that align across timeframes
            - **EMA(9)**: Fast-responding moving average for trend identification
            
            **Signal Interpretation:**
            - **Bullish**: Price above EMA(9) with upward momentum
            - **Bearish**: Price below EMA(9) with downward momentum  
            - **Neutral**: Consolidation or conflicting signals
            """)
    
    def settings_page(self):
        """Application settings and configuration"""
        st.title("⚙️ Settings")
        
        st.subheader("API Configuration")
        
        rate_limit = st.slider(
            "Coinbase API Rate Limit (requests/second)",
            min_value=1,
            max_value=20,
            value=10,
            help="Lower values reduce risk of rate limiting"
        )
        
        if st.button("Update Rate Limit"):
            st.session_state.coinbase_service = CoinbaseService(rate_limit_per_second=rate_limit)
            st.session_state.market_analyzer = PantheonMarketAnalyzer(
                coinbase_service=st.session_state.coinbase_service
            )
            st.success("Rate limit updated!")
        
        st.subheader("Analysis Settings")
        
        default_timeframes = st.multiselect(
            "Default Timeframes",
            options=["1m", "5m", "15m", "1h", "4h", "1d"],
            default=["5m", "15m", "1h"]
        )
        
        st.subheader("About")
        st.markdown("""
        **Pantheon Server v0.1.0**
        
        Built with:
        - 🏛️ pantheon-legends v0.2.0 (Analysis engines)
        - 🚀 FastAPI (Backend API)
        - 🎨 Streamlit (Web UI)
        - 🪙 Coinbase Advanced Trade API (Market data)
        
        For more information, visit the [GitHub repository](https://github.com/SpartanDigitalDotNet/pantheon-server).
        """)
    
    def _display_market_overview(self, overview: Dict):
        """Display market overview results"""
        st.subheader("Market Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pairs Analyzed", overview["pairs_analyzed"])
        
        with col2:
            st.metric("Successful", overview["successful_analyses"])
        
        with col3:
            st.metric("Failed", overview["failed_analyses"])
        
        with col4:
            success_rate = (overview["successful_analyses"] / overview["pairs_analyzed"]) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Top opportunities
        if overview["top_opportunities"]:
            st.subheader("🎯 Top Opportunities")
            
            for i, opp in enumerate(overview["top_opportunities"][:5]):
                with st.expander(f"#{i+1}: {opp['product_id']} (Score: {opp['score']:.3f})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Current Price", f"${opp['price']:.4f}")
                        if 'signal' in opp:
                            signal_color = "🟢" if opp['signal'] == "BUY" else "🔴" if opp['signal'] == "SELL" else "🟡"
                            st.metric("Signal", f"{signal_color} {opp['signal']}")
                    
                    with col2:
                        if 'confidence' in opp:
                            st.metric("Confidence", f"{opp['confidence']:.1%}")
                        if 'timeframe' in opp:
                            st.metric("Best Timeframe", opp['timeframe'])
                    
                    st.json(opp['analysis'])
        
        else:
            st.info("🔍 Click 'Generate Overview' to see top trading opportunities")
            
        # Market Sentiment Summary
        if overview.get("sentiment_breakdown"):
            st.subheader("📊 Market Sentiment")
            sentiment = overview["sentiment_breakdown"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bullish Signals", sentiment.get("bullish", 0))
            with col2:
                st.metric("Bearish Signals", sentiment.get("bearish", 0))
            with col3:
                st.metric("Neutral/Mixed", sentiment.get("neutral", 0))
            with col4:
                bullish_ratio = sentiment.get("bullish_ratio", 0)
                st.metric("Bullish Ratio", f"{bullish_ratio:.1%}")
        
        # Overall Market Sentiment
        if overview.get("market_sentiment"):
            sentiment_color = {
                "bullish": "🟢",
                "bearish": "🔴", 
                "neutral": "🟡"
            }
            st.info(f"**Overall Market Sentiment:** {sentiment_color.get(overview['market_sentiment'], '🟡')} {overview['market_sentiment'].title()}")
        
        # Analysis Engine Info
        if overview.get("pantheon_engines"):
            with st.expander("🔧 Analysis Engine Details"):
                st.write("**Available Pantheon Engines:**")
                for engine in overview["pantheon_engines"]:
                    st.write(f"• {engine}")
                st.write(f"**Legend Type Used:** {overview.get('legend_type', 'N/A')}")
                st.write(f"**Analysis Timestamp:** {overview.get('timestamp', 'N/A')}")
    
    def _get_tradingview_url(self, product_id: str) -> tuple:
        """Convert trading pair to TradingView URL formats (web and mobile app)"""
        # Convert BTC-USD to COINBASE:BTCUSD
        symbol = product_id.replace('-', '')
        tradingview_symbol = f"COINBASE:{symbol}"
        
        # Create both web and mobile app URLs
        web_url = f"https://www.tradingview.com/chart/?symbol={tradingview_symbol}&interval=1"
        mobile_url = f"tradingview://chart?symbol={tradingview_symbol}&interval=1"
        
        return web_url, mobile_url
    
    def _display_analysis_results(self, product_id: str, results: Dict):
        """Display individual analysis results with enhanced UI"""
        st.subheader(f"Analysis Results: {product_id}")
        
        # TradingView integration button (Option A - top placement)
        web_url, mobile_url = self._get_tradingview_url(product_id)
        
        st.markdown("### 📈 Quick Trading Actions")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Web TradingView button
            st.markdown(f"""
                <a href="{web_url}" target="_blank" style="text-decoration: none;">
                    <div style="
                        background: linear-gradient(90deg, #2962FF 0%, #1E53E5 100%);
                        color: white;
                        padding: 12px 16px;
                        border-radius: 8px;
                        font-weight: 600;
                        text-align: center;
                        font-size: 14px;
                        border: none;
                        cursor: pointer;
                        box-shadow: 0 2px 4px rgba(41, 98, 255, 0.3);
                        transition: all 0.2s ease;
                        display: inline-block;
                        width: 100%;
                        margin-bottom: 8px;
                    ">
                        �️ TradingView Web
                    </div>
                </a>
            """, unsafe_allow_html=True)
        
        with col2:
            # Mobile TradingView app button
            st.markdown(f"""
                <a href="{mobile_url}" style="text-decoration: none;">
                    <div style="
                        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
                        color: white;
                        padding: 12px 16px;
                        border-radius: 8px;
                        font-weight: 600;
                        text-align: center;
                        font-size: 14px;
                        border: none;
                        cursor: pointer;
                        box-shadow: 0 2px 4px rgba(255, 107, 53, 0.3);
                        transition: all 0.2s ease;
                        display: inline-block;
                        width: 100%;
                        margin-bottom: 8px;
                    ">
                        📱 TradingView App
                    </div>
                </a>
            """, unsafe_allow_html=True)
        
        with col3:
            # Copy symbol button for manual entry
            symbol_only = product_id.replace('-', '')
            st.markdown(f"""
                <div onclick="navigator.clipboard.writeText('COINBASE:{symbol_only}').then(() => {{
                    this.innerHTML = '✅ Copied!';
                    setTimeout(() => this.innerHTML = '📋 Copy Symbol', 2000);
                }})" style="
                    background: linear-gradient(90deg, #28A745 0%, #20C997 100%);
                    color: white;
                    padding: 12px 16px;
                    border-radius: 8px;
                    font-weight: 600;
                    text-align: center;
                    font-size: 14px;
                    border: none;
                    cursor: pointer;
                    box-shadow: 0 2px 4px rgba(40, 167, 69, 0.3);
                    transition: all 0.2s ease;
                    display: inline-block;
                    width: 100%;
                    margin-bottom: 8px;
                ">
                    📋 Copy Symbol
                </div>
            """, unsafe_allow_html=True)
        
        # Add helpful text
        st.caption("💡 **Web**: Opens in browser | **App**: Opens TradingView mobile app | **Copy**: Copy symbol for manual search")
        st.markdown("---")  # Separator line
        
        # Check if we have multiple timeframes with similar data
        if len(results) > 1:
            self._display_consolidated_analysis(product_id, results)
        else:
            # Single timeframe - show detailed view
            for timeframe, result in results.items():
                self._display_detailed_timeframe(timeframe, result, product_id)

    def _display_consolidated_analysis(self, product_id: str, results: Dict):
        """Display consolidated analysis across timeframes to avoid duplication"""
        
        # Main overview section
        sample_result = next(iter(results.values()))
        
        # Top-level summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Price", f"${sample_result['latest_price']:.2f}")
        with col2:
            st.metric("Candles", sample_result['candles_analyzed'])
        with col3:
            st.metric("Engine", sample_result['legend_type'])
        with col4:
            if 'analysis' in sample_result and 'price_change' in sample_result['analysis']:
                price_change = sample_result['analysis']['price_change']
                st.metric("Price Change", f"{price_change:.2f}%", delta=f"{price_change:.2f}%")
        
        # Timeframe comparison table
        st.markdown("### 📊 Multi-Timeframe Analysis")
        
        # Create comparison data with selection
        comparison_data = []
        timeframe_list = list(results.keys())
        
        for timeframe, result in results.items():
            if 'analysis' in result:
                analysis = result['analysis']
                comparison_data.append({
                    'Timeframe': timeframe.upper(),
                    'Signal': f"{self._get_signal_icon(analysis.get('signal', ''))} {analysis.get('signal', 'N/A').upper()}",
                    'Trend': f"{self._get_trend_icon(analysis.get('trend', ''))} {analysis.get('trend', 'N/A').upper()}",
                    'Confidence': f"{int(analysis.get('confidence', 0) * 100)}%",
                    'Momentum': f"{analysis.get('momentum', 0):.3f}",
                    'Vol Ratio': f"{analysis.get('volume_ratio', 0):.1f}x"
                })
        
        if comparison_data:
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            
            # Initialize session state for selected timeframe
            if 'selected_timeframe' not in st.session_state:
                st.session_state.selected_timeframe = timeframe_list[0]
            
            # Timeframe selection buttons instead of dropdown
            st.write("**Select timeframe for chart:**")
            cols = st.columns(len(timeframe_list))
            
            for i, tf in enumerate(timeframe_list):
                with cols[i]:
                    if st.button(
                        tf.upper(), 
                        key=f"tf_btn_{tf}",
                        type="primary" if st.session_state.selected_timeframe == tf else "secondary"
                    ):
                        st.session_state.selected_timeframe = tf
            
            selected_timeframe = st.session_state.selected_timeframe
            
            # Display the dataframe with highlighted selection
            st.dataframe(
                df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Timeframe": st.column_config.TextColumn("Timeframe", width="small"),
                    "Signal": st.column_config.TextColumn("Signal", width="medium"),
                    "Trend": st.column_config.TextColumn("Trend", width="medium"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Momentum": st.column_config.TextColumn("Momentum", width="small"),
                    "Vol Ratio": st.column_config.TextColumn("Vol Ratio", width="small")
                }
            )
        else:
            selected_timeframe = timeframe_list[0] if timeframe_list else "5m"
        
        # Chart section with fixed height and dynamic timeframe
        col_chart1, col_chart2 = st.columns([0.7, 0.3])
        
        with col_chart1:
            st.markdown(f"### 📈 Chart - {selected_timeframe.upper()}")
            self._display_interactive_chart(product_id, selected_timeframe)
        
        with col_chart2:
            st.markdown("### 💡 Consensus")
            self._display_consensus_recommendation(results)
        
        # Individual timeframe details (collapsed by default)
        with st.expander("🔍 Individual Timeframe Details", expanded=False):
            for timeframe, result in results.items():
                st.markdown(f"**{timeframe.upper()} Details:**")
                if 'analysis' in result:
                    st.json(result['analysis'])
                st.markdown("---")

    def _display_detailed_timeframe(self, timeframe: str, result: Dict, product_id: str):
        """Display detailed view for single timeframe"""
        with st.expander(f"📊 **{timeframe.upper()}**", expanded=True):
            if "error" in result:
                st.error(f"Error: {result['error']}")
                return
            
            # Two-column layout: Summary | Analysis (removed chart to avoid duplication)
            col_summary, col_analysis = st.columns([0.4, 0.6])
            
            with col_summary:
                st.metric("Price", f"${result['latest_price']:.2f}", 
                         delta=f"{result.get('analysis', {}).get('price_change', 0):.2f}%" if 'analysis' in result else None)
                st.metric("Candles", result['candles_analyzed'])
                st.metric("Engine", result['legend_type'])
            
            with col_analysis:
                if 'analysis' in result:
                    self._display_compact_analysis(result['analysis'])
                else:
                    st.warning("No analysis data")
            
            with st.expander("🔧 JSON", expanded=False):
                st.json(result['analysis'] if 'analysis' in result else result)

    def _get_signal_icon(self, signal: str) -> str:
        """Get icon for signal"""
        return {'buy': '🟢', 'sell': '🔴', 'hold': '🟡'}.get(signal.lower(), '⚪')
    
    def _get_trend_icon(self, trend: str) -> str:
        """Get icon for trend"""
        return {'bullish': '🐂', 'bearish': '🐻', 'sideways': '↔️'}.get(trend.lower(), '❓')
    
    def _display_consensus_recommendation(self, results: Dict):
        """Display consensus recommendation across timeframes"""
        signals = []
        confidences = []
        
        for result in results.values():
            if 'analysis' in result:
                analysis = result['analysis']
                signals.append(analysis.get('signal', '').lower())
                confidences.append(analysis.get('confidence', 0))
        
        if not signals:
            st.warning("No signals available")
            return
        
        # Calculate consensus
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        hold_count = signals.count('hold')
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        total = len(signals)
        
        if buy_count > sell_count and buy_count > hold_count:
            consensus = "BUY"
            icon = "🟢"
        elif sell_count > buy_count and sell_count > hold_count:
            consensus = "SELL"
            icon = "🔴"
        else:
            consensus = "HOLD"
            icon = "🟡"
        
        st.markdown(f"**Consensus:**")
        st.markdown(f"{icon} **{consensus}**")
        st.markdown(f"**Confidence:** {int(avg_confidence * 100)}%")
        
        # Vote breakdown
        st.markdown("**Votes:**")
        st.text(f"🟢 Buy: {buy_count}/{total}")
        st.text(f"🔴 Sell: {sell_count}/{total}")
        st.text(f"🟡 Hold: {hold_count}/{total}")
    
    def _display_tradingview_chart_compact(self, product_id: str):
        """Display compact TradingView chart widget for column layout"""
        symbol = product_id.replace('-', '')  # Convert BTC-USD to BTCUSD
        
        # TradingView widget HTML - optimized for column display
        tradingview_html = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container" style="height:400px;width:100%">
          <div class="tradingview-widget-container__widget" style="height:calc(100% - 20px);width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
          {{
          "autosize": true,
          "symbol": "COINBASE:{symbol}",
          "interval": "5",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "enable_publishing": false,
          "hide_top_toolbar": true,
          "hide_legend": true,
          "save_image": false,
          "hide_side_toolbar": true,
          "studies": [],
          "show_popup_button": false,
          "popup_width": "1000",
          "popup_height": "650",
          "container_id": "tradingview_chart_compact"
          }}
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
        
        # st.components.v1.html(tradingview_html, height=420)  # DISABLED - using main chart only
        st.info("📊 Chart function disabled - main chart displayed above")

    def _display_tradingview_chart(self, product_id: str):
        """Display TradingView chart widget"""
        symbol = product_id.replace('-', '')  # Convert BTC-USD to BTCUSD
        
        # TradingView widget HTML
        tradingview_html = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container" style="height:300px;width:100%">
          <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
          {{
          "autosize": true,
          "symbol": "COINBASE:{symbol}",
          "interval": "5",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "hide_legend": false,
          "save_image": false,
          "container_id": "tradingview_chart"
          }}
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
        
        # st.components.v1.html(tradingview_html, height=350)  # DISABLED - using main chart only
        st.info("📊 Chart function disabled - main chart displayed above")
    
    def _display_compact_analysis(self, analysis: Dict):
        """Ultra-compact professional analysis display"""
        
        # Signal row - single line
        signal = analysis.get('signal', 'unknown').upper()
        trend = analysis.get('trend', 'unknown').upper()
        confidence = int((analysis.get('confidence', 0) * 100))
        
        signal_icon = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(signal, '⚪')
        trend_icon = {'BULLISH': '🐂', 'BEARISH': '🐻', 'SIDEWAYS': '↔️'}.get(trend, '❓')
        
        st.markdown(f"**{signal_icon} {signal}** | **{trend_icon} {trend}** | **{confidence}%**")
        
        # Key metrics - condensed
        if 'indicators' in analysis:
            ind = analysis['indicators']
            if 'sma_20' in ind:
                st.text(f"SMA20: ${ind['sma_20']:.0f}")
            if 'trend_strength' in ind:
                st.text(f"Strength: {ind['trend_strength']:.3f}")
        
        # Market data - one line
        momentum = analysis.get('momentum', 0)
        vol_ratio = analysis.get('volume_ratio', 0)
        if momentum != 0:
            st.text(f"📊 Mom: {momentum:.3f} | Vol: {vol_ratio:.1f}x")
        
        # Recommendation - compact
        if signal == 'BUY' and confidence > 60:
            st.success("✅ Strong Buy")
        elif signal == 'SELL' and confidence > 60:
            st.error("❌ Strong Sell")
        elif signal == 'HOLD':
            st.info("⏸️ Hold")
        else:
            st.warning("⚠️ Unclear")

    def _display_interactive_chart(self, product_id: str, selected_timeframe: str):
        """Display TradingView chart with correct symbol format"""
        # Convert product_id to proper TradingView symbol format
        # product_id comes as "BTC-USD", we need "BTCUSD" for TradingView
        symbol = product_id.replace('-', '')  # BTC-USD becomes BTCUSD
        
        # Map timeframes to TradingView intervals
        timeframe_map = {
            '1m': '1',
            '5m': '5', 
            '15m': '15',
            '1h': '60',
            '4h': '240',
            '1d': '1D'
        }
        
        interval = timeframe_map.get(selected_timeframe, '5')
        
        # Use the correct TradingView widget embedding method
        tradingview_html = f"""
        <div style="height:400px;width:100%;position:relative;">
          <div id="tradingview_widget_{symbol}_{interval}" style="height:100%;width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
          {{
            "autosize": true,
            "symbol": "COINBASE:{symbol}",
            "interval": "{interval}",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "allow_symbol_change": false,
            "container_id": "tradingview_widget_{symbol}_{interval}",
            "studies": [
              "MASimple@tv-basicstudies"
            ],
            "hide_side_toolbar": false,
            "hide_top_toolbar": false,
            "save_image": false,
            "hide_legend": false
          }}
          </script>
        </div>
        """
        
        # Clean TradingView implementation with 9-period EMA
        chart_html = f"""
        <div style="height: 400px; width: 100%; border: 1px solid #333;">
            <iframe 
                src="https://www.tradingview.com/widgetembed/?frameElementId=tradingview_{symbol}_{interval}&symbol=COINBASE%3A{symbol}&interval={interval}&hidesidetoolbar=0&hidetoptoolbar=0&symboledit=0&saveimage=0&toolbarbg=f1f3f6&studies=EMA%409%40tv-basicstudies&hideideas=1&theme=dark&style=1&timezone=Etc%2FUTC"
                style="width: 100%; height: 100%; border: none;"
                frameborder="0"
                allowtransparency="true"
                scrolling="no">
            </iframe>
        </div>
        """
        
        st.components.v1.html(chart_html, height=420)

    def _display_mini_chart(self, product_id: str):
        """Mini chart placeholder - removed to prevent duplicate charts"""
        st.info("📊 Main chart displayed in analysis section above")
    
    
    def _display_batch_test_results(self, results: Dict, engine_type: LegendType, timeframe: str):
        """Display batch test results"""
        st.subheader("🧪 Batch Test Results")
        
        success_count = sum(1 for r in results.values() if "error" not in r)
        total_count = len(results)
        
        st.info(f"Tested {total_count} pairs using {engine_type.value} engine on {timeframe} timeframe")
        st.success(f"✅ {success_count}/{total_count} successful analyses")
        
        # Results table
        if success_count > 0:
            df_data = []
            for pair, result in results.items():
                if "error" not in result:
                    df_data.append({
                        "Pair": pair,
                        "Price": f"${result.get('latest_price', 0):.4f}",
                        "Candles": result.get('candles_analyzed', 0),
                        "Analysis": str(result.get('analysis', {}))[:100] + "..."
                    })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
    
    def _display_ema_signals(self, signals: Dict):
        """Display EMA(9) fakeout signals"""
        st.subheader(f"🎯 EMA(9) Signals: {signals['product_id']}")
        
        # Consensus signal
        consensus = signals['consensus']
        if consensus == "bullish":
            st.success(f"📈 BULLISH CONSENSUS")
        elif consensus == "bearish":
            st.error(f"📉 BEARISH CONSENSUS")
        else:
            st.info(f"➖ NEUTRAL CONSENSUS")
        
        # Timeframe breakdown
        for timeframe, tf_data in signals['timeframes'].items():
            with st.expander(f"📊 {timeframe.upper()} Analysis"):
                signal = tf_data['signal']
                price = tf_data.get('price')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if signal == "bullish":
                        st.success(f"📈 {signal.upper()}")
                    elif signal == "bearish":
                        st.error(f"📉 {signal.upper()}")
                    else:
                        st.info(f"➖ {signal.upper()}")
                
                with col2:
                    if price:
                        st.metric("Price", f"${price:.4f}")
                
                # Analysis details
                analysis = tf_data.get('analysis', {})
                if analysis:
                    st.json(analysis)


def main():
    """Main entry point for the Streamlit app"""
    ui = PantheonUI()
    ui.run()


if __name__ == "__main__":
    main()
