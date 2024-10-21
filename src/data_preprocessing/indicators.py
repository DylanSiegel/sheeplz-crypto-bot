import pandas as pd
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, UlcerIndex
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, VolumePriceTrendIndicator
from loguru import logger

def add_technical_indicators(df):
    try:
        df_list = []
        for tf in df['timeframe'].unique():
            df_tf = df[df['timeframe'] == tf].copy()
            if df_tf.isnull().values.any():
                logger.warning(f"NaN values found in data for timeframe {tf} before adding technical indicators. Dropping NaNs.")
                df_tf.dropna(inplace=True)

            # Ensure there is enough data for indicator calculation
            min_required_length = 30  # Adjust based on the maximum window size required by indicators
            if len(df_tf) < min_required_length:
                logger.warning(f"Insufficient data for timeframe {tf}. Skipping indicator calculation.")
                continue

            logger.debug(f"Adding technical indicators for timeframe {tf}...")

            # Add EMA
            df_tf['ema_14'] = EMAIndicator(close=df_tf['close'], window=14).ema_indicator()

            # Add SMA
            df_tf['sma_14'] = SMAIndicator(close=df_tf['close'], window=14).sma_indicator()

            # Add RSI
            df_tf['rsi_14'] = RSIIndicator(close=df_tf['close'], window=14).rsi()

            # Add Stochastic Oscillator
            stoch = StochasticOscillator(high=df_tf['high'], low=df_tf['low'], close=df_tf['close'], window=14)
            df_tf['stoch_k'] = stoch.stoch()
            df_tf['stoch_d'] = stoch.stoch_signal()

            # Add Bollinger Bands
            bollinger = BollingerBands(close=df_tf['close'], window=20, window_dev=2)
            df_tf['bb_mavg'] = bollinger.bollinger_mavg()
            df_tf['bb_hband'] = bollinger.bollinger_hband()
            df_tf['bb_lband'] = bollinger.bollinger_lband()
            df_tf['bb_pband'] = bollinger.bollinger_pband()
            df_tf['bb_wband'] = bollinger.bollinger_wband()

            # Add Keltner Channel
            keltner = KeltnerChannel(high=df_tf['high'], low=df_tf['low'], close=df_tf['close'], window=20)
            df_tf['kc_hband'] = keltner.keltner_channel_hband()
            df_tf['kc_lband'] = keltner.keltner_channel_lband()
            df_tf['kc_mband'] = keltner.keltner_channel_mband()
            df_tf['kc_pband'] = keltner.keltner_channel_pband()
            df_tf['kc_wband'] = keltner.keltner_channel_wband()

            # Add Average True Range (ATR)
            df_tf['atr_14'] = AverageTrueRange(high=df_tf['high'], low=df_tf['low'], close=df_tf['close'], window=14).average_true_range()

            # Add On-Balance Volume (OBV)
            df_tf['obv'] = OnBalanceVolumeIndicator(close=df_tf['close'], volume=df_tf['volume']).on_balance_volume()

            # Add MACD
            macd = MACD(close=df_tf['close'], window_slow=26, window_fast=12, window_sign=9)
            df_tf['macd'] = macd.macd()
            df_tf['macd_signal'] = macd.macd_signal()
            df_tf['macd_diff'] = macd.macd_diff()

            # Add ADX
            adx = ADXIndicator(high=df_tf['high'], low=df_tf['low'], close=df_tf['close'], window=14)
            df_tf['adx'] = adx.adx()
            df_tf['adx_pos'] = adx.adx_pos()
            df_tf['adx_neg'] = adx.adx_neg()

            # Add Ulcer Index (UI)
            df_tf['ulcer_index'] = UlcerIndex(close=df_tf['close'], window=14).ulcer_index()

            # Add Accumulation/Distribution Index (ADI)
            if len(df_tf) > 1:  # Ensure there is more than one row for volume-based indicators
                df_tf['adi'] = AccDistIndexIndicator(high=df_tf['high'], low=df_tf['low'], close=df_tf['close'], volume=df_tf['volume']).acc_dist_index()

                # Add Chaikin Money Flow (CMF)
                df_tf['cmf'] = ChaikinMoneyFlowIndicator(high=df_tf['high'], low=df_tf['low'], close=df_tf['close'], volume=df_tf['volume'], window=20).chaikin_money_flow()

                # Add Ease of Movement (EoM)
                eom = EaseOfMovementIndicator(high=df_tf['high'], low=df_tf['low'], volume=df_tf['volume'], window=14)
                df_tf['eom'] = eom.ease_of_movement()

                # Add Volume Price Trend (VPT)
                df_tf['vpt'] = VolumePriceTrendIndicator(close=df_tf['close'], volume=df_tf['volume']).volume_price_trend()

            if df_tf.empty:
                logger.warning(f"Dataframe for timeframe {tf} is empty after adding indicators. Skipping this timeframe.")
                continue

            df_list.append(df_tf)

        if not df_list:
            logger.error("No dataframes were processed successfully. Returning original dataframe.")
            return df

        df = pd.concat(df_list, axis=0)
        df.sort_index(inplace=True)  # Ensure the index is sorted after concatenation
        logger.info("Technical indicators added successfully.")
        return df
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        return df
