"""
Market Data Module
Fetches and processes market data from Fyers API
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
import pandas as pd
import numpy as np


class MarketDataProvider:
    """
    Provides market data and technical indicators
    """

    def __init__(self, fyers_client: fyersModel.FyersModel, index: str = "NIFTY"):
        """
        Initialize market data provider

        Args:
            fyers_client: Initialized Fyers API client
            index: Trading index (NIFTY or SENSEX)
        """
        self.fyers = fyers_client
        self.index = index

        # Symbol mapping
        self.index_symbol = "NSE:NIFTY50-INDEX" if index == "NIFTY" else "BSE:SENSEX-INDEX"

        self.logger = logging.getLogger("MarketData")

    def get_current_price(self) -> float:
        """Get current spot price"""
        try:
            response = self.fyers.quotes({"symbols": self.index_symbol})

            self.logger.debug(f"Price API response: {response}")

            if response and response.get("s") == "ok":
                data = response.get("d", [])
                if data and len(data) > 0:
                    price = float(data[0].get("v", {}).get("lp", 0))
                    self.logger.info(f"Current {self.index} price: {price}")
                    return price
            else:
                self.logger.error(f"Price API failed: {response}")

            return 0.0

        except Exception as e:
            self.logger.error(f"Error fetching price: {str(e)}")
            return 0.0

    def get_historical_candles(self, timeframe: str = "1", candles: int = 100) -> List[Dict]:
        """
        Get historical candle data

        Args:
            timeframe: Timeframe ("1" for 1-min, "5" for 5-min, etc.)
            candles: Number of candles

        Returns:
            List of candle dictionaries
        """
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=2)  # 2 days should be enough

            # Format dates
            from_timestamp = int(from_date.timestamp())
            to_timestamp = int(to_date.timestamp())

            # Fetch data
            data = {
                "symbol": self.index_symbol,
                "resolution": timeframe,
                "date_format": "1",  # Unix timestamp
                "range_from": str(from_timestamp),
                "range_to": str(to_timestamp),
                "cont_flag": "1"
            }

            response = self.fyers.history(data=data)

            self.logger.info(f"Fyers API request: {data}")
            self.logger.info(f"Fyers API response: {response}")

            if response and response.get("s") == "ok":
                candles_data = response.get("candles", [])

                self.logger.info(f"Received {len(candles_data)} candles from Fyers")

                # Convert to list of dicts
                candle_list = []
                for candle in candles_data:
                    if len(candle) >= 5:
                        candle_list.append({
                            "timestamp": candle[0],
                            "open": float(candle[1]),
                            "high": float(candle[2]),
                            "low": float(candle[3]),
                            "close": float(candle[4]),
                            "volume": int(candle[5]) if len(candle) > 5 else 0
                        })

                return candle_list[-candles:]  # Return last N candles
            else:
                self.logger.error(f"Fyers history API failed. Status: {response.get('s') if response else 'No response'}, "
                                f"Message: {response.get('message') if response else 'None'}")

            return []

        except Exception as e:
            self.logger.error(f"Error fetching candles: {str(e)}")
            return []

    def calculate_rsi(self, candles: List[Dict], period: int = 6) -> float:
        """Calculate RSI"""
        try:
            if len(candles) < period + 1:
                return 50.0  # Neutral RSI

            closes = [c["close"] for c in candles]

            # Calculate price changes
            deltas = np.diff(closes)

            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            # Calculate average gain/loss
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi)

        except Exception as e:
            self.logger.error(f"RSI calculation error: {str(e)}")
            return 50.0

    def calculate_ema(self, candles: List[Dict], period: int = 13) -> float:
        """Calculate EMA"""
        try:
            if len(candles) < period:
                return 0.0

            closes = [c["close"] for c in candles]

            # Calculate EMA
            ema = closes[0]
            multiplier = 2 / (period + 1)

            for close in closes[1:]:
                ema = (close * multiplier) + (ema * (1 - multiplier))

            return float(ema)

        except Exception as e:
            self.logger.error(f"EMA calculation error: {str(e)}")
            return 0.0

    def calculate_atr(self, candles: List[Dict], period: int = 10) -> float:
        """Calculate ATR (Average True Range)"""
        try:
            if len(candles) < period + 1:
                return 0.0

            true_ranges = []
            for i in range(1, len(candles)):
                high = candles[i]["high"]
                low = candles[i]["low"]
                prev_close = candles[i - 1]["close"]

                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr)

            atr = np.mean(true_ranges[-period:])
            return float(atr)

        except Exception as e:
            self.logger.error(f"ATR calculation error: {str(e)}")
            return 0.0

    def calculate_supertrend(self,
                           candles: List[Dict],
                           period: int = 10,
                           multiplier: float = 3.0) -> Dict[str, Any]:
        """Calculate Supertrend"""
        try:
            if len(candles) < period + 1:
                return {"value": 0.0, "direction": "NEUTRAL"}

            # Calculate ATR
            atr = self.calculate_atr(candles, period)

            # Get last candle
            last_candle = candles[-1]
            hl_avg = (last_candle["high"] + last_candle["low"]) / 2

            # Basic Supertrend calculation
            basic_upper = hl_avg + (multiplier * atr)
            basic_lower = hl_avg - (multiplier * atr)

            # Determine trend
            close = last_candle["close"]

            if close > basic_upper:
                direction = "UP"
                value = basic_lower
            elif close < basic_lower:
                direction = "DOWN"
                value = basic_upper
            else:
                direction = "NEUTRAL"
                value = hl_avg

            return {
                "value": float(value),
                "direction": direction,
                "upper": float(basic_upper),
                "lower": float(basic_lower)
            }

        except Exception as e:
            self.logger.error(f"Supertrend calculation error: {str(e)}")
            return {"value": 0.0, "direction": "NEUTRAL"}

    def get_market_data(self) -> Dict[str, Any]:
        """
        Get complete market data package for agents

        Returns:
            Dictionary containing all market data
        """
        try:
            self.logger.info("Fetching market data...")

            # Get current price
            current_price = self.get_current_price()

            # Get historical candles
            candles = self.get_historical_candles(timeframe="1", candles=100)

            if len(candles) == 0:
                self.logger.error("No candle data available")
                return {}

            # Calculate indicators
            rsi = self.calculate_rsi(candles, period=6)
            ema13 = self.calculate_ema(candles, period=13)
            atr = self.calculate_atr(candles, period=10)
            supertrend = self.calculate_supertrend(candles, period=10, multiplier=3.0)

            # Get price changes
            first_candle = candles[0]
            last_candle = candles[-1]

            price_change = last_candle["close"] - first_candle["open"]
            price_change_percent = (price_change / first_candle["open"]) * 100

            # Determine trend
            if supertrend["direction"] == "UP":
                trend = "UPTREND"
            elif supertrend["direction"] == "DOWN":
                trend = "DOWNTREND"
            else:
                trend = "RANGING"

            # Build data package
            market_data = {
                "sentiment_data": {
                    "current_price": current_price,
                    "price_change": price_change,
                    "price_change_percent": price_change_percent,
                    "volatility": atr,
                    "candles": candles[-10:],  # Last 10 candles
                    "trend": trend,
                    "rsi": rsi
                },
                "indicator_data": {
                    "price": current_price,
                    "rsi": rsi,
                    "ema": {"ema13": ema13},
                    "supertrend": supertrend,
                    "atr": atr
                },
                "price_data": {
                    "current_price": current_price,
                    "candles": candles[-20:],  # Last 20 candles
                    "high_of_day": max([c["high"] for c in candles]),
                    "low_of_day": min([c["low"] for c in candles]),
                    "prev_close": candles[-2]["close"] if len(candles) > 1 else current_price,
                    "trend": trend
                }
            }

            self.logger.info(f"Market data collected: Price={current_price:.2f}, "
                           f"RSI={rsi:.2f}, Trend={trend}")

            return market_data

        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return {}
