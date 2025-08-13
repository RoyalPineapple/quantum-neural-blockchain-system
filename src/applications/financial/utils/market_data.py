import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd

class MarketDataProcessor:
    """
    Process and prepare market data for quantum financial system.
    """
    
    def __init__(self, resolution: str):
        """
        Initialize market data processor.
        
        Args:
            resolution: Data time resolution ('1m', '5m', '1h', etc.)
        """
        self.resolution = resolution
        self.data_buffer = {}
        self.last_update = None
        
        # Feature calculation windows
        self.volatility_window = 20
        self.momentum_window = 10
        self.volume_window = 5
        
        # Technical indicators
        self.ma_windows = [5, 10, 20, 50]
        self.rsi_window = 14
        
    def process(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw market data into features.
        
        Args:
            market_data: Raw market data
            
        Returns:
            Dict[str, Any]: Processed features
        """
        # Update data buffer
        self._update_buffer(market_data)
        
        # Calculate features
        processed_data = {}
        for asset, data in market_data.items():
            processed_data[asset] = self._calculate_features(asset)
            
        return processed_data
        
    def get_current_price(self, asset: str) -> float:
        """
        Get current price for asset.
        
        Args:
            asset: Asset identifier
            
        Returns:
            float: Current price
        """
        if asset in self.data_buffer:
            return self.data_buffer[asset]['price'][-1]
        return 0.0
        
    def _update_buffer(self, market_data: Dict[str, Any]) -> None:
        """
        Update internal data buffer.
        
        Args:
            market_data: New market data
        """
        current_time = datetime.now()
        
        # Initialize buffer if needed
        if not self.data_buffer:
            self.data_buffer = {
                asset: {
                    'price': [],
                    'volume': [],
                    'timestamp': []
                }
                for asset in market_data.keys()
            }
            
        # Add new data
        for asset, data in market_data.items():
            if asset not in self.data_buffer:
                self.data_buffer[asset] = {
                    'price': [],
                    'volume': [],
                    'timestamp': []
                }
                
            self.data_buffer[asset]['price'].append(data['price'])
            self.data_buffer[asset]['volume'].append(data['volume'])
            self.data_buffer[asset]['timestamp'].append(current_time)
            
        # Trim old data
        self._trim_buffer()
        
        self.last_update = current_time
        
    def _trim_buffer(self) -> None:
        """Trim data buffer to maintain memory efficiency."""
        max_window = max(
            self.volatility_window,
            self.momentum_window,
            self.volume_window,
            max(self.ma_windows),
            self.rsi_window
        )
        
        for asset in self.data_buffer:
            for key in ['price', 'volume', 'timestamp']:
                self.data_buffer[asset][key] = self.data_buffer[asset][key][-max_window:]
                
    def _calculate_features(self, asset: str) -> Dict[str, float]:
        """
        Calculate features for asset.
        
        Args:
            asset: Asset identifier
            
        Returns:
            Dict[str, float]: Calculated features
        """
        data = self.data_buffer[asset]
        prices = np.array(data['price'])
        volumes = np.array(data['volume'])
        
        features = {
            'price': prices[-1],
            'volume': volumes[-1],
            'volatility': self._calculate_volatility(prices),
            'momentum': self._calculate_momentum(prices),
            'volume_profile': self._calculate_volume_profile(volumes),
            'technical_indicators': self._calculate_technical_indicators(prices)
        }
        
        return features
        
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """
        Calculate price volatility.
        
        Args:
            prices: Price array
            
        Returns:
            float: Volatility measure
        """
        if len(prices) >= self.volatility_window:
            returns = np.diff(np.log(prices[-self.volatility_window:]))
            return np.std(returns)
        return 0.0
        
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """
        Calculate price momentum.
        
        Args:
            prices: Price array
            
        Returns:
            float: Momentum indicator
        """
        if len(prices) >= self.momentum_window:
            return (prices[-1] / prices[-self.momentum_window]) - 1
        return 0.0
        
    def _calculate_volume_profile(self, volumes: np.ndarray) -> Dict[str, float]:
        """
        Calculate volume profile metrics.
        
        Args:
            volumes: Volume array
            
        Returns:
            Dict[str, float]: Volume metrics
        """
        if len(volumes) >= self.volume_window:
            recent_volumes = volumes[-self.volume_window:]
            return {
                'mean': np.mean(recent_volumes),
                'std': np.std(recent_volumes),
                'trend': np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            }
        return {
            'mean': 0.0,
            'std': 0.0,
            'trend': 0.0
        }
        
    def _calculate_technical_indicators(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate technical indicators.
        
        Args:
            prices: Price array
            
        Returns:
            Dict[str, float]: Technical indicators
        """
        indicators = {}
        
        # Moving averages
        for window in self.ma_windows:
            if len(prices) >= window:
                ma = np.mean(prices[-window:])
                indicators[f'ma_{window}'] = ma
            else:
                indicators[f'ma_{window}'] = prices[-1]
                
        # RSI
        if len(prices) >= self.rsi_window:
            returns = np.diff(prices[-self.rsi_window-1:])
            gains = np.maximum(returns, 0)
            losses = -np.minimum(returns, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
            indicators['rsi'] = rsi
        else:
            indicators['rsi'] = 50
            
        return indicators
        
    def get_historical_data(self, asset: str,
                          window: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical data for asset.
        
        Args:
            asset: Asset identifier
            window: Optional lookback window
            
        Returns:
            pd.DataFrame: Historical data
        """
        if asset not in self.data_buffer:
            return pd.DataFrame()
            
        data = self.data_buffer[asset]
        df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'price': data['price'],
            'volume': data['volume']
        })
        
        if window is not None:
            df = df.tail(window)
            
        return df
        
    def calculate_returns(self, asset: str,
                         window: Optional[int] = None) -> np.ndarray:
        """
        Calculate historical returns.
        
        Args:
            asset: Asset identifier
            window: Optional lookback window
            
        Returns:
            np.ndarray: Returns array
        """
        if asset not in self.data_buffer:
            return np.array([])
            
        prices = np.array(self.data_buffer[asset]['price'])
        returns = np.diff(np.log(prices))
        
        if window is not None:
            returns = returns[-window:]
            
        return returns
        
    def calculate_correlation(self, asset1: str, asset2: str,
                            window: Optional[int] = None) -> float:
        """
        Calculate correlation between assets.
        
        Args:
            asset1: First asset
            asset2: Second asset
            window: Optional correlation window
            
        Returns:
            float: Correlation coefficient
        """
        if asset1 not in self.data_buffer or asset2 not in self.data_buffer:
            return 0.0
            
        returns1 = self.calculate_returns(asset1, window)
        returns2 = self.calculate_returns(asset2, window)
        
        if len(returns1) > 0 and len(returns2) > 0:
            return np.corrcoef(returns1, returns2)[0,1]
        return 0.0
