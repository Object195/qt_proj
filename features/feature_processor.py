import numpy as np
import pandas as pd
import warnings
from numpy.lib.stride_tricks import sliding_window_view

class FeatureProcessor:
    """
    A class to construct an additional layer of engineered features from processed_data.csv.
    Transforms noisy raw technical indicators into smoother features (like rolling slopes/means).
    """
    def __init__(self):
        self.feature_names = []
        self.feature_groups = {}

    def _register_feature(self, group_name: str, feature_name: str):
        """
        Internal helper method to append the feature name to both 
        self.feature_names and the correct list in self.feature_groups.
        """
        if feature_name not in self.feature_names:
            self.feature_names.append(feature_name)
            
        if group_name not in self.feature_groups:
            self.feature_groups[group_name] = []
            
        if feature_name not in self.feature_groups[group_name]:
            self.feature_groups[group_name].append(feature_name)

    def calculate_grouped_importance(self, feature_importance_dict: dict) -> pd.Series:
        """
        Sums over the gain of all features in a specific group, normalizes 
        the results so they sum to 1, and returns a sorted Series.
        """
        grouped_importance = {group: 0.0 for group in self.feature_groups.keys()}
        for group, feats in self.feature_groups.items():
            for feat in feats:
                if feat in feature_importance_dict:
                    grouped_importance[group] += feature_importance_dict[feat]
                    
        series = pd.Series(grouped_importance)
        total_gain = series.sum()
        if total_gain > 0:
            series = series / total_gain
        return series[series > 0].sort_values(ascending=False)

    def calculate_individual_importance(self, feature_importance_dict: dict, group_name: str) -> pd.Series:
        """
        Extracts the gain of all features in a specific group, normalizes 
        the results so they sum to 1, and returns a sorted Series.
        """
        if group_name not in self.feature_groups:
            raise ValueError(f"Group '{group_name}' not found in feature groups.")
            
        individual_importance = {}
        for feat in self.feature_groups[group_name]:
            # Provide a default value of 0.0 if the feature was not used by XGBoost
            individual_importance[feat] = feature_importance_dict.get(feat, 0.0)
                
        series = pd.Series(individual_importance)
        total_gain = series.sum()
        if total_gain > 0:
            series = series / total_gain
        # Return all features, even those with 0 importance, so they show up on the plot
        return series.sort_values(ascending=False)

    def add_rolling_mean(self, df: pd.DataFrame, col_name: str, period: int) -> pd.DataFrame:
        """
        Calculates the simple rolling mean, creates the new column in df, and registers it.
        """
        feature_name = f"{col_name}_mean_{period}"
        group_name = col_name
        
        # Calculates the mean using only data from index i-period+1 up to i.
        # This is inherently backward-looking and has zero future bias.
        df[feature_name] = df[col_name].rolling(window=period).mean()
        self._register_feature(group_name, feature_name)
        
        return df

    def add_linear_slope(self, df: pd.DataFrame, col_name: str, period: int) -> pd.DataFrame:
        """
        Calculates rolling linear regression slope and R-squared fit 
        using a highly optimized vectorized NumPy approach.
        Appends the features to df.
        """
        if period < 2:
            raise ValueError("Period must be >= 2 for linear regression")
            
        feature_slope = f"{col_name}_slope_{period}"
        feature_r2 = f"{col_name}_R2_{period}"
        
        group_name = col_name
        
        y = df[col_name].values
        
        # Pad the beginning with NaNs to align sliding window output with pandas rolling.
        # The pad is exactly (period - 1), ensuring the sliding window at index i 
        # strictly evaluates the trailing history: elements [i - period + 1, ..., i].
        # No future bias is introduced.
        y_padded = np.pad(y, (period - 1, 0), mode='constant', constant_values=np.nan)
        
        # Shape: (len(df), period)
        y_windows = sliding_window_view(y_padded, window_shape=period)
        
        # x values for the regression (0 to period - 1)
        x = np.arange(period)
        x_diff = x - np.mean(x)
        S_xx = np.sum(x_diff ** 2) # S_xx is a scalar since x is constant across windows
        
        # Vectorized calculation over all windows
        y_diff = y_windows - np.mean(y_windows, axis=1, keepdims=True)
        S_xy = np.sum(y_diff * x_diff, axis=1)
        S_yy = np.sum(y_diff ** 2, axis=1)
        
        # Calculate Slope (beta)
        slope = S_xy / S_xx
        
        # Calculate R-squared
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = (S_xy ** 2) / (S_xx * S_yy)
            # Handle perfectly horizontal lines where variance of y (S_yy) is 0
            # Set to 0.0 to match SciPy's default behavior for horizontal line fits
            r2 = np.where(S_yy == 0, 0.0, r2)

        df[feature_slope] = slope
        df[feature_r2] = r2
        
        self._register_feature(group_name, feature_slope)
        self._register_feature(group_name, feature_r2)
        
        return df

    def add_cross_events(self, df: pd.DataFrame, col_name: str, period: int = 20, shift: float = 0.0) -> pd.DataFrame:
        """
        Processes crossing events (up and down) against a specified shift value.
        Generates features for days since the last crosses and the total number of crosses 
        in the specified period.
        """
 
        feature_days_down = f"{col_name}_cd"
        feature_days_up = f"{col_name}_cu"
        feature_total_crosses = f"{col_name}_c_tot"
        
        group_name = col_name
        
        x = df[col_name] - shift
        x_prev = x.shift(1)
        
        # Identify crossing points.
        # x_prev is t-1, x is t. If x_prev > 0 and x < 0, a cross down just occurred.
        cross_down = (x_prev > 0) & (x < 0)
        cross_up = (x_prev < 0) & (x > 0)
        
        # 3. Total number of crosses happened in the period
        any_cross = cross_down | cross_up
        df[feature_total_crosses] = any_cross.rolling(window=period).sum()
        
        # 1 & 2. Days after the last down_cross and up_cross
        # Use an integer sequence to represent trading days (rows) securely mapped to the DataFrame index
        row_indices = pd.Series(np.arange(len(df)), index=df.index)
        
        # .where() inserts NaNs where the condition is False. .ffill() carries forward the row index of the last True event.
        # If no cross has occurred yet at the beginning of the dataset, it safely defaults to NaN (unknown).
        last_down_idx = row_indices.where(cross_down).ffill()
        last_up_idx = row_indices.where(cross_up).ffill()
        
        df[feature_days_down] = row_indices - last_down_idx
        df[feature_days_up] = row_indices - last_up_idx
        
        self._register_feature(group_name, feature_days_down)
        self._register_feature(group_name, feature_days_up)
        self._register_feature(group_name, feature_total_crosses)
        
        return df

    def add_spike_events(self, df: pd.DataFrame, col_name: str, period: int = 20, shift: float = 0.0, above: bool = True,full_statistic=True) -> pd.DataFrame:
        """
        Processes spike events (values above or below a specified shift).
        Generates features for days since the last spike and the total number of spikes 
        in the specified period.
        """
            
        direction = "above" if above else "below"
        feature_days_since = f"{col_name}_ds_{direction}"
        feature_total_spikes = f"{col_name}_ts_{direction}"
        
        group_name = col_name
        
        x = df[col_name] - shift
        
        if above:
            is_spike = x > 0
        else:
            is_spike = x < 0
        if full_statistic:
            # 2. Total number days that satisfies x-shift >0 /<0
            df[feature_total_spikes] = is_spike.rolling(window=period).sum()
            self._register_feature(group_name, feature_total_spikes)

        # 1. Days past since the last event
        row_indices = pd.Series(np.arange(len(df)), index=df.index)
        last_spike_idx = row_indices.where(is_spike).ffill()
        
        df[feature_days_since] = row_indices - last_spike_idx
        
        self._register_feature(group_name, feature_days_since)
        
        
        return df

    def add_rolling_std(self, df: pd.DataFrame, col_name: str, period: int) -> pd.DataFrame:
        """
        Calculates the rolling standard deviation.
        """
        feature_name = f"{col_name}_std_{period}"
        group_name = col_name
        
        df[feature_name] = df[col_name].rolling(window=period).std()
        self._register_feature(group_name, feature_name)
        
        return df

    def add_rolling_percentile(self, df: pd.DataFrame, col_name: str, period: int) -> pd.DataFrame:
        """
        Calculates the rolling percentile (rank) of the current element 
        within a specified history window efficiently using pandas native rank.
        """
        feature_name = f"{col_name}_pct_{period}"
        group_name = col_name
        
        df[feature_name] = df[col_name].rolling(window=period).rank(pct=True)
        self._register_feature(group_name, feature_name)
        
        return df

    def add_rolling_acf(self, df: pd.DataFrame, col_name: str, period: int, lag: int = 1) -> pd.DataFrame:
        """
        Calculates the rolling autocorrelation of a feature within a lookback window.
        """
        feature_name = f"{col_name}_acf_{period}_{lag}"
        group_name = col_name
        
        df[feature_name] = df[col_name].rolling(window=period).corr(df[col_name].shift(lag))
        self._register_feature(group_name, feature_name)
        
        return df

    def add_rolling_hurst(self, df: pd.DataFrame, col_name: str, period: int, num_lags: int = 8) -> pd.DataFrame:
        """
        Calculates the rolling Hurst exponent using the standard Rescaled Range (R/S) Analysis algorithm.
        Assumes the input column is already a stationary series (e.g., Log Returns).
        """
        max_target_lag = period // 2
        if max_target_lag <= 8:
            raise ValueError("period must be > 16 to compute a linear fit with lags from 8 to period/2.")
            
        feature_name = f"{col_name}_hurst_{period}_{num_lags}"
        group_name = col_name
        
        # 1. Construct a log-spaced array of lags (Tau)
        lags = np.unique(np.geomspace(8, max_target_lag, num=num_lags).astype(int))
        if len(lags) < 2:
            raise ValueError("Not enough distinct lags generated. Increase period or num_lags.")
            
        # X variables for the log-log regression (constant across all time steps)
        log_lags = np.log(lags)
        log_lags_diff = log_lags - np.mean(log_lags)
        # Pre-calculate the linear regression weights for the dot product
        weights = log_lags_diff / np.sum(log_lags_diff ** 2)
        
        data_values = df[col_name].values
        # Pad the beginning with NaNs to align sliding window output with pandas rolling
        y_padded = np.pad(data_values, (period - 1, 0), mode='constant', constant_values=np.nan)
        
        # Extract rolling windows: Shape (T, period)
        y_windows = sliding_window_view(y_padded, window_shape=period)
        T = y_windows.shape[0]
        
        log_rs_matrix = np.full((T, len(lags)), np.nan)

        # Suppress expected RuntimeWarnings for slices containing only NaNs, which occur at the start of the series.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            for i, tau in enumerate(lags):
                # 3. Non-Overlapping Chunks
                num_chunks = period // tau
                if num_chunks == 0:
                    continue
                    
                # Take the most recent data to form complete chunks
                truncated_windows = y_windows[:, -num_chunks * tau:]
                
                # Reshape into (T, num_chunks, tau)
                chunks = truncated_windows.reshape(T, num_chunks, tau)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    # 4. The R/S Math
                    chunk_means = np.nanmean(chunks, axis=2, keepdims=True)
                    mean_adj = chunks - chunk_means
                    
                    # Cumulative sum (integrating the returns back into a detrended price path)
                    cum_sum = np.nancumsum(mean_adj, axis=2)
                    
                    # Range (Max - Min)
                    R = np.nanmax(cum_sum, axis=2) - np.nanmin(cum_sum, axis=2)
                    
                    # Standard Deviation
                    S = np.nanstd(chunks, axis=2, ddof=0)
                    
                    # R/S
                    rs = R / S
                    
                    # Average R/S across chunks
                    mean_rs = np.nanmean(rs, axis=1)
                    
                    log_rs_matrix[:, i] = np.log(mean_rs)
            
        # 5. Regression
        valid_mask = ~np.isnan(log_rs_matrix) & ~np.isinf(log_rs_matrix)
        
        with np.errstate(invalid='ignore'):
            slope = np.sum(log_rs_matrix * weights, axis=1)
            
        # Handle rows with partial NaNs individually (e.g., edges or missing data)
        has_nans = ~np.all(valid_mask, axis=1)
        valid_rows_with_some_nans = has_nans & np.any(valid_mask, axis=1)
        
        if np.any(valid_rows_with_some_nans):
            for idx in np.where(valid_rows_with_some_nans)[0]:
                valid_idx = valid_mask[idx]
                if np.sum(valid_idx) >= 2:
                    v_log_lags = log_lags[valid_idx]
                    v_log_rs = log_rs_matrix[idx, valid_idx]
                    
                    v_log_lags_diff = v_log_lags - np.mean(v_log_lags)
                    v_weights = v_log_lags_diff / np.sum(v_log_lags_diff ** 2)
                    slope[idx] = np.sum(v_weights * v_log_rs)
                else:
                    slope[idx] = np.nan
                    
        # Replace entirely invalid rows with NaN
        slope[~np.any(valid_mask, axis=1)] = np.nan
        
        df[feature_name] = pd.Series(slope, index=df.index)
        self._register_feature(group_name, feature_name)
        
        return df

    def add_raw_feature(self, col_name: str) -> pd.DataFrame:
        """
        Registers the existing raw feature column to the feature list 
        for inclusion in training data without duplicating it.
        """
        self._register_feature(col_name, col_name)
        

    def add_z_score(self, df: pd.DataFrame, col_name: str, period: int=20) -> pd.DataFrame:
        """
        Calculates the rolling Z-score (standard score) over a specified lookback window.
        """
        feature_name = f"{col_name}_z_{period}"
        group_name = col_name
        
        rolling_mean = df[col_name].rolling(window=period).mean()
        rolling_std = df[col_name].rolling(window=period).std()
        
        df[feature_name] = (df[col_name] - rolling_mean) / rolling_std
        self._register_feature(group_name, feature_name)
        
        return df

    def add_velocity_acceleration(self, df: pd.DataFrame, col_name: str, period: int, smooth_type: str = 'ma') -> pd.DataFrame:
        """
        Calculates the velocity (1st derivative) and acceleration (2nd derivative) 
        of a feature using backward differences, then smooths the result using MA or EMA.
        """
        feature_vel = f"{col_name}_vel_{smooth_type}_{period}"
        feature_acc = f"{col_name}_acc_{smooth_type}_{period}"
        group_name = col_name
        
        # Get default period for rolling std from config
        std_period = 20
        rolling_std = df[col_name].rolling(window=std_period).std()
        
        # 1st order derivative (backward difference: x[t] - x[t-1]) normalized by rolling std
        vel_raw = df[col_name].diff() / rolling_std
        
        # 2nd order derivative (backward difference of 1st derivative: v[t] - v[t-1])
        acc_raw = vel_raw.diff()
        
        if smooth_type == 'ma':
            df[feature_vel] = vel_raw.rolling(window=period).mean()
            df[feature_acc] = acc_raw.rolling(window=period).mean()
        elif smooth_type == 'ema':
            df[feature_vel] = vel_raw.ewm(span=period, adjust=False).mean()
            df[feature_acc] = acc_raw.ewm(span=period, adjust=False).mean()
        else:
            raise ValueError("smooth_type must be either 'ma' or 'ema'")
            
        self._register_feature(group_name, feature_vel)
        self._register_feature(group_name, feature_acc)
        
        return df

    def add_rolling_max_min(self, df: pd.DataFrame, col_name: str, period: int) -> pd.DataFrame:
        """
        Calculates the rolling maximum and minimum, generating two columns.
        """
        feature_max = f"{col_name}_max_{period}"
        feature_min = f"{col_name}_min_{period}"
        group_name = col_name
        
        df[feature_max] = df[col_name].rolling(window=period).max()
        df[feature_min] = df[col_name].rolling(window=period).min()
        
        self._register_feature(group_name, feature_max)
        self._register_feature(group_name, feature_min)
        
        return df

    def add_rolling_skewness(self, df: pd.DataFrame, col_name: str, period: int) -> pd.DataFrame:
        """
        Calculates the rolling skewness.
        """
        feature_name = f"{col_name}_skew_{period}"
        group_name = col_name
        
        df[feature_name] = df[col_name].rolling(window=period).skew()
        self._register_feature(group_name, feature_name)
        
        return df

    def add_rolling_kurtosis(self, df: pd.DataFrame, col_name: str, period: int) -> pd.DataFrame:
        """
        Calculates the rolling kurtosis.
        """
        feature_name = f"{col_name}_kurt_{period}"
        group_name = col_name
        
        df[feature_name] = df[col_name].rolling(window=period).kurt()
        self._register_feature(group_name, feature_name)
        
        return df