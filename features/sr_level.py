import numpy as np
import pandas as pd
import pandas_ta as ta
from config import SR_PARAMS
class SRLevelDetector:
    def __init__(self, print_para=False):
        self.N = SR_PARAMS['window_size']
        self.p = SR_PARAMS['penalty_fac']
        self.std_fac= SR_PARAMS['std_fac']
        self.atr_fac= SR_PARAMS['atr_fac']
        self.bb_length = SR_PARAMS['bb_length']
        self.bb_std = SR_PARAMS['bb_std']
        self.vol_filter = SR_PARAMS['vol_filter']
        self.Nvol = SR_PARAMS['vol_window']
        self.print_para = print_para

        # List of SR levels. Each level is a dict: {'V': float, 'M': float, 'S': float}
        self.levels = [] 
        self.pending_touches = set()

    def _get_sigma(self, level):
        if level['V'] == 0:
            return 0.0
        return np.sqrt(level['S'] / level['V'])

    def _update_level_state(self, level, p_new, v_new):
        """Updates an SR level using Welford's online algorithm."""
        old_M = level['M']
        new_V = level['V'] + v_new
        
        if new_V == 0:
            return

        # Welford's update
        # M_new = M + (v_new / V_new) * (p_new - M)
        new_M = old_M + (v_new / new_V) * (p_new - old_M)
        
        # S_new = S + v_new * (p_new - M) * (p_new - M_new)
        new_S = level['S'] + v_new * (p_new - old_M) * (p_new - new_M)
        
        level['V'] = new_V
        level['M'] = new_M
        level['S'] = new_S

    def _process_point(self, price, volume, atr):
        """
        Step 5: Check if point fits existing level, else create new.
        """
        best_idx = -1
        min_dist = float('inf')
        
        # Find best matching level
        for i, level in enumerate(self.levels):
            sigma = self._get_sigma(level)
            r = max(self.std_fac* sigma, self.atr_fac* atr)
            
            dist = abs(price - level['M'])
            if dist <= r:
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        
        if best_idx != -1:
            self._update_level_state(self.levels[best_idx], price, volume)
        else:
            # Initialize brand new SR level
            self.levels.append({'V': volume, 'M': price, 'S': 0.0})

    def get_levels(self):
        """Returns the current list of SR levels."""
        return pd.DataFrame(self.levels)

    def fit(self, df: pd.DataFrame):
        """
        Processes the DataFrame day-by-day to detect SR levels.
        Expects columns: 'open', 'high', 'low', 'close', 'volume'.
        """
        if self.print_para:
            print("SR Parameters:")
            print(SR_PARAMS)
        # Ensure lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # Pre-calculate indicators
        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = df['atr'].bfill()
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=self.bb_length, std=self.bb_std, ddof=0, talib=False)
        
        if bb is None:
            return self.get_levels()

        # Dynamically find column names since pandas_ta appends parameters to them
        bbu_col = next(c for c in bb.columns if c.startswith("BBU"))
        bbl_col = next(c for c in bb.columns if c.startswith("BBL"))
        
        df = pd.concat([df, bb], axis=1)

        # Start loop
        # We need history for N (lookback) and BB (usually 20)
        start_idx = max(self.N * 2, self.bb_length)
        
        for t in range(start_idx, len(df)):
            # Current Data
            row_t = df.iloc[t]
            p_close = row_t['close']
            p_high = row_t['high']
            p_low = row_t['low']
            vol_t = row_t['volume']
            atr_t = row_t['atr']
            
            # --- Step 1: Check Pending Touches ---
            touched_indices = set()
            
            for i, level in enumerate(self.levels):
                sigma = self._get_sigma(level)
                r = max(self.std_fac* sigma, self.atr_fac* atr_t)
                
                # Check Close, High, Low priority
                if abs(p_close - level['M']) <= r:
                    self._update_level_state(level, p_close, vol_t * self.p)
                    touched_indices.add(i)
                elif abs(p_high - level['M']) <= r:
                    self._update_level_state(level, p_high, vol_t * self.p)
                    touched_indices.add(i)
                elif abs(p_low - level['M']) <= r:
                    self._update_level_state(level, p_low, vol_t * self.p)
                    touched_indices.add(i)
            
            if touched_indices:
                self.pending_touches.add(t)

            # --- Step 2: Extrema Detection at t-N ---
            k = t - self.N
            #print(df.index[k])
            # Window: [t-2N, t] -> length 2N+1. Index N in this window corresponds to k.
            window_start = t - 2 * self.N
            window_end = t 
            
            subset_high = df['high'].iloc[window_start : window_end + 1].values
            subset_low = df['low'].iloc[window_start : window_end + 1].values
            
            is_max = (subset_high[self.N] == np.max(subset_high))
            is_min = (subset_low[self.N] == np.min(subset_low))
            #apply volume filter
            if self.vol_filter:
                window_start_v = k-self.Nvol
                window_end_v = k+self.Nvol
                subset_volume = df['volume'].iloc[window_start_v : window_end_v + 1].values
                is_vol_max = (subset_volume[self.Nvol] == np.max(subset_volume))
            else: 
                is_vol_max = True
                
                
            #print(is_max, is_min)
            # --- Step 3: Bollinger Band Confirmation ---
            row_k = df.iloc[k]
            bbu_k = row_k[bbu_col]
            bbl_k = row_k[bbl_col]
            #print(row_k)
            confirmed_price = None
            
            if is_max and row_k['high'] > bbu_k:
                confirmed_price = row_k['high']
            elif is_min and row_k['low'] < bbl_k:
                confirmed_price = row_k['low']
            # --- Step 4: Re-evaluate and Update ---
            if (confirmed_price is not None) and is_vol_max:
                vol_k = row_k['volume']
                atr_k = row_k['atr']
                
                weight_factor = 1.0
                if k in self.pending_touches:
                    self.pending_touches.remove(k)
                    weight_factor = 1.0 - self.p
                
                # Process the update (Step 5 logic inside)
                self._process_point(confirmed_price, vol_k * weight_factor, atr_k)
        
        return self.get_levels()