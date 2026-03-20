import numpy as np
import pandas as pd
import pandas_ta as ta
from config import SR_PARAMS
from .vol_indicator import integrate_vol
from visualize_daily import load_daily_data
from tqdm import tqdm
class SRLevelDetector:
    def __init__(self, intraday_df: pd.DataFrame, print_para=False):
        self.N = SR_PARAMS['window_size']
        self.p = SR_PARAMS['penalty_fac']
        self.std_fac= SR_PARAMS['std_fac']
        self.atr_fac= SR_PARAMS['atr_fac']
        self.atr_fac2 = SR_PARAMS.get('atr_fac2', 1.5)
        self.bb_length = SR_PARAMS['bb_length']
        self.bb_std = SR_PARAMS['bb_std']
        self.vol_filter = SR_PARAMS['vol_filter']
        self.vol_filter_std = SR_PARAMS.get('vol_filter_std', 1.0)
        self.Nvol = SR_PARAMS['vol_window']
        self.integrate_vol_atr_period = SR_PARAMS['integrate_vol_atr_period']
        self.integrate_vol_avg_fac = SR_PARAMS['integrate_vol_avg_fac']
        self.print_para = print_para
        self.intraday_df = intraday_df
        # List of SR levels. Each level is a dict: {'V', 'M', 'S'}
        # Pending touches: {time_index: [(level_index, 'high'/'low'), ...]}
        self.levels = [] 
        self.pending_touches = {}
        self.daily_vols = {}

    def _get_sigma(self, level):
        if level['V'] == 0:
            return 0.0
        return np.sqrt(level['S'] / level['V'])

    def _get_radius(self, level, atr):
        sigma = self._get_sigma(level)
        if level.get('count', 0) <= 1:
            return self.atr_fac2 * atr
        return max(self.std_fac * sigma, self.atr_fac * atr)

    def _update_level_state(self, level, p_new, v_new, add_count=True):
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
        if add_count:
            level['count'] = level.get('count', 0) + 1

    def _process_point(self, price, volume, atr):
        """
        Step 5: Check if point fits existing level, else create new.
        """
        best_idx = -1
        min_dist = float('inf')
        
        # Find best matching level
        for i, level in enumerate(self.levels):
            r = self._get_radius(level, atr)
            dist = abs(price - level['M'])
            if dist <= r:
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        
        if best_idx != -1:
            self._update_level_state(self.levels[best_idx], price, volume)
        else:
            # Initialize brand new SR level
            self.levels.append({'V': volume, 'M': price, 'S': 0.0, 'count': 1})

    def get_levels(self):
        """Returns the current list of SR levels."""
        return pd.DataFrame(self.levels)

    def process_day(self, t: int, df: pd.DataFrame):
        """
        Processes a single day at index t to detect SR levels.
        Expects columns: 'open', 'high', 'low', 'close', 'volume', 'atr'.
        """
        # Current Data
        row_t = df.iloc[t]
        p_open = row_t['open']
        p_close = row_t['close']
        p_high = row_t['high']
        p_low = row_t['low']
        atr_t = row_t['atr']
        
        # Calculate Dynamic Multipliers
        R = p_high - p_low
        if R == 0:
            M_high = 0.01
            M_low = 0.01
        else:
            W_L_high = p_high - max(p_open, p_close)
            W_L_low = min(p_open, p_close) - p_low
            M_high = max(0.01, W_L_high / R)
            M_low = max(0.01, W_L_low / R)

        # --- Step 1: Check Pending Touches ---
        touches_for_t = []
        updated_levels_this_step = set()

        vol_high_touch = None
        # Check for high touches
        for i, level in enumerate(self.levels):
            r = self._get_radius(level, atr_t)
            if abs(p_high - level['M']) <= r:
                # Calculate high touch volume lazily only if a touch occurs
                if vol_high_touch is None:
                    date_t = df.index[t].date()
                    intraday_for_t = load_daily_data(self.intraday_df, str(date_t))
                    if not intraday_for_t.empty:
                        vol_high_touch = abs(integrate_vol(intraday_for_t, p_high, self.integrate_vol_atr_period, self.integrate_vol_avg_fac))
                    else:
                        vol_high_touch = 0.0
                    if t not in self.daily_vols:
                        self.daily_vols[t] = {}
                    self.daily_vols[t]['high'] = vol_high_touch
                    
                if i not in updated_levels_this_step:
                    self._update_level_state(level, p_high, vol_high_touch * M_high)
                    updated_levels_this_step.add(i)
                touches_for_t.append((i, 'high', M_high))

        vol_low_touch = None
        # Check for low touches
        for i, level in enumerate(self.levels):
            r = self._get_radius(level, atr_t)
            if abs(p_low - level['M']) <= r:
                # Calculate low touch volume lazily only if a touch occurs
                if vol_low_touch is None:
                    date_t = df.index[t].date()
                    intraday_for_t = load_daily_data(self.intraday_df, str(date_t))
                    if not intraday_for_t.empty:
                        vol_low_touch = abs(integrate_vol(intraday_for_t, p_low, self.integrate_vol_atr_period, self.integrate_vol_avg_fac))
                    else:
                        vol_low_touch = 0.0
                    if t not in self.daily_vols:
                        self.daily_vols[t] = {}
                    self.daily_vols[t]['low'] = vol_low_touch

                if i not in updated_levels_this_step:
                    self._update_level_state(level, p_low, vol_low_touch * M_low)
                    updated_levels_this_step.add(i)
                touches_for_t.append((i, 'low', M_low))

        if touches_for_t:
            self.pending_touches[t] = touches_for_t

        # --- Step 2: Extrema Detection at t-N ---
        k = t - self.N
        # Window: [t-2N, t] -> length 2N+1. Index N in this window corresponds to k.
        window_start = t - 2 * self.N
        window_end = t 
        
        subset_high = df['high'].iloc[window_start : window_end + 1].values
        subset_low = df['low'].iloc[window_start : window_end + 1].values
        
        is_max = (subset_high[self.N] == np.max(subset_high))
        is_min = (subset_low[self.N] == np.min(subset_low))
        
        #apply volume filter
        if self.vol_filter is True or self.vol_filter == 'max':
            window_start_v = k-self.Nvol
            window_end_v = k+self.Nvol
            subset_volume = df['volume'].iloc[window_start_v : window_end_v + 1].values
            is_vol_max = (subset_volume[self.Nvol] == np.max(subset_volume))
        elif self.vol_filter == 'ma':
            window_start_v = k-self.Nvol
            window_end_v = k+self.Nvol
            subset_volume = df['volume'].iloc[window_start_v : window_end_v + 1].values
            is_vol_max = (subset_volume[self.Nvol] > (np.mean(subset_volume) + self.vol_filter_std * np.std(subset_volume)))
        else: 
            is_vol_max = True
            
        # --- Step 3: Confirmation ---
        row_k = df.iloc[k]
        confirmed_price = None
        
        if is_max:
            confirmed_price = row_k['high']
        elif is_min:
            confirmed_price = row_k['low']
            
        # --- Step 4: Re-evaluate and Update ---
        if (confirmed_price is not None) and is_vol_max:
            touch_type_of_turning_point = 'high' if is_max else 'low'
            
            # Check if volume was already calculated during a touch
            if k in self.daily_vols and touch_type_of_turning_point in self.daily_vols[k]:
                vol_k = self.daily_vols[k][touch_type_of_turning_point]
            else:
                date_k = df.index[k].date()
                intraday_for_k = load_daily_data(self.intraday_df, str(date_k))
                
                if not intraday_for_k.empty:
                    vol_k = abs(integrate_vol(intraday_for_k, confirmed_price, self.integrate_vol_atr_period, self.integrate_vol_avg_fac))
                else:
                    vol_k = 0.0
                
                if k not in self.daily_vols:
                    self.daily_vols[k] = {}
                self.daily_vols[k][touch_type_of_turning_point] = vol_k
            
            # proceed with update using integrated volume.
            atr_k = row_k['atr']
            
            is_processed_as_touch_recovery = False
            if k in self.pending_touches:
                touches_at_k = self.pending_touches.pop(k) 
                
                for level_idx, touch_type, touch_M in touches_at_k:
                    if touch_type == touch_type_of_turning_point:
                        level_to_update = self.levels[level_idx]
                        self._update_level_state(level_to_update, confirmed_price, vol_k * (1.0 - touch_M), add_count=False)
                        is_processed_as_touch_recovery = True
                        break
                
            if not is_processed_as_touch_recovery:
                self._process_point(confirmed_price, vol_k, atr_k)