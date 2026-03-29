# feature_engineer.py
import pandas as pd
import pandas_ta as ta
from . import indicators # Dynamically loads functions from this file

class FeatureEngineer:
    def __init__(self, features_config: list, pipeline_config: dict = None, target_tickers: list = None):
        self.features_config = features_config
        self.pipeline_config = pipeline_config if pipeline_config is not None else {}
        self.target_tickers = target_tickers

    def _resolve_params(self, params: dict) -> dict:
        """Resolves placeholder values in parameters using the main pipeline config."""
        resolved_params = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith('$$') and value.endswith('$$'):
                param_name = value.strip('$')
                if param_name in self.pipeline_config:
                    resolved_params[key] = self.pipeline_config[param_name]
                else:
                    raise ValueError(f"Placeholder '{value}' for param '{key}' not found in pipeline_config.")
            else:
                resolved_params[key] = value
        return resolved_params

    def apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reads config and applies standard pandas_ta or custom features dynamically."""
        df_featured = df.copy()

        for feature in self.features_config:
            name = feature['name']
            f_type = feature['type']
            params = feature.get('params', {}).copy()

            # Resolve any placeholder values (e.g., '$$forecast_horizon$$')
            params = self._resolve_params(params)

            if f_type == 'custom':
                func_name = feature['function']
                
                if self.target_tickers:
                    params['target_tickers'] = self.target_tickers
                
                if hasattr(indicators, func_name):
                    def _format_param(val):
                        if isinstance(val, pd.DataFrame): return f"<DataFrame shape={val.shape}>"
                        if isinstance(val, dict): return f"<dict keys={list(val.keys())}>"
                        if isinstance(val, list): return f"<list len={len(val)}>"
                        return val
                        
                    print_params = {k: _format_param(v) for k, v in params.items()}
                    print(f"Applying custom feature: {name} with params: {print_params}")
                    func = getattr(indicators, func_name) 
                    
                    # NEW: Pass the params into the function
                    result = func(df_featured, **params)
                    
                    if isinstance(name, list):
                        # If it returns a DataFrame with the correct number of columns, map by position
                        if isinstance(result, pd.DataFrame) and len(result.columns) == len(name):
                            for i, col_name in enumerate(name):
                                df_featured[col_name] = result.iloc[:, i]
                        else:
                            for col_name in name:
                                df_featured[col_name] = result[col_name]
                    else:
                        if isinstance(result, pd.DataFrame):
                            df_featured[name] = result.squeeze()
                        else:
                            df_featured[name] = result
                else:
                    raise ValueError(f"Function '{func_name}' not found in indicators.py")

            elif f_type == 'pandas_ta':
                print(f"Applying pandas_ta feature: {name}")
                
                CustomStrategy = ta.Strategy(
                    name=f"Dynamic_{name}",
                    ta=[{"kind": feature['kind'], **params}]
                )
                
                if self.target_tickers:
                    mask = df_featured['unique_id'].isin(self.target_tickers)
                    df_target = df_featured[mask].groupby('unique_id', group_keys=False).apply(
                        lambda x: self._apply_ta(x, CustomStrategy, rename_to=name),
                        include_groups=False
                    )
                    # Recombine with non-target rows (which remain unchanged)
                    df_featured = pd.concat([df_target, df_featured[~mask]]).sort_index()
                else:
                    df_featured = df_featured.groupby('unique_id', group_keys=False).apply(
                        lambda x: self._apply_ta(x, CustomStrategy, rename_to=name),
                        include_groups=False
                    )

        # Final Step: Drop reference tickers if targets were specified
        if self.target_tickers:
            df_featured = df_featured[df_featured['unique_id'].isin(self.target_tickers)].copy()

        return df_featured

    def _apply_ta(self, df_group, strategy, rename_to=None):
        before_cols = set(df_group.columns)
        df_group.ta.strategy(strategy)
        
        if rename_to:
            new_cols = list(set(df_group.columns) - before_cols)
            # Only rename if pandas_ta added exactly one column
            if len(new_cols) == 1:
                df_group.rename(columns={new_cols[0]: rename_to}, inplace=True)
        return df_group