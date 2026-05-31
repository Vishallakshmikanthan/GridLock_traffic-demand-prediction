import pandas as pd
import numpy as np

def create_datetime_features(df, datetime_col='datetime'):
    """
    Extract common time-based features from a datetime column,
    which is highly critical for traffic demand prediction.
    """
    df_feat = df.copy()
    
    if datetime_col in df_feat.columns:
        df_feat[datetime_col] = pd.to_datetime(df_feat[datetime_col])
        df_feat['hour'] = df_feat[datetime_col].dt.hour
        df_feat['dayofweek'] = df_feat[datetime_col].dt.dayofweek
        df_feat['month'] = df_feat[datetime_col].dt.month
        df_feat['is_weekend'] = df_feat['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        
    return df_feat

def engineer_features(df):
    """
    Main function to run the full feature engineering pipeline.
    """
    df = create_datetime_features(df)
    # Add other domain-specific features (weather, holidays, lag features, etc.)
    return df
