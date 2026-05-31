import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from features import engineer_features

def train_model():
    """
    Example training pipeline for the hackathon.
    """
    print("Loading data...")
    # NOTE: Uncomment and set your actual data path here
    # df = pd.read_csv('../data/raw/train.csv')
    
    # Example dummy data to make the script runnable out-of-the-box
    df = pd.DataFrame({
        'datetime': pd.date_range(start='1/1/2026', periods=100, freq='H'),
        'temperature': [20]*100,
        'traffic_volume': [100 + i for i in range(100)]
    })

    print("Engineering features...")
    df = engineer_features(df)
    
    # Define features and target
    X = df.drop(['traffic_volume', 'datetime'], axis=1, errors='ignore')
    y = df['traffic_volume']
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Validating model...")
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    print(f"Validation MSE: {mse:.4f}")
    
    return model

if __name__ == "__main__":
    train_model()
