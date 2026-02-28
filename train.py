import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

def train():
    # Loading Data
    df = pd.read_csv('tokyo_weather.csv', parse_dates=['date'])
    df = df.sort_values('date')

    # Feature Engineering
    df['month'] = df['date'].dt.month
    df['lag1'] = df['max_temp'].shift(1)
    df['lag2'] = df['max_temp'].shift(2)
    df['lag3'] = df['max_temp'].shift(3)
    
    # Drop missing values (first 3 days)
    df_train = df.dropna()

    # Training
    features = ['month', 'lag1', 'lag2', 'lag3']
    X = df_train[features]
    y = df_train['max_temp']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Drop missing values (first 3 days)
    joblib.dump(model, 'weather_model.joblib')
    joblib.dump(features, 'model_columns.joblib')
    print("Successfully serialized trained model to weather_model.joblib")
    print("Feature columns serialized to model_columns.joblib")

if __name__ == "__main__":
    train()