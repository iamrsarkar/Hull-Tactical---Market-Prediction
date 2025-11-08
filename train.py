import pandas as pd
import lightgbm as lgb

def train_model(preprocessed_train_path, model_path):
    """
    Trains a LightGBM model and saves it.
    """
    # Load the preprocessed training data
    train_df = pd.read_csv(preprocessed_train_path)

    # Separate features and target
    X_train = train_df.drop(columns=['date_id', 'market_forward_excess_returns'])
    y_train = train_df['market_forward_excess_returns']

    # Initialize and train the LightGBM model
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    model.booster_.save_model(model_path)

if __name__ == '__main__':
    train_model('preprocessed_train.csv', 'model.txt')
