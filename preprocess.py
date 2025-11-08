import pandas as pd

def preprocess_data(train_path, test_path, preprocessed_train_path, preprocessed_test_path):
    """
    Loads, preprocesses, and saves the training and test data.
    """
    # Load the data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate date_id and target variable
    train_date_ids = train_df['date_id']
    test_date_ids = test_df['date_id']
    y_train = train_df['market_forward_excess_returns']

    # Drop unnecessary columns
    train_df = train_df.drop(columns=['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns'])
    test_df = test_df.drop(columns=['date_id', 'is_scored', 'lagged_forward_returns', 'lagged_risk_free_rate', 'lagged_market_forward_excess_returns'])

    # Fill missing values with the mean
    for col in train_df.columns:
        if train_df[col].isnull().any():
            mean_val = train_df[col].mean()
            train_df[col] = train_df[col].fillna(mean_val)
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(mean_val)

    # Save the preprocessed data
    train_df['date_id'] = train_date_ids
    train_df['market_forward_excess_returns'] = y_train
    test_df['date_id'] = test_date_ids

    train_df.to_csv(preprocessed_train_path, index=False)
    test_df.to_csv(preprocessed_test_path, index=False)

if __name__ == '__main__':
    preprocess_data('train.csv', 'test.csv', 'preprocessed_train.csv', 'preprocessed_test.csv')
