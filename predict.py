import pandas as pd
import lightgbm as lgb

def predict(preprocessed_test_path, model_path, submission_path):
    """
    Generates predictions and creates a submission file.
    """
    # Load the preprocessed test data
    test_df = pd.read_csv(preprocessed_test_path)
    date_ids = test_df['date_id']
    X_test = test_df.drop(columns=['date_id'])

    # Load the trained model
    model = lgb.Booster(model_file=model_path)

    # Generate predictions
    predictions = model.predict(X_test)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({'date_id': date_ids, 'prediction': predictions})

    # The competition allows values between 0 and 2.
    # We'll scale our predictions to be within a reasonable range, e.g., 0.5 to 1.5
    # A simple scaling can be done, but a more sophisticated approach might be needed.
    min_pred = submission_df['prediction'].min()
    max_pred = submission_df['prediction'].max()
    submission_df['prediction'] = 0.5 + (submission_df['prediction'] - min_pred) / (max_pred - min_pred)

    # Format the submission file
    submission_df.to_csv(submission_path, index=False)

if __name__ == '__main__':
    predict('preprocessed_test.csv', 'model.txt', 'submission.csv')
