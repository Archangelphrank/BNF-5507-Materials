# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Impute Missing Values


def impute_missing_values(data, strategy='mean'):
    """Fill missing values in the dataset while handling non-numeric columns."""
    numeric_cols = data.select_dtypes(include=['number'])  # Only numeric columns
    if strategy == 'mean':
        data[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())
    elif strategy == 'median':
        data[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.median())
    elif strategy == 'mode':
        data[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mode().iloc[0])
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'mode'.")
    return data

# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    # TODO: Remove duplicate rows
    return data.drop_duplicates()

    pass

# 3. Normalize Numerical Data
def normalize_data(data, method='minmax'):
    """Apply normalization to numerical features."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    numeric_cols = data.select_dtypes(include=['number'])  # Select numeric columns

    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid method. Choose 'minmax' or 'standard'.")

    data[numeric_cols.columns] = scaler.fit_transform(numeric_cols)
    return data.copy()  # Ensure original DataFrame isn't modified directly

# 4. Remove Redundant Features   


def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns based on correlation threshold."""
    numeric_data = data.select_dtypes(include=['number'])  # Select numeric columns
    corr_matrix = numeric_data.corr().abs()

    # Select the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation higher than the threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    # Drop redundant columns only from numeric data
    return data.drop(columns=to_drop)



    pass

# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]
    
   # Convert continuous target variable into discrete bins
    target = pd.cut(target, bins=3, labels=[0, 1, 2])  # Adjust bins based on your data distribution


    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None