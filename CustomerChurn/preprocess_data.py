import pandas as pd

def read_data(data_path):
    telco_data = pd.read_csv(data_path)
    return telco_data

def process_data(telco_data):
    telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')
    telco_data.drop(columns='customerID', inplace=True)
    telco_data['PaymentMethod'] = telco_data['PaymentMethod'].str.replace(' (automatic)', '', regex=False)
    return telco_data
    

def normalize_data(telco_data):
    # telco_data_tranformed = telco_data.copy()
    label_encoding_columns = ['gender', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'Churn']

    # encode categorical binary features using label encoding
    for column in label_encoding_columns:
        if column == 'gender':
            telco_data[column] = telco_data[column].map({'Female': 1, 'Male': 0})
        else: 
            telco_data[column] = telco_data[column].map({'Yes': 1, 'No': 0}) 

    # one-hot encoding (categorical variables with more than two levels)
    one_hot_encoding_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract', 'PaymentMethod']

    # encode categorical variables with more than two levels using one-hot encoding
    telco_data = pd.get_dummies(telco_data, columns = one_hot_encoding_columns)

    # min-max normalization (numeric variables)
    min_max_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # scale numerical variables using min max scaler
    for column in min_max_columns:
            # minimum value of the column
            min_column = telco_data[column].min()
            # maximum value of the column
            max_column = telco_data[column].max()
            # min max scaler
            telco_data[column] = (telco_data[column] - min_column) / (max_column - min_column)   

    telco_data.dropna(inplace=True)
    return telco_data

if __name__ == "__main__":

    DATA_PATH = "./data/"
    DATA_FILE = "Telco Customer Churn.csv"

    FINAL_FILE = "cleaned_data.csv"

    # 1. Read data
    telco_data = read_data(DATA_PATH+DATA_FILE)

    # 2. Preprocess the data
    processed_data = process_data(telco_data)

    # 3. Clean the data
    normalize_dataframe = normalize_data(processed_data)

    # 4. Save the cleaned data for training
    normalize_dataframe.to_csv(DATA_PATH+FINAL_FILE, index=False)