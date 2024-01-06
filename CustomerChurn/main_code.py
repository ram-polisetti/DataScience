#import necessary libraries

import pandas as pd

from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


telco_data = pd.read_csv('/Users/ramcharansatyasaitejapolisetti/Documents/Winter2024/DataScience/CustomerChurn/data/Telco Customer Churn.csv')

telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')
telco_data.drop(columns='customerID', inplace=True)
telco_data['PaymentMethod'] = telco_data['PaymentMethod'].str.replace(' (automatic)', '', regex=False)

# function that computes the mutual information score between a categorical series and the column Churn
def compute_mutual_information(categorical_series):
    return mutual_info_score(categorical_series, telco_data['Churn'])

# select categorical variables excluding the response variable
categorical_variables = telco_data.select_dtypes(include='object').drop('Churn', axis=1)

# compute the mutual information score between each categorical variable and the target
feature_importance = categorical_variables.apply(compute_mutual_information).sort_values(ascending=False)

# visualize feature importance
print(feature_importance)


telco_data_tranformed = telco_data.copy()

label_encoding_columns = ['gender', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'Churn']

# encode categorical binary features using label encoding
for column in label_encoding_columns:
    if column == 'gender':
        telco_data_tranformed[column] = telco_data_tranformed[column].map({'Female': 1, 'Male': 0})
    else: 
        telco_data_tranformed[column] = telco_data_tranformed[column].map({'Yes': 1, 'No': 0}) 

# one-hot encoding (categorical variables with more than two levels)
one_hot_encoding_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                            'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract', 'PaymentMethod']

# encode categorical variables with more than two levels using one-hot encoding
telco_data_tranformed = pd.get_dummies(telco_data_tranformed, columns = one_hot_encoding_columns)

# min-max normalization (numeric variables)
min_max_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

# scale numerical variables using min max scaler
for column in min_max_columns:
        # minimum value of the column
        min_column = telco_data_tranformed[column].min()
        # maximum value of the column
        max_column = telco_data_tranformed[column].max()
        # min max scaler
        telco_data_tranformed[column] = (telco_data_tranformed[column] - min_column) / (max_column - min_column)   

telco_data_tranformed.dropna(inplace=True)

X = telco_data_tranformed.drop(columns='Churn')
y = telco_data_tranformed.loc[:, 'Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=40, shuffle=True)



# define the parameter grid
grid_parameters = {'n_estimators': [80, 90, 100, 110, 115, 120],
                   'max_depth': [3, 4, 5, 6],
                   'max_features': [None, 1.0, 'sqrt', 'log2'], 
                   'min_samples_split': [2, 3, 4, 5]}


# define the RandomizedSearchCV class for trying different parameter combinations
random_search = RandomizedSearchCV(estimator=GradientBoostingClassifier(),
                                   param_distributions=grid_parameters,
                                   cv=5,
                                   n_iter=150,
                                   n_jobs=-1)

# fitting the model for random search 
random_search.fit(X_train, y_train)

# print best parameter after tuning
print(random_search.best_params_)

# make the predictions
random_search_predictions = random_search.predict(X_test)

# construct the confusion matrix
confusion_matrix = confusion_matrix(y_test, random_search_predictions)


print(classification_report(y_test, random_search_predictions))