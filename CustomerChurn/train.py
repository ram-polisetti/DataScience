import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib

if __name__ == "__main__":

    # 1. Load the cleaned data
    DATA_PATH = "./data/"
    FILE_NAME = "cleaned_data.csv"

    telco_data = pd.read_csv(DATA_PATH+FILE_NAME)

    X = telco_data.drop(columns='Churn')
    y = telco_data.loc[:, 'Churn']

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

    # make the predictions
    random_search_predictions = random_search.predict(X_test)


    MODEL_FILE_NAME = "gradient_best_model.joblib"
    MODEL_FOLDER = "./models/"


    # Serialize the model
    joblib.dump(random_search, MODEL_FOLDER+MODEL_FILE_NAME)
