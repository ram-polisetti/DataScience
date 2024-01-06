#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np 
import pandas as pd


# In[2]:


telco_data = pd.read_csv('/kaggle/input/wa-fnusec-telcocustomerchurn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
telco_data.head()


# In[3]:


telco_data.shape


# In[4]:


telco_data.columns


# In[5]:


telco_data.info()


# In[6]:


telco_data.isnull().sum()


# Dataset comprises of 7043 rows and 21 attributes. There are no null values in the dataset but the column TotalCharges is wrongly detected as object.

# In[7]:


telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')


# In[8]:


telco_data.info()


# In[9]:


telco_data.isnull().sum()


# TotalCharges have 11 Missing values. 

# In[10]:


telco_data[telco_data.TotalCharges.isnull()]


# All these observations have zero tenure eventhough the monthly charges are not null. These observations look contradictory.

# In[11]:


# lets look at unique values for each column
for column in telco_data.columns:
    print('Column: {} - Unique Values: {}'.format(column, telco_data[column].unique()))


# These columns can be categorized into 3 categories
# 
# 1. Demographic Information
#     - gender
#     - SeniorCitizen
#     - Partner
#     - Dependents
# 2. Customer Account Information
#     - tenure
#     - Contract
#     - PaperlessBilling
#     - PaymentMethod
#     - MonthlyCharges
#     - TotalCharges
# 3. Services Information
#     - PhoneService
#     - MultipleLines
#     - InternetServices
#     - OnlineSecurity
#     - OnlineBackup
#     - DeviceProtection
#     - TechSupport
#     - StreamingTV
#     - StreamingMovies

# In[12]:


#Dropping customerID 
telco_data.drop(columns='customerID', inplace=True)


# Column: PaymentMethod - Unique Values: ['Electronic check' 'Mailed check' 'Bank transfer (automatic)'
#  'Credit card (automatic)']
#  the string '(automatic)' is redundant. 

# In[13]:


telco_data['PaymentMethod'] = telco_data['PaymentMethod'].str.replace(' (automatic)', '', regex=False)


# In[14]:


telco_data.PaymentMethod.unique()


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[16]:


fig, ax = plt.subplots(figsize=(10, 6))
churn_response_proportion = telco_data['Churn'].value_counts(normalize=True)
sns.barplot(x=churn_response_proportion.index, y=churn_response_proportion, palette=['salmon', 'lightgreen'])
plt.title("Proportion of observations of the response variable")
plt.xlabel('Churn')
plt.ylabel('Proportion of observations')
plt.show()


# the dataset is imbalanced. lets take a look at normalized stacked bar plots for each variable to understand the influence on Churn column

# Lets start with demographic information
# 
# - gender
# - SeniorCitizen
# - Partner
# - Dependents

# In[17]:


def percentage_stacked_plot(columns_to_plot, super_title):
    
    '''
    Prints a 100% stacked plot of the response variable for independent variable of the list columns_to_plot.
            Parameters:
                    columns_to_plot (list of string): Names of the variables to plot
                    super_title (string): Super title of the visualization
            Returns:
                    None
    '''
    
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)

    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows)) 
    fig.suptitle(super_title, fontsize=22,  y=.95)
 

    # loop to each column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        # calculate the percentage of observations of the response variable for each group of the independent variable
        # 100% stacked bar plot
        prop_by_independent = pd.crosstab(telco_data[column], telco_data['Churn']).apply(lambda x: x/x.sum()*100, axis=1)

        prop_by_independent.plot(kind='bar', ax=ax, stacked=True,
                                 rot=0, color=['springgreen','salmon'])

        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)

        # set title and labels
        ax.set_title('Proportion of observations by ' + column,
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)


# In[18]:


# demographic column names
demographic_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

# stacked plot of demographic columns
percentage_stacked_plot(demographic_columns, 'Demographic Information')


# 
# - The churn rate for senior citizens is nearly twice as high as the churn rate for young citizens. 
# 
# - Gender does not appear to be a major predictive factor for churn. The churn rate is similar for both male and female customers. 
# 
# - Customers who have a partner are less likely to churn compared to customers without a partner.

# Lets do the same anlysis on Customer Account Information
# - contract
# - papelessbilling
# - paymentmethod

# In[19]:


# customer account column names
account_columns = ['Contract', 'PaperlessBilling', 'PaymentMethod']

# stacked plot of customer account columns
percentage_stacked_plot(account_columns, 'Customer Account Information')


# 
# - Customers with month-to-month contracts have higher churn rates than customers with yearly contracts.
# 
# - Customers who pay by electronic check are more likely to churn compared to those using other payment methods. 
# 
# - Customers who are subscribed to paperless billing have a higher churn rate than those who receive paper bills.

# Lets do the analysis on Customer Account Information
# - tenure
# - monthlycharges
# - totalcharges
# 
# For all numeric attributes, we see differing distributions between customers who churned and those who did not. This suggests that each of these attributes will be useful in predicting whether a given customer is likely to churn. Specifically, the distribution of values is different for churning versus non-churning customers across all the numeric attributes. This indicates these features contain signal that can help distinguish between the two classes.

# In[20]:


def histogram_plots(columns_to_plot, super_title):
    '''
     Prints a histogram for each independent variable of the list columns_to_plot.
            Parameters:
                    columns_to_plot (list of string): Names of the variables to plot
                    super_title (string): Super title of the visualization
            Returns:
                    None
    '''
    # set number of rows and number of columns
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)

    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows)) 
    fig.suptitle(super_title, fontsize=22,  y=.95)
 

    # loop to each demographic column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        # histograms for each class (normalized histogram)
        telco_data[telco_data['Churn']=='No'][column].plot(kind='hist', ax=ax, density=True, 
                                                       alpha=0.5, color='springgreen', label='No')
        telco_data[telco_data['Churn']=='Yes'][column].plot(kind='hist', ax=ax, density=True,
                                                        alpha=0.5, color='salmon', label='Yes')
        
        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)

        # set title and labels
        ax.set_title('Distribution of ' + column + ' by churn',
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)
            
# customer account column names
account_columns_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
# histogram of costumer account columns 
histogram_plots(account_columns_numeric, 'Customer Account Information')


# - There is a trend of higher churn rates when monthly charges are high.
# - Customers who have been with the company for a short tenure (new customers) tend to have higher churn rates.
# - Customers with high total charges accrued over time are less likely to churn.

# Lets do the analysis for services
# - PhoneService
# - MultipleLines
# - InternetServices
# - OnlineSecurity
# - OnlineBackup
# - DeviceProtection
# - TechSupport
# - StreamingTV
# - StreamingMovies

# In[21]:


# services column names
services_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# stacked plot of services columns
percentage_stacked_plot(services_columns, 'Services Information')


# 
# 
# - Phone service attributes like having phone service or multiple lines do not appear to be major predictive factors for churn. The churn percentage is nearly equal across the different classes of these variables.
# 
# - Customers who have online security enabled churn less than those without online security.
# 
# - Customers without tech support tend to have higher churn rates compared to those who have tech support.

# Lets understand the mutual dependency between independent categorical variables and the response variable

# In[22]:


from sklearn.metrics import mutual_info_score

# function that computes the mutual information score between a categorical series and the column Churn
def compute_mutual_information(categorical_series):
    return mutual_info_score(categorical_series, telco_data['Churn'])

# select categorical variables excluding the response variable
categorical_variables = telco_data.select_dtypes(include='object').drop('Churn', axis=1)

# compute the mutual information score between each categorical variable and the target
feature_importance = categorical_variables.apply(compute_mutual_information).sort_values(ascending=False)

# visualize feature importance
print(feature_importance)


# In[23]:


# Plotting the horizontal bar graph
plt.figure(figsize=(8, 6))
feature_importance_sorted = feature_importance.sort_values() 
plt.barh(feature_importance_sorted.index, feature_importance_sorted.values, color='skyblue')
plt.xlabel('Mutual Information Score')
plt.title('Mutual Information Scores for Categorical Variables')
plt.tight_layout()
plt.show()


# The variables gender, PhoneService, and MultipleLines have mutual information scores very close to 0 with respect to the target churn variable. This indicates these features do not have a strong relationship with churn. This aligns with our previous conclusions from visualizing the data

# In[24]:


telco_data_tranformed = telco_data.copy()

# label encoding (binary variables)
label_encoding_columns = ['gender', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'Churn']

# encode categorical binary features using label encoding
for column in label_encoding_columns:
    if column == 'gender':
        telco_data_tranformed[column] = telco_data_tranformed[column].map({'Female': 1, 'Male': 0})
    else: 
        telco_data_tranformed[column] = telco_data_tranformed[column].map({'Yes': 1, 'No': 0}) 


# In[25]:


# one-hot encoding (categorical variables with more than two levels)
one_hot_encoding_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                            'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract', 'PaymentMethod']

# encode categorical variables with more than two levels using one-hot encoding
telco_data_tranformed = pd.get_dummies(telco_data_tranformed, columns = one_hot_encoding_columns)


# In[26]:


telco_data_tranformed.head()


# In[27]:


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


# In[28]:


telco_data_tranformed.head()


# In[29]:


telco_data_tranformed.isnull().sum()


# In[30]:


telco_data_tranformed.dropna(inplace=True)
telco_data_tranformed.isnull().sum()


# In[31]:


X = telco_data_tranformed.drop(columns='Churn')
y = telco_data_tranformed.loc[:, 'Churn']
X.columns, y.name


# In[32]:


from sklearn.model_selection import train_test_split
# split the data in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=40, shuffle=True)


# In[33]:


from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[34]:


def create_models(seed=2):
    '''
    Create a list of machine learning models.
            Parameters:
                    seed (integer): random seed of the models
            Returns:
                    models (list): list containing the models
    '''

    models = []
    models.append(('dummy_classifier', DummyClassifier(random_state=seed, strategy='most_frequent')))
    models.append(('k_nearest_neighbors', KNeighborsClassifier()))
    models.append(('logistic_regression', LogisticRegression(random_state=seed)))
    models.append(('support_vector_machines', SVC(random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(random_state=seed)))
    models.append(('gradient_boosting', GradientBoostingClassifier(random_state=seed)))
    
    return models

# create a list with all the algorithms we are going to assess
models = create_models()


# In[35]:


from sklearn.metrics import accuracy_score
# test the accuracy of each model using default hyperparameters
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    # fit the model with the training data
    model.fit(X_train, y_train).predict(X_test)
    # make predictions with the testing data
    predictions = model.predict(X_test)
    # calculate accuracy 
    accuracy = accuracy_score(y_test, predictions)
    # append the model name and the accuracy to the lists
    results.append(accuracy)
    names.append(name)
    # print classifier accuracy
    print('Classifier: {}, Accuracy: {})'.format(name, accuracy))


# Gradient Boosting has higher accuracy than all other classifiers

# In[36]:


get_ipython().system('pip install numpy==1.22.0')


# In[37]:


from sklearn.model_selection import RandomizedSearchCV

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


# In[38]:


from sklearn.metrics import confusion_matrix

# make the predictions
random_search_predictions = random_search.predict(X_test)

# construct the confusion matrix
confusion_matrix = confusion_matrix(y_test, random_search_predictions)

# visualize the confusion matrix
confusion_matrix


# In[39]:


from sklearn.metrics import classification_report
print(classification_report(y_test, random_search_predictions))

