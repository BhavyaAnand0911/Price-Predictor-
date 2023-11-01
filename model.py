import inline
import matplotlib
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

housing = pd.read_csv("./data.csv")

# housing.head() # this will display the first 5 rows of the data in table format
# housing.info() # this will give the information about the data such as total number of entries

# housing['CHAS'].value_counts() # will give the count of all the possible values of the given feature

print(f"This is the first five rows of the Housing data: \n {housing.describe()}")

# plt.hist(housing["CHAS"],bins=50)
# plt.show()

# Train Test Splitting

# This is for learning how the function works internally
# import numpy as np

# def split_train_test(data,test_ratio):
#     np.random.seed(42) #

#     shuffled=np.random.permutation(len(data))
#     test_set_size=int(len(data)* test_ratio)
#     test_indices=shuffled[:test_set_size]
#     train_indices=shuffled[test_set_size:]
#     return data.iloc[test_indices], data.iloc[train_indices]
# test_set,train_set=split_train_test(housing,0.2)

# print(f"Rows in training set: {len(train_set)}\n Rows in test set: {len(test_set)}")


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print(f"Rows in training set: {len(train_set)}\n Rows in test set: {len(test_set)}")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index,test_index in split.split(housing, housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]

housing=strat_train_set.copy() # made housing the training set

# Looking for correlations
corr_matrix=housing.corr()
corr_matrix["MEDV"].sort_values(ascending=False) # calculating the correlation of money w.r.t all other features

# pearson correlation coefficient 1 means strong positive correlation
# when this value increases the price of the property will increase

attributes=['MEDV','RM','ZN','LSTAT']

plt.scatter(housing['RM'], housing['MEDV'], alpha=0.9)
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.title('Scatter Plot of RM vs. MEDV')

plt.show()

# Trying out attribute combinations
housing['TAXRM']=housing['TAX']/housing['RM']
# housing['TAXRM']

corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

plt.scatter(housing['TAXRM'], housing['MEDV'], alpha=0.9)
plt.xlabel('TAXRM')
plt.ylabel('MEDV')
plt.title('Scatter Plot of TAXRM vs. MEDV')

plt.show()

housing=strat_train_set.drop("MEDV", axis=1)
housing_lables= strat_train_set["MEDV"].copy()

# Handling Missing Attributes
# There are 3 ways to handle missing attributes:
# 1. Get rid of the missing data points # newHousing = housing.dropna(subset["RM"])
# 2. Get rid of the whole attribute # housing.drop("RM",axis=1).shape
# 3. Set them to some value (0,mean or median) # housing["RM"].fillna(housing["RM"].median())

imputer = SimpleImputer(strategy="median")
imputer.fit(housing)

imputer_stats = imputer.statistics_
print(imputer_stats)

X=imputer.transform(housing)
housing_corrected=pd.DataFrame(X,columns=housing.columns)
print(housing_corrected.describe())

# Scikit-learn Design
# Primarily this has three types of objects
# 1. Estimators (imputer) this has fit method (fits the data in the dataset and calculates internal parameters) & transform method.
# 2. Transformers (takes input and returns output based on the learnings of the fit() this has a funtion fit_transform() )
# 3. Predictors (LinearRegression model is an example of predictor)

# Creating Pipeline

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

housing_pipe_corrected=my_pipeline.fit_transform(housing) # housing_pipe_corrected

#model=LinearRegression() because other models gave more accurate predictions
model=DecisionTreeRegressor()
model.fit(housing_pipe_corrected,housing_lables)

# Evaluating the model

housing_predictions=model.predict(housing_pipe_corrected)

lin_rmse= np.sqrt(mean_squared_error(housing_lables, housing_predictions))
print(lin_rmse)
# Cross Validation of the model

scores= cross_val_score(model,housing_pipe_corrected, housing_lables,scoring="neg_mean_squared_error", cv=10)
rmse_scores=np.sqrt(-scores)
print(rmse_scores)

def print_scores(scores):
    print("Scores: ",scores)
    print("Mean: ",scores.mean())
    print("Std Deviation: ",scores.std())

print_scores(rmse_scores)

X_test= strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_prediction=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_prediction)
final_rmse=np.sqrt(final_mse)

print(f"This final prediction results: {final_rmse}" )






