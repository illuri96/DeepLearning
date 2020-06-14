from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import missingno as msno
from sklearn.preprocessing import LabelEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())

#output different data types 
print(train.dtypes)


# string label to categorical values
for i in range(train.shape[1]):
    if train.iloc[:, i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(train.iloc[:, i].values) + list(test.iloc[:, i].values))
        train.iloc[:, i] = lbl.transform(list(train.iloc[:, i].values))
        test.iloc[:, i] = lbl.transform(list(test.iloc[:, i].values))

print(train['SaleCondition'].unique())



# Which columns have nan?
print('training data+++++++++++++++++++++')
for i in np.arange(train.shape[1]):
    n = train.iloc[:, i].isnull().sum()
    if n > 0:
        print(list(train.columns.values)[i] + ': ' + str(n) + ' nans')

print('testing data++++++++++++++++++++++ ')
for i in np.arange(test.shape[1]):
    n = test.iloc[:, i].isnull().sum()
    if n > 0:
        print(list(test.columns.values)[i] + ': ' + str(n) + ' nans')

# split data for training
y_train = train['SalePrice']
X_train = train.drop(['Id', 'SalePrice'], axis=1)
X_test = test.drop('Id', axis=1)

# dealing with missing data
Xmat = pd.concat([X_train, X_test])
Xmat = Xmat.drop(['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], axis=1)
Xmat = Xmat.fillna(Xmat.median())

# check whether there are still nan
msno.matrix(df=Xmat, figsize=(20, 14), color=(0.5,0,0))

print(Xmat.columns.values)
print(str(Xmat.shape[1]) + ' columns')

# add a new feature 'total sqfootage'
Xmat['TotalSF'] = Xmat['TotalBsmtSF'] + Xmat['1stFlrSF'] + Xmat['2ndFlrSF']
print('There are currently ' + str(Xmat.shape[1]) + ' columns.')


# log-transform the dependent variable for normality
y_train = np.log(y_train)

ax = sns.distplot(y_train)
plt.show()


# train and test
X_train = Xmat.iloc[:train.shape[0], :]
X_test = Xmat.iloc[train.shape[0]:, :]

# Compute the correlation matrix
corr = X_train.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

# feature importance using random forest
rf = RandomForestRegressor(n_estimators=80, max_features='auto')
rf.fit(X_train, y_train)
print('Training done using Random Forest')

ranking = np.argsort(-rf.feature_importances_)
f, ax = plt.subplots(figsize=(11, 9))
sns.barplot(x=rf.feature_importances_[
            ranking], y=X_train.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()
plt.show()

# use the top 30 features only
X_train = X_train.iloc[:, ranking[:30]]
X_test = X_test.iloc[:, ranking[:30]]

# interaction between the top 2
X_train["Interaction"] = X_train["TotalSF"]*X_train["OverallQual"]
X_test["Interaction"] = X_test["TotalSF"]*X_test["OverallQual"]

# zscoring
X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()

# heatmap
f, ax = plt.subplots(figsize=(11, 5))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.heatmap(X_train, cmap=cmap)
plt.show()

# relation to the target
fig = plt.figure(figsize=(12, 7))
for i in np.arange(30):
    ax = fig.add_subplot(5, 6, i+1)
    sns.regplot(x=X_train.iloc[:, i], y=y_train)

plt.tight_layout()
plt.show()

# outlier deletion
Xmat = X_train
Xmat['SalePrice'] = y_train
Xmat = Xmat.drop(Xmat[(Xmat['TotalSF'] > 5) &
                      (Xmat['SalePrice'] < 12.5)].index)
Xmat = Xmat.drop(Xmat[(Xmat['GrLivArea'] > 5) &
                      (Xmat['SalePrice'] < 13)].index)

# recover
y_train = Xmat['SalePrice']
X_train = Xmat.drop(['SalePrice'], axis=1)

# XGBoost

print("Parameter optimization")
xgb_model = xgb.XGBRegressor()
reg_xgb = GridSearchCV(xgb_model,
                       {'max_depth': [2, 4, 6],
                        'n_estimators': [50, 100, 200]}, verbose=1)
reg_xgb.fit(X_train, y_train)
print(reg_xgb.best_score_)
print(reg_xgb.best_params_)


# SVR

reg_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})
reg_svr.fit(X_train, y_train)

print(reg_svr.best_score_)
print(reg_svr.best_params_)

# second feature matrix
X_train2 = pd.DataFrame({'XGB': reg_xgb.predict(X_train),
                         'SVR': reg_svr.predict(X_train),
                         })
print(X_train2.head())

# second-feature modeling using linear regression

reg = linear_model.LinearRegression()
reg.fit(X_train2, y_train)

# prediction using the test set
X_test2 = pd.DataFrame({'XGB': reg_xgb.predict(X_test),
                        'SVR': reg_svr.predict(X_test),
                        })

# Don't forget to convert the prediction back to non-log scale
y_pred = np.exp(reg.predict(X_test2))
test_ID = test['Id']

output = pd.DataFrame({
    "Id": test_ID,
    "SalePrice": y_pred
})
output.to_csv('houseprice.csv', index=False)

