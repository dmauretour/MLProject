import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import missingno as msno
import seaborn as sns
import seaborn as sb # visualization
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor




df = pd.read_csv('Melbourne_housing_FULL.csv')
df.rename(columns={'Lattitude':'Latitude', 'Longtitude':'Longitude'}, inplace=True)
df.isnull().sum()
#msno.bar(df)
#plt.show()

# Q1 = df['Price'].quantile(0.25)
# Q3 = df['Price'].quantile(0.75)
# IQR = Q3-Q1
# Lower_Whisker = Q1 - 1.5*IQR
# Upper_Whisker = Q3 + 1.5*IQR
# df = df[(df['Price']>Lower_Whisker)&(df['Price']<Upper_Whisker)]
# plt.figure(figsize=(10,5))
# sns.distplot(df['Price'],hist=True, kde=False, color='blue')
# plt.ylabel('Counts')
# plt.show()

# sb.set_style('whitegrid') # plot style
# plt.rcParams['figure.figsize'] = (20, 10) # plot size
# sb.heatmap(df.corr(), annot = True, cmap = 'magma')

# plt.savefig('heatmap.png')
# plt.show()

# sns.distplot(df['Price'])
# plt.show()

# JG1 = sns.jointplot('Rooms', 'Price', data=df, kind='hex', color='g')
# JG2 = sns.jointplot('Bathroom', 'Price', data=df, kind='hex', color='b')
# JG3 = sns.jointplot('Car', 'Price', data=df, kind='hex', color='r')
# JG4 = sns.jointplot('Distance', 'Price', data=df, kind='hex', color='orange')
# JG1.savefig('JG1.png')
# plt.close(JG1.fig)
# JG2.savefig('JG2.png')
# plt.close(JG2.fig)
# JG3.savefig('JG3.png')
# plt.close(JG3.fig)
# JG4.savefig('JG4.png')
# plt.close(JG4.fig)
# f, ax = plt.subplots(2,2,figsize=(20,16))
# ax[0,0].imshow(mpimg.imread('JG1.png'))
# ax[0,1].imshow(mpimg.imread('JG2.png'))
# ax[1,0].imshow(mpimg.imread('JG3.png'))
# ax[1,1].imshow(mpimg.imread('JG4.png'))
# [ax.set_axis_off() for ax in ax.ravel()]
# plt.tight_layout()
# plt.show()


df = pd.concat([df, pd.get_dummies(df["Type"]), pd.get_dummies(df["Method"]), pd.get_dummies(df["Regionname"])], axis=1)
df = df.drop(["Suburb", "Address", "SellerG", "CouncilArea", "Type", "Method", "Regionname"], 1)
df['Date'] = [pd.Timestamp(x).timestamp() for x in df["Date"]]
df = df.dropna()
df.head()

X=df.drop('Price', axis=1)
y=df['Price']
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3, random_state=0)

def Predictive_Model(estimator):
    estimator.fit(train_X, train_y)
    prediction = estimator.predict(test_X)
    print('R_squared:', metrics.r2_score(test_y, prediction))
    print('Square Root of MSE:',np.sqrt(metrics.mean_squared_error(test_y, prediction)))
    plt.figure(figsize=(10,5))
    sns.distplot(test_y, hist=True, kde=False)
    sns.distplot(prediction, hist=True, kde=False)
    plt.legend(labels=['Actual Values of Price', 'Predicted Values of Price'])
    plt.xlim(0,)
    plt.show()
def FeatureBar(model_Features, Title, yLabel):
    plt.figure(figsize=(10,5))
    plt.bar(df.columns[df.columns!='Price'].values, model_Features)
    plt.xticks(rotation=45)
    plt.title(Title)
    plt.ylabel(yLabel)
    plt.show()

lr = LinearRegression()
Predictive_Model(lr)

knn = KNeighborsRegressor(n_neighbors=5)
Predictive_Model(knn)

dt = DecisionTreeRegressor(max_depth=15, random_state=0)
Predictive_Model(dt)

regressor = ['Linear Regression', 'KNN', 'Decision Tree']
models = [LinearRegression(), KNeighborsRegressor(n_neighbors=5), DecisionTreeRegressor(max_depth=15, random_state=0)]
R_squared = []
RMSE = []
for m in models:
    m.fit(train_X, train_y)
    prediction_m = m.predict(test_X)
    r2 = metrics.r2_score(test_y, prediction_m)
    rmse = np.sqrt(metrics.mean_squared_error(test_y, prediction_m))
    R_squared.append(r2)
    RMSE.append(rmse)
basic_result = pd.DataFrame({'R squared':R_squared,'RMSE':RMSE}, index=regressor)
#print(basic_result)

# scoring={'R_squared':'r2','MSE':'neg_mean_squared_error'}
# def CrossVal(estimator):
#     scores = cross_validate(estimator, X, y, cv=10, scoring=scoring)
#     r2 = scores['test_R_squared'].mean()
#     mse = abs(scores['test_Square Root of MSE'].mean())
#     print('R_squared:', r2)
#     print('Square Root of MSE:', np.sqrt(mse))

# lr2 = LinearRegression()
# CrossVal(lr2)

print("Cross Validation Score: ", cross_validate(lr, test_X, test_y, cv=5))
print("Cross Validation Score: ", cross_validate(knn, test_X, test_y, cv=5))
print("Cross Validation Score: ", cross_validate(dt, test_X, test_y, cv=5))

lr_scores = cross_validate(LinearRegression(), X, y, cv=10, scoring='r2')
knn_scores = cross_validate(KNeighborsRegressor(n_neighbors=16), X, y, cv=10, scoring='r2')
dt_scores = cross_validate(DecisionTreeRegressor(max_depth=9, random_state=0), X, y, cv=10, scoring='r2')
lr_test_score = lr_scores.get('test_score')
knn_test_score = knn_scores.get('test_score')
dt_test_score = dt_scores.get('test_score')
box= pd.DataFrame({'Linear Regression':lr_test_score,'K-Nearest Neighbors':knn_test_score, 'Decision Tree':dt_test_score})
box.index = box.index + 1
box.loc['Mean'] = box.mean()
print(box)

