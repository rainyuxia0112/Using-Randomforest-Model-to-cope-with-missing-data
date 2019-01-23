import pandas as pd
data=pd.read_csv('/Users/rain/Desktop/data.csv',header= None,sep=',',index_col=None)
X=data.iloc[:,0:7].values
Y=data.iloc[:,8:9].values


#missing data
def missing_data(n,arr):
    '''n is which col as missing col, using randomforst regressor'''
    import numpy as np
    import pandas as pd
    index = np.array(np.where(arr[:, n] == 0))[0]
    y_train = arr[np.where(arr[:, n] != 0), n].T
    x_train = np.delete(arr[np.where(arr[:, n] != 0)],n,1)
    x_test = np.delete(arr[np.where(arr[:, n] == 0)],n,1)
    # using random forest
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(x_train, y_train)

    # Predicting a new result
    y_pre = regressor.predict(x_test)
    for i in range(len(index)):
        arr[index[i], n] = y_pre[i]
    a=arr
    return (a)


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_ = sc_x.fit_transform(X)
sc_y = StandardScaler()
Y_ = sc_y.fit_transform(Y.reshape(-1,1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_, Y, test_size = 0.2, random_state = 0)

# SVR
from sklearn.svm import SVR
reg=SVR(kernel='rbf')
reg.fit(X_train,y_train)
y_pre=reg.predict(X_test)

import sklearn
sklearn.metrics.mean_squared_error(y_true=y_test, y_pred=y_pre)


#poly
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import PolynomialFeatures
ploy_reg= PolynomialFeatures(degree=1)
X_ploy=ploy_reg.fit_transform(X_train)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_ploy,y_train)
y_pre=reg.predict(ploy_reg.fit_transform(X_test))    #预测时要进行转变
import sklearn
sklearn.metrics.mean_squared_error(y_true=y_test, y_pred=y_pre)



#linear

import numpy as np
X = np.append(arr=np.ones((103,1)).astype(int), values=X, axis=1)   # 把b0 也变成 b0*x0

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

from sklearn.model_selection import train_test_split
#x为数据集的feature熟悉，y为label.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pre=reg.predict(x_test)
import sklearn
sklearn.metrics.mean_squared_error(y_true=y_test, y_pred=y_pre)










