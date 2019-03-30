""" this py is to build a simple function to cope with missing data in dataframe
"""
### import packages and data
import pandas as pd
data=pd.read_csv('/Users/rain/Desktop/data.csv',header= None,sep=',',index_col=None)
X=data.iloc[:,0:7].values
Y=data.iloc[:,8:9].values

#missing data
def missing_data(n,arr,type):
    '''
    this function is to use random forest model to cope with missing data
    n -- col as missing col, using randomforst regressor
    arr -- the predictors (X)
    type -- the type of missing data, such as continuous or category
    return -- new array of predictors
    '''
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










