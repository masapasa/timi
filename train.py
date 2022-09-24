# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# %%
df = pd.read_csv("/home/aswin/Documents/timi/house_price.csv")
df
# %%
dropColumns = ["Id", "MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LandSlope", "Condition1", "Condition2", "BldgType", "OverallCond", "RoofStyle", 
               "RoofMatl", "Exterior1st", "Exterior2nd","MasVnrType", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1",
              "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "Heating", "Electrical", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "HalfBath"] + ["SaleCondition", "SaleType", "YrSold", "MoSold", "MiscVal", "MiscFeature", "Fence", "PoolQC", "PoolArea", "ScreenPorch", "3SsnPorch", "EnclosedPorch", "OpenPorchSF", "WoodDeckSF", "PavedDrive", "GarageCond", "GarageQual", "GarageType", "FireplaceQu", "Functional", "KitchenAbvGr", "BedroomAbvGr"]

droppedDf = df.drop(columns=dropColumns, axis=1)
droppedDf.head()
# %%
col_list=droppedDf.columns.tolist()

def splitCategoricalAndNumericalData(col_list: list):
    cat_cols=[]
    num_cols=[]
    for i in col_list:
        if df[i].dtype == "object":
            cat_cols.append(i)
        elif df[i].dtype == "int64" or df[i].dtype == "float64":
            num_cols.append(i)
    return cat_cols, num_cols

cat_cols, num_cols = splitCategoricalAndNumericalData(col_list) 
cat_cols, num_cols
# %%
droppedDf["Alley"].fillna("NO", inplace=True)
droppedDf["Alley"].isna().sum()
droppedDf["LotFrontage"].fillna(df.LotFrontage.mean(), inplace=True)
droppedDf["LotFrontage"].isna().sum()
droppedDf["GarageFinish"].fillna("NO", inplace=True)
droppedDf["GarageFinish"].isna().sum()
droppedDf["GarageYrBlt"].fillna(df.GarageYrBlt.mean(), inplace=True)
droppedDf["GarageYrBlt"].isna().sum()
droppedDf["BsmtQual"].fillna("NO", inplace=True)
droppedDf["BsmtQual"].isna().sum()
droppedDf["MasVnrArea"].fillna(0, inplace=True)
droppedDf["MasVnrArea"].isna().sum()
# %%
droppedDf['MasVnrAreaCatg'] = np.where(droppedDf.MasVnrArea>1000,'BIG',
                                      np.where(droppedDf.MasVnrArea>500,'MEDIUM',
                                              np.where(droppedDf.MasVnrArea>0,'SMALL','NO')))
droppedDf['MasVnrAreaCatg'].isna().sum()
# %%
inputDf = droppedDf.drop(['SalePrice'],axis=1)
inputDf = inputDf.iloc[[0]].copy()
for i in inputDf:
    if inputDf[i].dtype == "object":
        inputDf[i] = droppedDf[i].mode()[0]
    elif inputDf[i].dtype == "int64" or inputDf[i].dtype == "float64":
        inputDf[i] = droppedDf[i].mean()
inputDf

obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
for feature in obj_feat:
    inputDf[feature] = inputDf[feature].astype('category')
# %%
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

# %%
df = droppedDf.copy()
obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)
for feature in obj_feat:
    x = df.drop(['SalePrice'],axis=1)
    y = df.SalePrice

# train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=1)
x.iloc[0].index
# %%
model = lgb.LGBMRegressor(max_depth=5, 
                          n_estimators = 100, 
                          learning_rate = 0.2,
                          min_child_samples = 30)
model.fit(x_train, y_train)

pred_y_train = model.predict(x_train)
pred_y_test = model.predict(x_test)

r2_train = metrics.r2_score(y_train, pred_y_train)
r2_test = metrics.r2_score(y_test, pred_y_test)

msle_train =metrics.mean_squared_log_error(y_train, pred_y_train)
msle_test =metrics.mean_squared_log_error(y_test, pred_y_test)

print(f"Train r2 = {r2_train:.2f} \nTest r2 = {r2_test:.2f}")
print(f"Train msle = {msle_train:.2f} \nTest msle = {msle_test:.2f}")
# %%
