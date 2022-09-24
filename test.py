#%%
import streamlit as st
import pandas as pd
import pickle
df = pd.read_csv("/home/aswin/Documents/house-prices-prediction-LGBM/data/house_price.csv")

dropColumns = ["Id", "MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LandSlope", "Condition1", "Condition2", "BldgType", "OverallCond", "RoofStyle",
            "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "Heating", "Electrical", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "HalfBath"] + ["SaleCondition", "SaleType", "YrSold", "MoSold", "MiscVal", "MiscFeature", "Fence", "PoolQC", "PoolArea", "ScreenPorch", "3SsnPorch", "EnclosedPorch", "OpenPorchSF", "WoodDeckSF", "PavedDrive", "GarageCond", "GarageQual", "GarageType", "FireplaceQu", "Functional", "KitchenAbvGr", "BedroomAbvGr"]

droppedDf = df.drop(columns=dropColumns, axis=1)
modelName = "/home/aswin/Documents/house-prices-prediction-LGBM/trained_model.model"
loaded_model = pickle.load(open(modelName, 'rb'))
st.title("House Prices Prediction")
st.write("##### This is a simple model for house prices prediction.")

st.sidebar.title("Model Parameters")
st.sidebar.write("### Feature importance of model")

expander= st.sidebar.expander("Click Here for Feature Importance of Model ")
expander.write("## Feature Importance of Model")

# Get Feature importance of model
featureImportances = pd.Series(loaded_model.feature_importances_,index = droppedDf.columns).sort_values(ascending=False)[:20]
featureImportances
# %%
inputDf = droppedDf.iloc[[0]].copy()
inputDf
# %%
