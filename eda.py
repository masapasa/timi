# %% [markdown]
# # House Prices Prediction

# %% [markdown]
# ### Importing

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import bamboolib as bam
import seaborn as sns
import plotly.express as px


# %% [markdown]
# #### Load dataset

# %%
df = pd.read_csv("/home/aswin/Documents/timi/house_price.csv")
df

# %% [markdown]
# #### Explore the data

# %%
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df.isnull().sum().sort_values(ascending=False)

# %% [markdown]
# #### Pandas profiling - optional

# %%
# import pandas_profiling 

# profile = droppedDf.profile_report(title='Pandas Profiling Report')
# profile.to_file(output_file="Data_Profiling_v3.html")

# %% [markdown]
# #### EDA for categorical and numerical features

# %% [markdown]
# Show the distrubition of target value that column "SalePrice"

# %%
fig = px.histogram(df, x='SalePrice')
fig

# %% [markdown]
# There are many outliers. The distrubition is like a normal distrubition. The feature is target value. Because of we will not make changing.

# %% [markdown]
# We will use features that have correlation with "SalePrice".

# %%
corr = df.corr()
g = sns.heatmap(corr,  vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},fmt='.2f', cmap='coolwarm')
sns.despine()
g.figure.set_size_inches(14,10)
    
plt.show()