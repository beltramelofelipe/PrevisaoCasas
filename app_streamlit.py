#invite people for the Kaggle party
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import streamlit as st
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# warnings.filterwarnings('ignore')
# # %matplotlib inline

st.write("""

Prevendo valor de Casas\n
App que utiliza machine learning para prever valores de casas
""")
df = pd.read_csv('train_clean.csv')
st.subheader('Informações dos dados')

user_input = st.sidebar.text_input('Digite seu nome')
y = df.SalePrice
feature = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
       'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

X = df[feature]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def get_user_data():

    OverallQual = st.sidebar.slider('Qualidade do material', 0, 10, 1)
    GrLivArea = st.sidebar.slider('Area de estar (chão) pes quadrados', 0.0, 10.0, 1.0)
    GarageCars = st.sidebar.slider('Garagem', 0, 4, 1)    
    GarageArea = st.sidebar.slider('Tamanho da Garagem em metros quadrados', 0, 1390, 1)
    TotalBsmtSF = st.sidebar.slider('Total de metros quadrados de area de porao', 0.0, 10.0, 1.0)
    FullBath = st.sidebar.slider('Banheiros completos acima do nivel', 0, 4, 1)
    TotRmsAbvGrd = st.sidebar.slider('Total de quartos acima do nível (não inclui banheiros)', 0, 14, 1)
    YearBuilt = st.sidebar.slider('Ano de construção', 1872, 2010, 2000) 

    user_data = {'Qualidade do material': OverallQual,
                'Area de estar (chão) pes quadrados': GrLivArea,
                'Garagem':GarageCars,
                'Tamanho da Garagem em metros quadrados':GarageArea,
                'Total de metros quadrados de area de porao':TotalBsmtSF,
                'Banheiros completos acima do nivel':FullBath,
                'Total de quartos acima do nível (não inclui banheiros)':TotRmsAbvGrd,
                'Ano de construção': YearBuilt
                }

    features = pd.DataFrame(user_data, index=[0])
     
    return features

user_input_variables = get_user_data()

graf = st.bar_chart(user_input_variables)

st.subheader('Informaçãoes da Casa')
st.write(user_input_variables)

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X,train_y)

predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y,predictions)

st.subheader('Acuracia do modelo ')
st.write(rf_val_mae * 100)

prediction = rf_model.predict(user_input_variables)

st.subheader('O valor da casa: ')
st.write(prediction)