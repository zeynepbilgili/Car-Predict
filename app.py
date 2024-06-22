#!/usr/bin/env python
# coding: utf-8

# # Car Prediction #
# İkinci el araç fiyatlarını (özelliklerine göre) tahmin eden modeller oluşturma ve MLOPs ile Hugging Face üzerinden yayımlayacağız.
# 

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split  # Veri setini bölme işlemleri
from sklearn.linear_model import LinearRegression  # Doğrusal regresyon
from sklearn.metrics import r2_score,mean_squared_error  # Modelimizin performansını ölçmek için
from sklearn.compose import ColumnTransformer  # Sütun dönüşüm işlemleri
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Kategori - sayısal dönüşüm  ve ölçeklendirme
from sklearn.pipeline import Pipeline  # Veri işleme hattı


# In[2]:


# Excell dosyalarını okumak için


# In[3]:


get_ipython().system('pip install xlrd')


# ## Veri dosyasını yükle

# In[4]:


ls


# In[5]:


df=pd.read_excel('cars.xls')
df


# In[6]:


df.info()


# In[7]:


# Veri ön işleme


# In[8]:


X=df.drop('Price',axis=1)  # Fiyat sütunu çıkar fiyata etki edenler kalsın
y=df['Price']  # Tahmin


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# ####  Veri ön işleme, standartlaştırma ve OHE işlemlerini otomatikleştiriyoruz (standarlaştırıyoruz). Artık preprocess kullanarak kullanıcında arayüz aracılığıyla gelen veriyi mdoelimize uygun hale çevirebiliriz.
# 

# In[10]:


preprocess=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),['Mileage', 'Cylinder','Liter','Doors']),
        ('cat',OneHotEncoder(),['Make','Model','Trim','Type'])
    ]
)


# In[11]:


my_model=LinearRegression()


# In[12]:


# Pipeline'ı tanımla
pipe=Pipeline(steps=[('preprocessor',preprocess),('model',my_model)])


# In[13]:


# Pipeline fit
pipe.fit(X_train,y_train)


# In[14]:


y_pred=pipe.predict(X_test)
print('RMSE',mean_squared_error(y_test,y_pred)**0.5)
print('R2',r2_score(y_test,y_pred))


# In[15]:


# İsterseniz veri setinin tammamıyla tekrar eğitim yapabilirsiniz.
# pipe.fit(X,y)


# ## Streamlit ile Modeli Yayma/Deploy/Kullanıma Sunma

# In[16]:


get_ipython().system('pip install streamlit')


# In[17]:


df['Mileage'].max()


# In[18]:


df['Type'].unique()


# In[19]:


df['Liter'].max()


# #### Python ile yapılan çalışmnalrın hızlı bir şekilde deploy edilmesi için HTML render arayüzler tasarlamanızı sağlar.

# In[22]:


import streamlit as st
# Price tahmin fonksiyonu tanımla
def price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather):
    input_data=pd.DataFrame({'Make':[make],
                             'Model':[model],
                             'Trim':[trim],
                             'Mileage':[mileage],
                             'Type':[car_type],
                             'Cylinder':[cylinder],
                             'Liter':[liter],
                             'Doors':[doors],
                             'Cruise':[cruise],
                             'Sound':[sound],
                             'Leather':[leather]})
    prediction=pipe.predict(input_data)[0]
    return prediction
st.title("II. El Araba Fiyatı Tahmin:red_car: @zeynepbilgili")
st.write('Arabanın özelliklerini seçiniz')
make=st.selectbox('Marka',df['Make'].unique())
model=st.selectbox('Model',df[df['Make']==make]['Model'].unique())
trim=st.selectbox('Trim',df[(df['Make']==make) &(df['Model']==model)]['Trim'].unique())
mileage=st.number_input('Kilometre',100,200000)
car_type=st.selectbox('Araç Tipi',df[(df['Make']==make) &(df['Model']==model)&(df['Trim']==trim)]['Type'].unique())
cylinder=st.selectbox('Cylinder',df['Cylinder'].unique())
liter=st.number_input('Yakıt hacmi',1,10)
doors=st.selectbox('Kapı sayısı',df['Doors'].unique())
cruise=st.radio('Hız Sbt.',[True,False])
sound=st.radio('Ses Sis.',[True,False])
leather=st.radio('Deri döşeme.',[True,False])
if st.button('Tahmin'):
    pred=price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather)
    st.write('Fiyat:$', round(pred[0],2))


# In[23]:


# streamlit run C:\ProgramData\anaconda3\Lib\site-packages\ipykernel_launcher.py

