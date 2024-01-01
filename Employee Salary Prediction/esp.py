import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

df= pd.read_csv("Salary_Data.csv")
df.head()

df.shape

df.columns

df.columns=['Age', 'Gender', 'Degree', 'Job_Title', 'Exp_Years','Salary']
df

df.dtypes

df.info()

df[df.duplicated()]

df[df.duplicated()].shape

df1=df.drop_duplicates(keep='first')
df1.shape

df1.isnull().sum()

df1.dropna(how='any', inplace=True)

df1.isnull().sum()

df1.describe()

corr=df1[['Age','Exp_Years','Salary']].corr()
corr

sns.heatmap(corr, annot=True)

df1['Degree'].value_counts()

df1['Degree'].value_counts().plot(kind='bar')

df1['Job_Title'].value_counts()

df1['Job_Title'].unique()

df1['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')

df1.head()

# Feature Engineering

from sklearn.preprocessing import LabelEncoder
label_Encoder=LabelEncoder()

df1["Gender_Encode"]=label_Encoder.fit_transform(df1['Gender'])

df1["Degree_Encode"]=label_Encoder.fit_transform(df1['Degree'])

df1["Job_Title_Encode"]=label_Encoder.fit_transform(df1['Job_Title'])

df1.head()

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()

df1['Age Scaled']=std_scaler.fit_transform(df1[['Age']])
df1['Exp_Years Scaled']=std_scaler.fit_transform(df1[['Exp_Years']])

df1.head()

X = df1[['Age Scaled','Gender_Encode','Degree_Encode','Job_Title_Encode','Exp_Years Scaled']]
y = df1['Salary']

X.head()

y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_train.head(10)

x_test.head()

from sklearn.linear_model import LinearRegression
Liner_Regression_model = LinearRegression()

Liner_Regression_model.fit(x_train, y_train)

y_pred = Liner_Regression_model.predict(x_test)

y_pred

y_test.head(10)

df = pd.DataFrame({'y_actual':y_test, 'y_predicted':y_pred})
df

df['error']=df['y_predicted']-df['y_actual']
df

df['abs_error']=abs(df['error'])
df

mean_absolute_error=df['abs_error'].mean()
mean_absolute_error

from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

r2_score(y_test, y_pred)

round(mean_absolute_error(y_test, y_pred),2)

Liner_Regression_model.coef_

Liner_Regression_model.intercept_



