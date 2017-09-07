# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:56:03 2017

@author: thomaseames
"""


"""
###################
# MAIO Exercise 1 #
###################
Tom Eames & Joris van Lammeren
"""

#import relevant libraries
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statsmodels.tsa as tsa

#%%

"""
##################
# Data importing #
##################
"""

#import data files
with open("data1a.txt") as f:
    Data1 = f.readlines()
Data1 = [x.strip() for x in Data1] 

with open("data1b.txt") as f:
    Data2 = f.readlines()
Data2 = [x.strip() for x in Data2] 

with open("data1c.txt") as f:
    Data3 = f.readlines()
Data3 = [x.strip() for x in Data3]

with open("data1d.txt") as f:
    Data4 = f.readlines()
Data4 = [x.strip() for x in Data4]  

with open("data1e.txt") as f:
    Data5 = f.readlines()
Data5 = [x.strip() for x in Data5] 

for i in list(range(0,len(Data1))):
    Data1[i]=float(Data1[i])
    Data3[i]=float(Data3[i])
    Data4[i]=float(Data4[i])
    Data5[i]=float(Data5[i])
    
for j in list(range(0,len(Data2))):
    Data2[j]=float(Data2[j])
    
#organise data into one array
Data=[Data1,Data2,Data3,Data4,Data5]

#%%
"""
################################
# Auto/partial autocorrelation #
################################
"""


for i in range(len(Data)):
    plot_acf(Data[i], lags=50)
    plot_pacf(Data[i], lags=50)


#%%
"""
##########################
# Determine coefficients #   
##########################
"""

#initialise target 

#Using the ARIMA function, determine coefficients

ARMA_mdl1=tsa.arima_model.ARIMA(Data1, order=(0,0,1))  
ARMA_fit1=ARMA_mdl1.fit()
print(ARMA_fit1.summary())

ARMA_mdl2=tsa.arima_model.ARIMA(Data2, order=(0,0,2))  
ARMA_fit2=ARMA_mdl2.fit()
print(ARMA_fit2.summary())

ARMA_mdl3=tsa.arima_model.ARIMA(Data3, order=(1,0,0))  
ARMA_fit3=ARMA_mdl3.fit()
print(ARMA_fit3.summary())

ARMA_mdl4=tsa.arima_model.ARIMA(Data4, order=(1,0,1))  
ARMA_fit4=ARMA_mdl4.fit()
print(ARMA_fit4.summary())

ARMA_mdl5=tsa.arima_model.ARIMA(Data5,order=(1,0,1))  
ARMA_fit5=ARMA_mdl5.fit()
print(ARMA_fit5.summary())

       
#%%
"""
###################
# Check residuals #
###################
"""


res1=pd.DataFrame(ARMA_fit1.resid)
plot_acf(res1, lags=50)
print(res1.describe())

res2=pd.DataFrame(ARMA_fit2.resid)
plot_acf(res2, lags=50)
print(res2.describe())

res3=pd.DataFrame(ARMA_fit3.resid)
plot_acf(res3, lags=50)
print(res3.describe())

res4=pd.DataFrame(ARMA_fit4.resid)
plot_acf(res4, lags=50)
print(res4.describe())

res5=pd.DataFrame(ARMA_fit5.resid)
plot_acf(res5, lags=50)
print(res5.describe())

