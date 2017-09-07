import numpy as np
import matplotlib.pyplot as plt
import random
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm


with open("/Users/jorisvanlammeren/Documents/Studie/MAIO/data1c.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content] 

for i in list(range(0,len(content))):
    content[i]=float(content[i])
    
plot_acf(content,lags=10)
plot_pacf(content, lags=10)

#data 1a ARIMA(2,0,0) a1>0, a2<0 

arma_mod = sm.tsa.ARMA(content, order=(1,0))
arma_res = arma_mod.fit(trend='c', disp=1)
print(arma_res.summary())


#%%
const=arma_res.params[0]
a1=arma_res.params[1]
a2=0

n=len(content)
x = 0
x1 = 1
x2 = 0
xlijst = np.array([])

for i in list(range(0,n)):
    Zt = random.uniform(-1, 1)
    x = a1*x1 + a2*x2 + Zt
    xlijst = np.append(xlijst,x)
    x2 = x1
    x1 = x
    
#plot_acf(xlijst,lags=10)
#plot_pacf(xlijst, lags=10)

data1=np.array([])

#%%
for i in list(range(0,len(content))):
    lala=content[i]-xlijst[i]
    data1=np.append(data1,lala)

plt.plot(data1)
plt.show()
plot_acf(data1,lags=10)
plot_pacf(data1,lags=10)
#%%

arma_mod1 = sm.tsa.ARMA(data1, order=(1,0))
arma_res1 = arma_mod1.fit(trend='c', disp=1)
print(arma_res1.summary())

const2=arma_res1.params[0]
a1=arma_res1.params[1]
a2=0
#%%
n=len(content)
x = 0
x1 = 1
x2 = 0
x2lijst = np.array([])

for i in list(range(0,n)):
    
    x = a1*x1 + a2*x2 
    x2lijst = np.append(x2lijst,x)
    x2 = x1
    x1 = x
#%%
data2=np.array([])

for i in list(range(0,len(content))):
    lala=content[i]-x2lijst[i]
    data2=np.append(data2,lala)
    
plt.plot(data2)
plt.show()
plot_acf(data2,lags=10)
plot_pacf(data2,lags=10)

