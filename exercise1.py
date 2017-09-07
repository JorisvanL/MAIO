#Python version 3.6

import numpy as np
import random
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm


a1 = 0.75
a2 = -0.75

n=10000
x = 50
x1 = 0
x2 = 0
xlijst = np.array([])

for i in list(range(0,n)):
    Zt = random.uniform(-1, 2) 
    x = a1*x1 + a2*x2 +Zt
    xlijst = np.append(xlijst,x)
    x2 = x1
    x1 = x
    
    
print (xlijst)
plt.plot(xlijst)
plt.xlabel('t')
plt.ylabel('X_t')
plt.grid(True)
plt.show()


plot_acf(xlijst,lags=10)
plot_pacf(xlijst,lags=10)


arma_mod = sm.tsa.ARMA(xlijst, order=(2,0))
arma_res = arma_mod.fit(trend='c', disp=-1)
print(arma_res.summary())

const=arma_res.params[0]
a1=arma_res.params[1]
a2=arma_res.params[2]

n=len(xlijst)
x = 4
x1 = 0
x2 = 0
x2lijst = np.array([])

for i in list(range(0,n)):
    Zt = random.uniform(0, 1)
    x = a1*x1 + a2*x2 + Zt
    x2lijst = np.append(x2lijst,x)
    x2 = x1
    x1 = x
    
#plot_acf(xlijst,lags=10)
#plot_pacf(xlijst, lags=10)

data1=np.array([])

#%%
for i in list(range(0,len(content))):
    lala=xlijst[i]-x2lijst[i]
    data1=np.append(data1,lala)

arma_mod = sm.tsa.ARMA(data1, order=(2,0))
arma_res = arma_mod.fit(trend='c', disp=-1)
print(arma_res.summary())
plot_acf(data1,lags=10)
