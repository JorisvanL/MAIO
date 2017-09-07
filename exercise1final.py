#Python version 3.6



"""
###################
# MAIO Exercise 1 #
###################
Tom Eames & Joris van Lammeren
"""

#import relevant libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


#%%
"""
#########################
# Initialise parameters #
#########################
"""

#define alpha1 and alpha2
a1 = 1
a2 = -1

#initial values for the dataset to be generated
n=100
x = 0
x1 = 10
x2 = 0

#initialise a blank array
xlijst = np.array([])

#%%
"""
####################
# Generate dataset #
####################
"""

for i in list(range(0,n)):
    Zt = random.uniform(-1, 1) #white noise
    x = a1*x1 + a2*x2 
    xlijst = np.append(xlijst,x)
    x2 = x1
    x1 = x
    
#%%
"""
#########
# Plots #
#########
"""

# plot the data
plt.plot(xlijst)
plt.xlabel('t',fontsize=12)
plt.ylabel('$X_t$',fontsize=12)
plt.title("AR(2) generated dataset")
plt.grid(True)
plt.show()

#plot the autocorrelation
plt.figure()
plot_acf(xlijst)

plt.xlabel("Lag")
