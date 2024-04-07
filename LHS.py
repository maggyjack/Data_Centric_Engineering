from scipy.stats import qmc
import matplotlib.pyplot as plt

#initialise the sampler
sampler = qmc.LatinHypercube(d=2) #set number of dimensions
sample = sampler.random(n=20) #set sample size and generate the inital sample

#define the first variable upper and lower bounds
var1low = 0.001
var1upper = 0.2

#define the second variable upper and lower bounds
var2low = 0
var2upper = 20

#format into combined upper and lower bounds for the sampler to read
lowerBound = [var1low, var2low]
upperBound = [var1upper, var2upper]

#scale the sample ranged from 0 - 1, into the desired scaling
scaledSample = qmc.scale(sample, lowerBound, upperBound)

#generate a plot of the sample
plt.plot(scaledSample[:,0], scaledSample[:,1],'o')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('Sample Space (LHS)')
plt.show()