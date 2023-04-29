from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

path = '/content/drive/MyDrive/Honours work/Climate Seminar - Kedar/Daily Prec 1951-2020.csv'
precip = pd.read_csv(path)

#%%
import seaborn as sns
import numpy as np
import statistics as stats
import random as rd
import matplotlib.pyplot as plt
data_ratio = []


for i in range(len(precip.axes[0])):
  dis_prec = list(precip.iloc[i])
  dis_prec.pop(0)
  dis_prec.pop(0)
  dis_prec.pop(0)
  data_ratio += [stats.stdev(dis_prec) / pd.Data
  
 k= []

ratios = data_ratio.copy()

#print(data_ratio)

def maxi(list1):
  m = list1[0]
  for i in list1:
    if m < i:
      m = i
  return m


while len(k) < 10:
  m = maxi(ratios)
  for j in range(len(data_ratio)):
    #print(data_ratio[i])
    if m == data_ratio[j]:
      k += [j]
  ratios.remove(m)

print(k)


for i in k:
  dis_prec = list(precip.iloc[i])
  print("District:",dis_prec[1])
  print("State:",dis_prec[2])
  print(i)
  print(data_ratio[i])
  dis_prec.pop(0)
  dis_prec.pop(0)
  dis_prec.pop(0)

  ax = sns.distplot(dis_prec,
                  bins=100,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
  ax.set(xlabel='Distribution ', ylabel='Frequency')
  #plt.xlim(0.1,4)
  #plt.ylim(0,2000)
  plt.show()
  new = pd.Series(dis_prec)
  print(new.describe())
  
#Looking at one district
dis_prec = list(precip.iloc[95])
dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.describe()
print( stats.stdev(dis_prec) / pd.DataFrame(dis_prec).mad() [0])


#Checking heuristics

from scipy.stats import skew

print("Skewness:" , skew(dis_prec, axis=0, bias=True))

from scipy.stats import kurtosis

print("Kurtosis:" ,kurtosis(dis_prec, axis=0, fisher = True, bias=True))

import random as rd
import numpy as np
import matplotlib.pyplot as plt


dis_prec = list(precip.iloc[95])
dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)

dis_prec1 = []
for k in dis_prec:
  if k == 0:
    dis_prec1 += [0.00001]
  else:
    dis_prec1 += [k]

s = 1000

data = []
for i in range(s):
  data += [rd.sample(dis_prec1,i)]

#print(len(data_uniform))

data_mean = []

for r in range(len(data)):
  data_mean += [np.mean(data[r])]


x = list(range(1,1001))


plt.scatter(x,data_mean)
plt.xlabel("Sample Size")
plt.ylabel("Sample Mean")
plt.title("Testing if the Law of Large Numbers holds")
plt.ylim(0,2)
plt.show()

import seaborn as sns
import numpy as np
import statistics as stats
import random as rd

dis_prec = list(precip.iloc[95])
dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)

print(np.min(dis_prec1))
s = 1000

n = 3000
data = []
for i in range(s):
  data += [rd.sample(dis_prec1,n)]



data_mean = []

for r in range(len(data)):
  data_mean += [np.mean(data[r])]


ax = sns.distplot(data_mean,
                  bins=100,
                  kde=True,
                  color='magenta',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Precipitation Mean(in mm) ', ylabel='Frequency')
plt.title("Testing if the Central Limit Theorem holds")

#showing that tail wags the dog
new = pd.Series(dis_prec1)
print(new.describe())

ct = 0
for k in dis_prec:
  if k == 0:
    ct+=1

print(ct/len(dis_prec)*100)

#Infinite variance
exp_var = []
for i in range(1,1001):
  sample = pareto.rvs(b=1.14, scale=1, size=200)
  exp_var += [stats.variance(sample)]

x = list(range(1,1001))
print(np.min(exp_var))
print(np.max(exp_var))
plt.scatter(x,exp_var)
plt.ylabel("Variance")
plt.show()

plt.scatter(x,exp_var)
#plt.ylim(0,4000)
plt.ylabel("Variance")
exp_var_nor = []
for i in range(1,1001):
  sample = norm.rvs(loc = 0, scale = 1, size = 200)
  exp_var_nor += [stats.variance(sample)]

x = list(range(1,1001))

plt.scatter(x,exp_var_nor,color='red')
plt.ylim(-1,200)
plt.ylabel("Variance")

#trying to fit distributions to the data

import scipy
from scipy.stats import pareto
import matplotlib.pyplot as plt

dis_prec = list(precip.iloc[95])
dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)

#ax = sns.distplot(dis_prec,
#                  bins=100,
#                  kde=True,
#                  color='skyblue',
#                  hist_kws={"linewidth": 15,'alpha':1})
#ax.set(xlabel='Distribution ', ylabel='Frequency')

_, bins, _ = plt.hist(dis_prec, 100)

b,l,s = scipy.stats.pareto.fit(dis_prec,b, loc=0, scale=1)
print(b,l,s)
best_fit_line = scipy.stats.pareto.pdf(bins, b,l,s)
plt.hist(dis_prec, 100, density=1)
plt.plot(bins, best_fit_line)
plt.xlim(0,20)
plt.ylim(0,1)
plt.show()


import scipy
from scipy.stats import expon
import matplotlib.pyplot as plt
from scipy.stats import poisson

dis_prec = list(precip.iloc[95])
dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)

#ax = sns.distplot(dis_prec,
#                  bins=100,
#                  kde=True,
#                  color='skyblue',
#                  hist_kws={"linewidth": 15,'alpha':1})
#ax.set(xlabel='Distribution ', ylabel='Frequency')

_, bins, _ = plt.hist(dis_prec, 100, density=1)

b,l = scipy.stats.expon.fit(dis_prec)
print("Expon",b,l)
m = np.mean(dis_prec)
#print(m)
best_fit_line = scipy.stats.expon.pdf(dis_prec, (1/m))
#plt.hist(dis_prec, 100, density=1)
#plt.plot(dis_prec, best_fit_line)
#plt.xlim(0,20)
#plt.ylim(0,1)
#plt.show()

#Poisson
l = len(dis_prec)
_, bins, _ = plt.hist(dis_prec, 100)
plt.show()
f = []
for i in range(len(bins)-1):
  ct = 0
  for j in dis_prec:
    if bins[i] <= j and j < bins[i+1]:
      ct +=1
  f += [ct]

print(f[0])

m = np.mean(f)
print(m)
x = list(range(100))
best_fit_line = scipy.stats.poisson.pmf(x, m)
#plt.hist(dis_prec, 100, density=1)
#plt.plot(x, best_fit_line)
#plt.xlim(0,20)
#plt.ylim(0,1)
plt.show()
plt.show()

chi_square_test_statistic, p_value = stats.chisquare(dis_prec, best_fit_line)
print('chi_square_test_statistic is : ' +str(chi_square_test_statistic))
print('p_value : ' + str(p_value))
print('Chi square critical value',stats.chi2.ppf(1-0.05, df=6))


##qqplot for exponential

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


dis_prec = list(precip.iloc[95])
dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)

dis_prec = np.array(dis_prec)

fig = sm.qqplot(dis_prec, dist = scipy.stats.distributions.expon(0.0,0.539786434040899), line="q")
plt.show()
fig = sm.qqplot(dis_prec, dist = scipy.stats.distributions.expon(0.0,0.539786434040899), line="45")
plt.show()
fig = sm.qqplot(dis_prec, dist = scipy.stats.distributions.expon(0.0,0.539786434040899), line="s")
plt.show()
fig = sm.qqplot(dis_prec, dist = scipy.stats.distributions.expon(0.0,0.539786434040899), line="r")
plt.show()
fig = sm.qqplot(dis_prec, dist = scipy.stats.distributions.expon(0.0,0.539786434040899), line=None)
plt.show()

#Fitting a zipf distribution

import scipy
import scipy.stats as stats
import numpy as np
from scipy.stats import pareto
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

dis_prec = list(precip.iloc[95])
dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)

l = len(dis_prec)
_, bins, _ = plt.hist(dis_prec, 100, density=1)
plt.show()
rank = []
for i in range(len(bins)-1):
  ct = 0
  for j in dis_prec:
    if bins[i] <= j and j < bins[i+1]:
      ct +=1
  rank += [ct/len(dis_prec)]
#print(bins)
rank.sort(reverse=True)
xx = list(range(1,101))

#new = [j/len(dis_prec) for j in rank]

plt.plot(xx,rank,'r-o')

def harmonic_n(s,N):
  sum = 0
  for i in range(1,N+1):
    sum += (1/i**s)
  return sum

def zipf_law(k,s,N):
  f = 1 /( (k**s) * harmonic_n(s,N))
  return f

#Let N be 10
s = 4.3
y = [zipf_law(k,s,len(rank)) for k in xx]
#print(y)
plt.plot(xx,y,'b-o')
plt.xlabel("Rank")
plt.ylabel("Frequency")
#plt.ylim(0,20)
plt.xlim(0,20)

#Modifying the Zipf Distribution 
import scipy
import scipy.stats as stats
import numpy as np
from scipy.stats import pareto
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

dis_prec = list(precip.iloc[95])
dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)

l = len(dis_prec)
_, bins, _ = plt.hist(dis_prec, 100, density=1)
plt.show()
rank = []
for i in range(len(bins)-1):
  ct = 0
  for j in dis_prec:
    if bins[i] <= j and j < bins[i+1]:
      ct +=1
  rank += [ct]
#print(bins)
rank.sort(reverse=True)

for j in range(len(rank)):
  if rank[j] < 5:
    print(j)
    break

new_rank = []
for t in range(len(rank)):
  if t <= 24:
    new_rank +=[rank[t]]
  else :
    #print(type(new_rank[24]))
    new_rank[24] = (new_rank[24]+rank[t])


for k in range(len(rank)):
  rank[k]= rank[k]/l


#plt.hist(dis_prec, 100, density=1)



#print(new_rank[-1])

xx = list(range(1,101))

plt.plot(xx,rank,'r-o')

#Let N be 10
s = 4.3
y = [zipf_law(k,s,len(rank)) for k in xx]
#print(y)
plt.plot(xx,y,'b-o')

zipf_obs = []
for z in range(len(y)):
  if z <=24:
    zipf_obs += [y[z]*len(dis_prec)]
  else:
    #print(z)
    #print(type(y[z]))
    #print(y[z])
    #print(type(zipf_obs[24]))
    #print(zipf_obs[24])
    zipf_obs[24] = (zipf_obs[24]+y[z])*len(dis_prec)



new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])

new_rank.remove(new_rank[-1])
zipf_obs.remove(zipf_obs[-1])



print("Obs",new_rank)
print("Expected",zipf_obs)

plt.xlim(0,20)
plt.ylim(0,0.3)
plt.show()


chi_square_test_statistic, p_value = stats.chisquare(new_rank, zipf_obs)
#print(ks_2samp(new_rank,zipf_obs))
print('chi_square_test_statistic is : ' +str(chi_square_test_statistic))
print('p_value : ' + str(p_value))
print('Chi square critical value',stats.chi2.ppf(1-0.05, df=6))

nx = list(np.linspace(0,0.1,101))

freq = []
for i in range(len(nx)-1):
  ct = 0
  for j in y:
    if nx[i] <= j and j < nx[i+1]:
      ct +=1
  freq += [ct]

nx.remove(nx[-1])
#print(len(nx))
plt.plot(nx,freq,'r-o')

pdf = []
a = (1/4.3) / np.min(y)**(-1/4.3)

for j in y:
  pdf += [a/(j**(1 + (1/4.3)))]
#print(y)
plt.plot(y,pdf,'b-o')
#print(len(y))
plt.xlim(0,0.1)

#fitting a weibull distribution
import scipy
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


dis_prec = list(precip.iloc[95])

dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)


# Fit Weibull function, some explanation below
params = scipy.stats.weibull_min.fit(dis_prec, floc=0, f0=1)
shape = params[0]
scale = params[2]
print('shape:',shape)
print('scale:',scale)

#### Plotting
# Histogram first
values,bins,hist = plt.hist(dis_prec,bins=51,range=(0,10))
center = (bins[:-1] + bins[1:]) / 2.

# Using all params and the stats function
plt.ylim(0,2000)
plt.plot(center,scipy.stats.weibull_min.pdf(center,*params),lw=4,label='scipy')

# Using my own Weibull function as a check
def weibull(u,shape,scale):
  '''Weibull distribution for wind speed u with shape parameter k and scale parameter A'''
  return (shape / scale) * (u / scale)**(shape-1) * np.exp(-(u/scale)**shape)

plt.plot(center,weibull(center,shape,scale),label='Precipitation',lw=2)
plt.legend()
print(scipy.stats.weibull_min.pdf(bins,*params))
chi_square_test_statistic, p_value = stats.chisquare(dis_prec, scipy.stats.weibull_min.pdf(bins,*params))
#print(ks_2samp(new_rank,zipf_obs))
print('chi_square_test_statistic is : ' +str(chi_square_test_statistic))
print('p_value : ' + str(p_value))
print('Chi square critical value',stats.chi2.ppf(1-0.05, df=6))

# Zipf Plot
import scipy
import scipy.stats as stats
import numpy as np
from scipy.stats import pareto
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

dis_prec = list(precip.iloc[95])

dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)

print(np.max(dis_prec))

t = list(np.linspace(0,np.max(dis_prec),100))
cdf = []
for j in range(len(t)-1):
  cdf += [0]
  for k in range(len(dis_prec)):
    if dis_prec[k] <= t[j]:
      cdf[j] = cdf[j]+1
  cdf[j] = cdf[j]/len(dis_prec)

ccdf_log = []

for j in cdf:
  ccdf_log += [np.log(1-j)]

t_log = []

t.remove(t[-1])

for x in t:
  t_log += [np.log(x)]

plt.plot(t_log, ccdf_log)
plt.xlabel("log(x)")
plt.ylabel("log(1- F(x))")

#Mean Excess Function
dis_prec = list(precip.iloc[95])

dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)

thresh_v = list(np.linspace(0,np.max(dis_prec),100))

thresh_v.remove(thresh_v[-1])
meplot = [0]*99
for j in range(len(thresh_v)):
  ct = 0
  s = 0
  for k in dis_prec:
    if k > thresh_v[j]:
      s+= (k-thresh_v[j])
      ct+= 1
  meplot[j] = s/ct


plt.plot(thresh_v,meplot)
plt.xlabel("Threshold x")
plt.ylabel("e(x)")

#Fitting a pareto using MM and testing using a qqplot
import scipy
from scipy.stats import pareto
import matplotlib.pyplot as plt

dis_prec = list(precip.iloc[95])
dis_prec.pop(0)
dis_prec.pop(0)
dis_prec.pop(0)

b,l,s = scipy.stats.pareto.fit(dis_prec,method = "MM")
print(b,l,s)
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

fig = sm.qqplot(np.array(dis_prec), dist = scipy.stats.distributions.pareto(1.78,0.56,1.09*10**(-5)), line="45")
plt.show()

#Tesing using probplot
scipy.stats.probplot(dis_prec, sparams=(1.787933088364047, 0.5684784097342674,1.0923288694281289e-05), dist='pareto', fit=True, plot=plt, rvalue=False)
plt.show()

#Code for kappa metric
def kappa(n,n_0,data):
  k = 2 - ((np.log(n)- np.log(n_0))/np.log(MAD(n,data)/MAD(n_0,data)))
  return k
