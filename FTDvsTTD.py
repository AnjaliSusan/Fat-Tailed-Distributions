import matplotlib.pyplot as plt
import numpy as np
import random as rd


#Understanding the Law of large Numbers

#given pareto with low tail exponent
x_min = np.sqrt(2) / (1 + np.sqrt(2))
alpha = 1 + np.sqrt(2)
std_dev_of_obs_mean = []
exp_mean = []
for i in range(1,10001):
  sample = []
  for j in range(i):
    sample += [(np.random.pareto(alpha) +1 ) * x_min]
  exp_mean += [np.mean(sample)]

x = list(range(1,10001))

#given weibull distribution
alpha = 1
l = 1
std_dev_of_obs_mean = []
for i in range(1,31):
  sample = []
  exp_mean = []
  for k in range(100):
    for j in range(i):
      sample += [l * (np.random.weibull(alpha))]
    exp_mean += [np.mean(sample)]
  std_dev_of_obs_mean += [np.sqrt(np.var(exp_mean))]

x = list(range(1,31))

plt.plot(x, std_dev_of_obs_mean)

plt.plot(x, exp_mean)

#Central Limit Theorem

# uniform distribution
from scipy.stats import uniform
import seaborn as sns
import numpy as np
import statistics as stats
# random numbers from uniform distribution
s = 10000
start = 0
width = 1
n = 3
data_uniform = []
for i in range(s):
  data_uniform += [uniform.rvs(size=n, loc = start, scale=width)]

#print(len(data_uniform))

data_uniform_mean = []

for r in range(len(data_uniform)):
  data_uniform_mean += [np.mean(data_uniform[r])]

#print(len(data_uniform_mean))
#print("Minimum" ,np.min(data_uniform_mean))
#print("Maximum" ,np.max(data_uniform_mean))
#print("Std Deviation" ,stats.stdev(data_uniform_mean))
#print("Variance" ,stats.variance(data_uniform_mean))
#print("Mean", np.mean(data_uniform_mean))

ax = sns.distplot(data_uniform_mean,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Uniform Distribution ', ylabel='Frequency')

#gamma distribution
from scipy.stats import gamma


s = 10000
start = 0
width = 1
n = 40
data_uniform = []
for i in range(s):
  data_uniform += [gamma.rvs(a = 1, scale = 1, size=n)]

#print(len(data_uniform))

data_uniform_mean = []

for r in range(len(data_uniform)):
  data_uniform_mean += [np.mean(data_uniform[r])]

#print(len(data_uniform_mean))
#print("Minimum" ,np.min(data_uniform_mean))
#print("Maximum" ,np.max(data_uniform_mean))
#print("Std Deviation" ,stats.stdev(data_uniform_mean))
#print("Variance" ,stats.variance(data_uniform_mean))
#print("Mean", np.mean(data_uniform_mean))

ax = sns.distplot(data_uniform_mean,
                  bins=100,
                  kde=True,
                  color='red',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Gamma Distribution ', ylabel='Frequency')


#pareto distribution
from scipy.stats import pareto
import seaborn as sns
import numpy as np
import statistics as stats

s = 10000
start = 0
width = 1
n = 10000
data = []
for i in range(s):
  data += [pareto.rvs(b=1.14, scale=1, size=n)]

#print(len(data_uniform))

data_mean = []

for r in range(len(data)):
  data_mean += [np.mean(data[r])]

chk = 0
for i in data_mean:
  if i < 8.14:
    chk += 1
print(chk)

print("Minimum" ,np.min(data_mean))
print("Maximum" ,np.max(data_mean))
print("Std Deviation" ,stats.stdev(data_mean))
print("Variance" ,stats.variance(data_mean))
print("Mean", np.mean(data_mean))

ax = sns.distplot(data_mean,
                  bins=100,
                  kde=True,
                  color='magenta',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Pareto Distribution ', ylabel='Frequency')

#pareto distribution
from scipy.stats import pareto
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

s = 1000
start = 0
width = 1
n = 30
data = []
for i in range(s):
  data += [pareto.rvs(b=1.14, scale=1, size=n)]

#print(len(data_uniform))

data_mean = []

for r in range(len(data)):
  data_mean += [np.mean(data[r])]

print(data_mean)
x = list(range(1,1001))




plt.show()
plt.plot(data_mean)
plt.show()


#Understanding exponential vs subexpomemtial
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,8)
import math

def exponential1(t):
  return math.exp(-t)

def exponential2(t):
  return math.exp(-t**(1/2))


x = list(i/1000 for i in range(10000))
y1 = [exponential1(i) for i in x]

y2 = [exponential2(i) for i in x]

plt.plot(x,y1,'red')
plt.plot(x,y2,'blue')

plt.legend(["Exponential Distribution","Subexponential Distribution"])
#zipfs distribution
def harmonic_n(s,N):
  sum = 0
  for i in range(1,N+1):
    sum += (1/i**s)
  return sum
def zipf_law(k,s,N):
  f = 1 /( (k**s) * harmonic_n(s,N))
  return f

xx = list(range(1,11))

#Let N be 10
s = 1
y = [zipf_law(k,s,10) for k in xx]
plt.plot(xx,y,'b-o')
s = 2
y = [zipf_law(k,s,10) for k in xx]
plt.plot(xx,y,'r-o')
s = 3
y = [zipf_law(k,s,10) for k in xx]
plt.plot(xx,y,'c-o')
s = 4
y = [zipf_law(k,s,10) for k in xx]
plt.plot(xx,y,'m-o')
plt.xlabel('Rank(k)')
plt.ylabel('Normalised frequency')
plt.legend(['s = 1','s = 2','s = 3','s = 4'])
plt.show()


#Studying the ration of std deviation and mad

from scipy.stats import pareto
import seaborn as sns
import numpy as np
import statistics as stats

s = 100
start = 0
width = 1
n = 100
data_pareto = []
for i in range(s):
  data_pareto += [pareto.rvs(b=1.14, scale=1, size=n)]
data_pareto_ratio = []


for i in range(s):
  data_pareto_ratio += [stats.stdev(data_pareto[i]) / pd.DataFrame(data_pareto[i]).mad()]

plt.scatter(range(1,101),data_pareto_ratio)
plt.ylim([1, 8])
plt.show()

#Exponential ratio

s = 10000
start = 0
width = 1
n = 50
data_expon = []
for i in range(s):
  data_expon += [expon.rvs(size=n)]
data_expon_ratio = []


for i in range(s):
  data_expon_ratio += [stats.stdev(data_expon[i]) / pd.DataFrame(data_expon[i]).mad()]

plt.scatter(data_expon_ratio,range(1,10001))
#plt.ylim([1.1, 6])
plt.show()

#Normal Distribution from scipy.stats import norm
from scipy.stats import norm
import seaborn as sns
i=

s = 100
start = 0
width = 1
n = 200
data_norm = []
for i in range(s):
  data_norm += [norm.rvs(size=n)]
data_norm_ratio = []


for i in range(s):
  data_norm_ratio += [stats.stdev(data_norm[i]) / pd.DataFrame(data_norm[i]).mad()]

plt.plot(data_norm_ratio)
#plt.ylim([1.1, 6])
plt.show()

#Pareto Distribution from scipy.stats import pareto
import seaborn as sns
import numpy as np
import statistics as stats

s = 1000
start = 0
width = 1
n = 200
data_pareto = []
for i in range(s):
  data_pareto += [pareto.rvs(b=1.14, scale=1, size=n)]
data_pareto_ratio = []
data_pareto_stdev = []
data_pareto_mad = []

for i in range(s):
  data_pareto_stdev += [stats.stdev(data_pareto[i])]

for i in range(s):
  data_pareto_mad += [pd.DataFrame(data_pareto[i]).mad()]



for i in range(s):
  data_pareto_ratio += [stats.stdev(data_pareto[i]) / pd.DataFrame(data_pareto[i]).mad()]

plt.scatter(range(1,1001),data_pareto_stdev,label ="Std Dev")
#plt.ylim([1, 8])
plt.show()
plt.scatter(range(1,1001),data_pareto_mad,label ="Mean Abs Dev")
#plt.ylim([1, 8])
plt.show()
plt.scatter(range(1,1001),data_pareto_ratio,label ="Ratio")
#plt.ylim([1, 8])
plt.show()


# import pareto distribution
from scipy.stats import pareto
import seaborn as sns
import numpy as np
import statistics as stats
import pandas as pd
import matplotlib.pyplot as plt

start = 0
width = 1
n = 1000
data_pareto = []
data_pareto = pareto.rvs(b=2, scale=0.5, size=n)

#plt.scatter(range(1,201),data_pareto,label ="Std Dev")
#plt.ylim([1, 8])
#plt.show()
ax = sns.distplot(data_pareto,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(tth='Pareto Distribution ', ylabel='Frequency')


# import pareto distribution
from scipy.stats import norm
import seaborn as sns
import numpy as np
import statistics as stats
import pandas as pd
import matplotlib.pyplot as plt

start = 0
width = 1
n = 1000
data_norm = []
data_norm = norm.rvs(size=n)

#plt.scatter(range(1,201),data_pareto,label ="Std Dev")
#plt.ylim([1, 8])
#plt.show()
ax = sns.distplot(data_norm,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Pareto Distribution ', ylabel='Frequency')


#Understanding probabilities

from scipy.stats import norm
import seaborn as sns
import numpy as np
import statistics as stats
import pandas as pd
import matplotlib.pyplot as plt

s = 1000
start = 0
width = 1
n = 1000
data_norm = []
mean_abs_dev = []
std_dev = []
prob = []
prob_1 = []
probsd = []
for i in range(s):
  data_norm += [norm.rvs(size=n)]
  mean_abs_dev += [pd.DataFrame(data_norm[i]).mad()]
  std_dev += [stats.stdev(data_norm[i])]
  z1 = float(np.mean(data_norm[i]) + (3*mean_abs_dev[i]))
  z2 = float(np.mean(data_norm[i]) - (3*mean_abs_dev[i]))

  z3 = float(np.mean(data_norm[i]) + (3*std_dev[i]))
  z4 = float(np.mean(data_norm[i]) - (3*std_dev[i]))


  j = 0
  for x in data_norm[i]:
    if x <= z1 and x >= z2 :
      j+= 1
  prob += [j/n]
  r = 1.29
  t = 0
  for x in data_norm[i]:
    if x >= r:
      t+= 1
  prob_1 +=[t/n]

  sd = 0
  for x in data_norm[i]:
    if x <= z3 and x>= z4:
      sd+= 1
  probsd += [sd/n]

print("Standard Normal Distribution")
print("Probability (x < Mean +- 3 MAD) =", stats.mean(prob))
print("Probability (x < Mean +- 3 SD) =", np.mean(probsd))
print("Probability (x > 1.29) =", stats.mean(prob_1))
print("Variance of probability (x > 1.58) =", stats.variance(prob_1))

from scipy.stats import pareto

s = 1000
start = 0
width = 1
n = 1000
data_pareto = []
mean_abs_dev = []
std_dev = []
prob = []
prob_1 = []
probsd = []

for i in range(s):
  data_pareto += [pareto.rvs(b=2, scale=0.5, size=n)]
  mean_abs_dev += [pd.DataFrame(data_pareto[i]).mad()]
  std_dev += [stats.stdev(data_pareto[i])]
  z1 = float(np.mean(data_pareto[i]) + (3*mean_abs_dev[i]))
  z2 = float(np.mean(data_pareto[i]) + (3*std_dev[i]))
  j = 0
  for x in data_pareto[i]:
    if x <= z1  :
      j+= 1
  prob += [j/n]
  r = 1.58
  t = 0
  for x in data_pareto[i]:
    if x >= r:
      t+= 1
  prob_1 +=[t/n]
  sd = 0
  for x in data_pareto[i]:
    if x <= z2:
      sd+= 1
  probsd += [sd/n]


print("Pareto Distribution: alpha = 2, x_min = 0.5")
print("Probability (x < Mean + 3 MAD) =", stats.mean(prob))
print("Variance of probability (x < Mean + 3 MAD) =", stats.variance(prob))
print(np.mean(std_dev))
print("Probability (x < Mean + 3 SD) =", np.mean(probsd))
print(stats.variance(probsd))
print("Probability (x > 1.58) =", stats.mean(prob_1))
print("Variance of probability (x > 1.58) =", stats.variance(prob_1))

from scipy.stats import expon
import seaborn as sns
import numpy as np
import statistics as stats
import pandas as pd
import matplotlib.pyplot as plt

s = 1000
start = 0
width = 1
n = 1000
data_expon = []
mean_abs_dev = []
std_dev += []
prob = []
probsd = []

for i in range(s):
  data_expon += [expon.rvs(size=n)]
  mean_abs_dev += [pd.DataFrame(data_expon[i]).mad()]
  std_dev += [stats.stdev(data_expon[i])]
  z1 = float(np.mean(data_expon[i]) + (3*mean_abs_dev[i]))
  z2 = float(np.mean(data_pareto[i]) + (3*std_dev[i]))
  j = 0
  for x in data_expon[i]:
    if x <= z1:
      j+= 1
  prob += [j/n]
  sd = 0
  for x in data_expon[i]:
    if x <= z2:
      sd+= 1
  probsd += [sd/n]


print("Exponential Distribution, lambda = 1")
print("Probability (x < Mean + 3 MAD) =", np.mean(prob))
print("Probability (x < Mean + 3 SD) =", np.mean(probsd))
print(stats.variance(probsd))
print("Variance of probability (x < Mean + 3 MAD) =", stats.variance(prob))

#Looking at quantiles

from IPython.display import display

def paretoquantile(p,a,x_m):
  v = x_m/((1-p)**(1/a))
  return v

def exponentialquantile(p, l):
  v = - np.log(1-p) / l
  return v

p = [0.8997, 0.9495, 0.9699, 0.9890, 0.9901, 0.9920, 0.9940, 0.9960, 0.9980, 0.9990,0.9999,0.99999,0.999999]

norm = []
pareto = []
expon = []

for i in range(len(p)):
    norm += [st.norm.ppf(p[i]) + 1]
    pareto += [paretoquantile(p[i],2,0.5)]
    expon += [exponentialquantile(p[i],1)]

index_labels=['Quantile', 'Normal(1,1)','Exponential(1)','Pareto(2,0.5)']

df = pd.DataFrame([p, norm, expon, pareto ],index_labels)

new_header = df.iloc[0]
df = df[1:]
df.columns = new_header
display(df)
import matplotlib.pyplot as plt
plt.plot(norm,p,color = 'red')
plt.plot(expon,p,color='blue')
plt.plot(pareto,p,color='green')
plt.show()

#Reduction in Variance
import matplotlib.pyplot as plt
  
# x axis values
x = np.linspace(0,100,100)
# corresponding y axis values
k1= 1
k2 = 0.5
k3 = 0

y1 = []
for i in x:
  y1 += [i**((k1-1)/(2-k1))]

y2 = []
for i in x:
  y2 += [i**((k2-1)/(2-k2))]

y3 = []
for i in x:
  y3 += [i**((k3-1)/(2-k3))]




plt.plot(x, y1, 'blue',label = "K = 1")
plt.plot(x, y2, 'red', label = "K = 0.5")
plt.plot(x, y3, 'green', label = "K = 0")
  

plt.xlabel('n summands')

plt.ylabel('Variance')
  

plt.title('Reduction in Variance')
  
plt.legend()

plt.show()
