### Libraries

from scipy.stats.qmc import LatinHypercube
import numpy as np
from scipy.stats import gamma
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy
import pandas as pd
import plotly.express as px
from random import seed
from random import random
from datetime import datetime

now = datetime.now()
ts = str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"_"+str(now.hour)+"h"+str(now.minute)+"_"

Y = 60 # duration of planning period, 2021-2080
l = 12 # ha of land required
dr = 0.02 #discount rate
N = 100000 # Number of samples


scenList = list(range(0, N))
yrList = list(range(0,Y))
plList = list(range(0,8))
df=np.zeros(len(yrList))
popSt = np.zeros((N,Y))


seed(1)
grow = np.zeros((N,Y))
for s in scenList:
    tempg = list()
    tempg.append(0.007 if random() < 0.5 else 0.013)
    for i in range(1, Y):
        deltaG = -.0028 if random() < 0.5 else .0028
        value = tempg[i-3] + deltaG if i > 2 else tempg[i-1] + deltaG
        tempg.append(value)
    grow[s,] = tempg
    print(grow)
    pop = list()
    for y in yrList:
        popval = pop[y - 2] * (1 + grow[s, y]) if y > 2 else 1
        pop.append(popval)
    # todo store pgi and create distribution
    popSt[s, :] = pop
#plt.plot(popg)
#plt.show()
#temp1 = pd.DataFrame(scenList)
temp2 = pd.DataFrame(popSt).transpose()
#temp3 = pd.concat([temp1,temp2],axis=1)
temp4=pd.melt(temp2)

yr = pd.DataFrame(yrList*N)
yr.columns = ["year"]
popdf = pd.concat([yr,temp4],axis=1)
#print(popdf)

fig = px.line(popdf, x='year', y='value',line_group = 'variable', color='variable',
              title="Results of random walk for user growth",
              labels={  # replaces default labels by column name
              "year": "Year since 2020", "value": "Population index [1 at year 2020"},
                  template="plotly_white")
img_name = "fig3.jpeg"
fig.update_layout(showlegend=False)
fig.update_yaxes(side="right")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=4)
#fig.show()

engine = LatinHypercube(d=3,seed=42)
sample = engine.random(n=N)
#pop =     list(sample[:,0]*0.010+0.010)

tun_shape = 5
tun_year = 20 # 2040
tun_gam =   list(gamma.ppf(sample[:,1], tun_shape) + tun_year)

tundf = pd.DataFrame(tun_gam)
tundf.columns=["year"]

fig = px.histogram(tundf, x='year',histnorm='probability density',
              title="Sample distribution of acceptance for constructing the highway extension",
              labels={  # replaces default labels by column name
              "year": "Year since 2020",'probability density': "Probability density"},
                  template="plotly_white")
img_name = "fig4.jpeg"
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=4)
#fig.show()

av_shape = 7.5
av_year = 34 # 2054
av_gam =   list(gamma.ppf(sample[:,2], av_shape) + av_year)

cavdf = pd.DataFrame(av_gam)
cavdf.columns=["year"]

fig = px.histogram(cavdf, x='year', histnorm='probability density',
              title="Sample distribution of 50% market share for connected automated vehicles",
              labels={  # replaces default labels by column name
              "year": "Year since 2020",'probability density': "Probability density"},
                  template="plotly_white")
img_name = "fig5.jpeg"
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=4)
#fig.show()


import pandas as pd
from scipy.stats.qmc import LatinHypercube
from scipy.stats import gamma
from scipy.stats import norm
sa_id=0
sa=0

scenList = list(range(0, N))
yrList = list(range(0,Y))
sitList = list(range(0,8))
df=np.zeros(len(yrList))
popSt = np.zeros((N,Y))

engine = LatinHypercube(d=4,seed=42)
sample = engine.random(n=N)

engine = LatinHypercube(d=1,seed=44)
RWsample2 = engine.random(n=(N*(Y-1)))

ts="2023-6-12_16h47_"

df = pd.read_csv("results/"+ts+"resultsRefToH.csv",sep=";",decimal=",")
print(df)
df=df.drop(columns=['Unnamed: 0'])
print(df)
sdf = pd.melt(df)
print(sdf)

seed(1)
grow = np.zeros((N,Y))
gList = norm.ppf(list(sample[:, 3]))*0.001+0.005
eList = norm.ppf(list(RWsample2[:,0]))*0.02
for s in scenList:
    pop = list()
    tempe = list()
    #tempg.append(0.007 if sample[s,2] < 0.5 else 0.013)
    z0 = 1
    pop.append(z0)
    deltaG = gList[s]
    tempe.append(0)
    for i in range(1, Y):
        ind = ((s*(Y-1)+i)-1)
        temp_e = eList[ind]
        tempe.append(temp_e)
        zi = z0*((1+deltaG)**i)+sum(tempe)
        pop.append(zi)
    # todo store pgi and create distribution
    popSt[s, :] = pop

vi_mean = .275
vi_sd = 0.055
#vi_gam =   list(gamma.ppf(sample[:,1], vi_shape, scale=vi_scale) * (.035+0.005*sa_id)+.05-.02*sa_id) #was 0.06
vi_norm =   list(norm.ppf(sample[:,1]) * (vi_sd+0.005*sa_id)+(vi_mean-.02*sa_id))
#print(sum(vi_gam)/len(vi_gam))

con_shape = 5
con_year = 17#-sa_con # 2040
sa_factor = 0.9**sa #if sa_con==-1 else 1.1**sa
con_gam =   list(gamma.ppf(sample[:,0], con_shape)*sa_factor + con_year)
#print("con_gam")
#print(con_gam)

condf = pd.DataFrame(con_gam*8)+2023
condf.columns=["year"]

#print(vi_gam)
#print(tun_gam)
#print(av_gam)

vidf = pd.DataFrame(vi_norm*8)
vidf.columns=["percentage"]

Y=60
N = 100000
scenList = list(range(0, N))

scen = pd.DataFrame(scenList*8)
sdf = pd.concat([sdf,scen], axis=1)
#print(sdf)

time = [31]*N+[29]*N+[27]*N+[6]*N+[4]*N+[2]*N+[25]*N+[0]*N
#print(len(time))
timedf = pd.DataFrame(time)
sdf = pd.concat([sdf,timedf], axis=1)
#print(sdf)

sdf = pd.concat([sdf,condf], axis=1)
#print(sdf)

popEnd = pd.DataFrame(popSt[:,(Y-1)].tolist()*8)
sdf = pd.concat([sdf,popEnd], axis=1)
#print(sdf)

sdf = pd.concat([sdf,vidf], axis=1)
print(sdf)


sdf.columns = ['situation','net_benefit','scenario','time_saved','accepted','pop','induced']

sdf.to_csv("results/"+ts+"resultsSA.csv",sep=";",decimal=",")

from plotly.subplots import make_subplots
import plotly.express as px

img_name = "fig11.jpeg"
fig = px.scatter(sdf, x="time_saved", y="net_benefit",symbol="situation",color="situation")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)

img_name = "fig12.jpeg"
fig = px.scatter(sdf, x="accepted", y="net_benefit",symbol="situation",color="situation")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)

img_name = "fig13.jpeg"
fig = px.scatter(sdf, x="pop", y="net_benefit",symbol="situation",color="situation")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)

img_name = "fig14.jpeg"
fig = px.scatter(sdf, x="induced", y="net_benefit",symbol="situation",color="situation")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)

img_name = "fig21.jpeg"
fig = px.violin(sdf, x="time_saved", y="net_benefit",color="situation",box=True)
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)

img_name = "fig22.jpeg"
fig = px.density_contour(sdf, x="accepted", y="net_benefit",color="situation")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)

img_name = "fig23.jpeg"
fig = px.density_contour(sdf, x="pop", y="net_benefit",color="situation")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)

img_name = "fig24.jpeg"
fig = px.density_contour(sdf, x="induced", y="net_benefit",color="situation")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)

img_name = "fig32.jpeg"
fig = px.scatter(sdf, x="accepted", y="net_benefit",symbol="situation",color="situation", opacity=0.02, trendline="lowess")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)

img_name = "fig33.jpeg"
fig = px.scatter(sdf, x="pop", y="net_benefit",symbol="situation",color="situation", opacity=0.02, trendline="lowess")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)

img_name = "fig34.jpeg"
fig = px.scatter(sdf, x="induced", y="net_benefit",symbol="situation",color="situation", opacity=0.02, trendline="lowess")
fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6)