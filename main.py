### Libraries
from fractions import Fraction
from scipy.stats.qmc import LatinHypercube
import numpy as np
from scipy.stats import gamma
from scipy.stats import norm
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy
import pandas as pd
import plotly.express as px
from random import seed
from random import random
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.subplots as sp
import statsmodels
import sys
from PIL import Image

pio.templates.default = "plotly_white"

from datetime import datetime
import os

### Turn sensitivity analysis on (0 = No) (1= Yes)
sa = 0
# either -1 or 1 or 0
sa_con= 0
sa_rw= 0
sa_id = 0

# don't touch
sa_con = sa_con*sa
sa_rw = sa_rw*sa
sa_id = sa_id*sa

now = datetime.now()
ts = str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"_"+str(now.hour)+"h"+str(now.minute)+"_"


ts = ts+"sa_" if sa==1 else ts
ts = ts+"con("+str(sa_con)+")_" if sa_con!=0 else ts
ts = ts+"rw("+str(sa_rw)+")_" if sa_rw!=0 else ts
ts = ts+"id("+str(sa_id)+")_" if sa_id!=0 else ts

if not os.path.exists("images"):
    os.mkdir("images")
if not os.path.exists("results"):
    os.mkdir("results")


### Input variables
## General
Y = 60 # duration of planning period, 2023-2082
l = 12 # ha of land required
dr = 0.02 #discount rate
N = int(50000*(0.01**sa)) # Sample size
## Costs & duration

# for decision 1: participative consultation shortening
c_part = -1000000
t_part = 8
t_part_N = 4

# for decision 2: land acquisition
c_acq = -2000000*0.05# -500*10000*l*0.05 #unit price of ha multiplied by area
#acq_pen = 0.05 # combined relative cost of owning land and loss of value of land when sold
#b_acq = 0#-1*(1-acq_pen)*c_acq
t_acq = 0
t_acq_N = 2

## for decision 3: mitigating interventions for induced demand
c_rail = -15000000
t_rail = 0
t_rail_N = 25


# for delay cost
acc = 0#1  # relative time to free flow time that is accepted delay
ph = float(3) # number of peak hours in a day
vtt = float(20) # value of travel time
dpy = float(250) # weekdays per year

## Factors for sensitivity analysis

#sa1 =


### Input distributions / Samples
## Plot the distributions

# Population

engine = LatinHypercube(d=4,seed=32)#seed=42
sample = engine.random(n=N)

engine = LatinHypercube(d=1,seed=54)#seed=44
RWsample2 = engine.random(n=(N*(Y-1)))


format1 = "png"
format2 = "tiff"


scenList = list(range(0, N))
yrList = list(range(0,Y))
sitList = list(range(0,8))
df=np.zeros(len(yrList))
popSt = np.zeros((N,Y))


#seed(1)
#grow = np.zeros((N,Y))
#for s in scenList:
#    tempg = list()
#    tempg.append(0.007 if sample[s,2] < 0.5 else 0.013)
#    for i in range(1, Y):
#        ind = ((s*(Y-1)+i)-1)
#        deltaG = -.0028-.0005*sa_rw if RWsample2[ind,0] < 0.5 else .0028+.0005*sa_rw
#        value = tempg[i-3] + deltaG if i > 2 else tempg[i-1] + deltaG
#        tempg.append(value)
#    grow[s,] = tempg
#    pop = list()
#    for y in yrList:
#        popval = pop[y - 2] * (1 + grow[s, y]) if y > 2 else 1
#        pop.append(popval)
#    # todo store pgi and create distribution
#    popSt[s, :] = pop
#print("popSt")
#print(popSt)

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
#print("popSt")
#print(popSt)
popmax = np.zeros((Y))
popmin = np.zeros((Y))
for t in yrList:
    popmax[t] = max(popSt[:, t])
    popmin[t] = min(popSt[:, t])



poprand = popSt[5495,:] #picked a random number between 0 and N #5494
yr = pd.DataFrame(yrList) + 2023

prd = poprand.flatten().tolist()
pmn = popmin.flatten().tolist()
pmx = popmax.flatten().tolist()
yli = yr.values.flatten().tolist()

img_name = "fig3_popWalkScens"

layout = go.Layout(xaxis=dict(
                       title="Year"
                   ),
                   yaxis=dict(
                       title="Population index [2023 = 1.0]"
                   ),font=dict(family="Open Sans",size=22)
)

fig = go.Figure([
    go.Scatter(
        x=yli,
        y=prd,
        line=dict(color='rgb(0,100,80)'),
        mode='lines',
        showlegend=False
    ),
    go.Scatter(
        x=yli+yli[::-1],  # x, then x reversed
        y=pmn+pmx[::-1],  # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    )
],layout=layout)

fig.write_image("images/"+ts+img_name+"."+format1,width=1000, height=800, scale=1,format=format1)

image = Image.open("images/"+ts+img_name+"."+format1)
image.save(
            "images/"+ts+img_name+"."+format2,
            dpi=(600, 600),
            compression=None,
            artist="Your_Name_Here",
        )
#os.remove(f"{file_name}.png")

#fig.show()

##temp1 = pd.DataFrame(scenList)
#temp2 = pd.DataFrame(popSt).transpose()
##temp3 = pd.concat([temp1,temp2],axis=1)
#temp4=pd.melt(temp2)
#
#yr = pd.DataFrame(yrList*N) + 2023
#yr.columns = ["year"]
#popdf = pd.concat([yr,temp4],axis=1)
##print(popdf)
#
#fig = px.line(popdf, x='year', y='value',line_group = 'variable', color='variable',
#              title="Results of random walk for user growth",
#              labels={  # replaces default labels by column name
#              "year": "Year", "value": "Population index [1 at year 2023]"},
#                  template="plotly_white")
#img_name = "fig3.jpeg"
#fig.update_layout(showlegend=False)
#fig.update_yaxes(side="right")

###fig.write_image("images/"+ts+img_name,width=800, height=600, scale=4)
#fig.show()

con_shape = 5
con_year = 17-sa_con # 2040
sa_factor = 0.9**sa if sa_con==-1 else 1.1**sa
con_gam =   list(gamma.ppf(sample[:,0], con_shape)*sa_factor + con_year)
#print("con_gam")
#print(con_gam)

condf = pd.DataFrame(con_gam)+2023
condf.columns=["year"]

img_name = "fig4_hist_acc"
fig = px.histogram(condf, x='year',histnorm='probability density',
                   title="",
              #title="Sample distribution of acceptance for constructing the highway extension",
              labels={  # replaces default labels by column name
              "year": "Year","probability density": "Probability density"},
                  template="plotly_white")
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Probability density",
    font=dict(
        family="Open Sans",
        size=22
    )
)


#fig.write_image("images/"+ts+img_name,width=400, height=300, scale=8,format="jpeg")
fig.write_image("images/"+ts+img_name+"."+format1,width=1000, height=800, scale=1,format=format1)
image = Image.open("images/"+ts+img_name+"."+format1)
image.save(
            "images/"+ts+img_name+"."+format2,
            dpi=(600, 600),
            compression=None,
            artist="Your_Name_Here",
        )
#fig.show()

vi_shape = 8
vi_scale = 0.5
vi_mean = .275
vi_sd = 0.055
#vi_gam =   list(gamma.ppf(sample[:,1], vi_shape, scale=vi_scale) * (.035+0.005*sa_id)+.05-.02*sa_id) #was 0.06
vi_norm =   list(norm.ppf(sample[:,1]) * (vi_sd+0.005*sa_id)+(vi_mean-.02*sa_id))
print(max(vi_norm))
print(min(vi_norm))
#print(sum(vi_gam)/len(vi_gam))

#print(vi_gam)
#print(tun_gam)
#print(av_gam)

vidf = pd.DataFrame((vi_norm))
vidf.columns=["percentage"]

img_name = "fig5_hist_inducedDem"
fig = px.histogram(vidf, x='percentage', histnorm='probability density',
                    title="",
                    #title="Sample distribution of highway induced demand without mitigating interventions",
                    labels={   "percentage": "Induced demand as a share of actual demand", # replaces default labels by column name
                    "probability density": "Probability density"},
                    template="plotly_white")
fig.update_layout(
    xaxis_title="Added relative demand due to construction",
    yaxis_title="Probability density",
    font=dict(
        family="Open Sans",
        size=22
    )
)

#fig.write_image("images/"+ts+img_name,width=400, height=300, scale=8,format="jpeg")
fig.write_image("images/"+ts+img_name+"."+format1,width=1000, height=800, scale=1,format=format1)
image = Image.open("images/"+ts+img_name+"."+format1)
image.save(
            "images/"+ts+img_name+"."+format2,
            dpi=(600, 600),
            compression=None,
            artist="Your_Name_Here",
        )
#fig.show()

### Simulation


d1 = list([1,1,0,1,1,0,0,0])
d2 = list([1,0,1,1,0,1,0,0])
d3 = list([1,1,1,0,0,0,1,0])


# C = (c_acq + b_acq + c_cons + c_hw) + (c_noTunnelAV + c_delay +
v0 = float(2500) # cars per h
cc = float(1400) # cars per h
c=cc
y_acq = 12 #2035

dC = np.zeros((N,Y,8))
sC = np.zeros((N,Y,8))
ownerCost = np.zeros((N,8))
delayCost = np.zeros((N,8))
safetyCost = np.zeros((N,8))
yConSt = np.zeros((N,8))
yAVSt = np.zeros((N,8))
#popSt = np.zeros((N,Y))
vSt = np.zeros((N,Y))

for sit in sitList:
    #time[sit] =  t_acq + (1-d2a[sit])*t_acq_N + t_part + (1-d1a[sit])*t_part_N + t_cav + (1-d3[sit])*t_cav_N
    for sc in scenList:

        y_con = con_gam[sc] + t_acq + (1-d2[sit])*t_acq_N + t_part + (1-d1[sit])*t_part_N
        #print(y_con)
        yConSt[sc,sit] = y_con
        #cond = (y_con<35)
        y_rail = y_con+t_rail+(1-d3[sit])*t_rail_N #if cond else 99999
        #print(y_rail)

        d_noCon=1 if y_con > Y else 0
        d_noRail=1 if y_rail > Y else 0
        #pop = list()
        vv = list()
        for y in yrList:
            cond = ((d3[sit] != 1)&(y > y_con)&(y<=y_rail))
            #print(cond)
            v = float(v0) * popSt[sc, y] * ((1 + vi_norm[sc])) if cond else float(v0) * popSt[sc, y]
            #print(v)
            c=float(4000) if y > y_con else cc
            #print(power1)
            #print(v)
            # v = v0 * popSt[sc, y]  if y > 0 else v0
            #print(v)
            dCtemp = (1+0.15*float((v/c)**4))-1-acc
            #print(dCtemp)
            dc = max(dCtemp,0)
            vv.append(v)
            dC[sc,y,sit] = -1*(v*dc)*ph*vtt*dpy / ((1 + dr) ** y) # [time/time] * [CHF/year]
            #sC[sc,y,sit] = -3150000*(1-hca*d_trig)*p/ ((1 + dr) ** y)
        vSt[sc, :] = vv
        #popSt[sc,:] = pop
        #power3 = (y_con *d3[sit]+y_rail*(1-d3[sit]))#todo remove the y_rail as the coordination costs should only impact if mitigating
        ownerCost[sc, sit] = (c_part * d1[sit]) + ((c_acq*(1-(1/((1+dr)**(y_con-y_acq))))/dr) * d2[sit] / ((1 + dr) ** y_acq)) + (c_rail * d3[sit]/ ((1 + dr) ** y_con))

delayCost = dC.sum(1)
#safetyCost = sC.sum(1)

#print(ownerCost)
#print(delayCost)
#print(safetyCost)

netB = ownerCost + delayCost #+ safetyCost
df = pd.DataFrame(netB)
df.columns = ['A','B','C','D','E','F','G','H']
dfH = np.tile(np.array(df.loc[:,"H"]).transpose(), (8, 1)).transpose()

refdf = df-dfH
# print(refdf)

df.to_csv("results/"+ts+"results.csv",sep=";",decimal=",")
refdf.to_csv("results/"+ts+"resultsRefToH.csv",sep=";",decimal=",")

ocdf = pd.DataFrame(ownerCost)
ocdf.columns = ['A','B','C','D','E','F','G','H']
ocdf.to_csv("results/"+ts+"ownerCosts.csv",sep=";",decimal=",")

dcdf = pd.DataFrame(delayCost)
dcdf.columns = ['A','B','C','D','E','F','G','H']
dcdf.to_csv("results/"+ts+"delayCosts.csv",sep=";",decimal=",")



img_name = "fig1_CDF-Results.jpeg"
fig = px.ecdf(df, x=["A","B","C","D","E","F","G","H"],line_dash = 'variable',
              title="Empirical CDF for process variants of differing responsiveness",
              labels={  "probability": "Empirical CDF", # replaces default labels by column name
                  "variable":"Variant", "value": "Net benefit [CHF]"},
                  template="plotly_white")

#fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")
#fig.show()

img_name = "fig2_CDF-ResultsRefToH"
fig = px.ecdf(refdf, x=["A","B","C","D","E","F","G","H"],line_dash = 'variable',
              title="",
              #title="Empirical CDF for situations of differing responsiveness",
              labels={   "probability": "Empirical CDF", # replaces default labels by column name
                  "variable": "Variant", "value": "Net benefit relative to Process variant H [CHF]"},
                  template="plotly_white")
fig.update_layout(
    xaxis_title="Net benefit relative to Process variant H [CHF]",
    yaxis_title="Empirical cumulative distribution function",
    font=dict(
        family="Open Sans",
        size=22
    )
)
fig.write_image("images/"+ts+img_name+"."+format1, width=1000, height=800, scale=1,format=format1)
image = Image.open("images/"+ts+img_name+"."+format1)
image.save(
            "images/"+ts+img_name+"."+format2,
            dpi=(600, 600),
            compression=None,
            artist="Your_Name_Here",
        )
#fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")
#fig.show()



#refdf = pd.read_csv("results/"+ts+"resultsRefToH.csv",sep=";",decimal=",")
#print(df)
#refdf=refdf.drop(columns=['Unnamed: 0'])
#print(refdf)
sdf = pd.melt(refdf)
print(sdf)

condff = pd.DataFrame(con_gam*8)+2023
condff.columns=["year"]

#print(vi_gam)
#print(tun_gam)
#print(av_gam)

vidff = pd.DataFrame(vi_norm*8)
vidff.columns=["percentage"]

scen = pd.DataFrame(scenList*8)
sdf = pd.concat([sdf,scen], axis=1)
#print(sdf)

time = [31]*N+[29]*N+[27]*N+[6]*N+[4]*N+[2]*N+[25]*N+[0]*N
#print(len(time))
timedf = pd.DataFrame(time)
sdf = pd.concat([sdf,timedf], axis=1)
#print(sdf)

sdf = pd.concat([sdf,condff], axis=1)
#print(sdf)

popEnd = pd.DataFrame(popSt[:,(Y-1)].tolist()*8)
sdf = pd.concat([sdf,popEnd], axis=1)
#print(sdf)

sdf = pd.concat([sdf,vidff], axis=1)
print(sdf)


sdf.columns = ['situation','net_benefit','scenario','time_saved','accepted','pop','induced']
sdf.to_csv("results/"+ts+"resultsSA.csv",sep=";",decimal=",")

#import joypy


#jdf1 = pd.DataFrame(refdf)
#jdf1 = pd.melt(refdf)
#jdf1 = pd.concat([jdf1,condff.apply(np.ceil)], axis=1)
#jdf1['year'] = jdf1['year'].astype({'year': 'int'})
#jdf1['year'] = jdf1['year'].astype({'year': 'str'})
#jdf1 = jdf1[jdf1["variable"]!="H"]

#fig, axes = joypy.joyplot(jdf1, by = "value",legend=True,alpha=0.8,figsize=(16,16))
#fig, axes = joypy.joyplot(jdf1, by = "variable",legend=True,alpha=0.8,figsize=(16,16),hist=True,bins=16)#ylim='own',
#fig.write_image("images/testme"+ts+img_name,width=800, height=600, scale=6,format="jpeg")

#for ax in axes[:-1]:  # last axis is just for global settings
#    ax.secondary_yaxis('right', functions=(lambda x:x, lambda x:x))

#fig.savefig('images/testme.png')
print("fin")
# plt.show()


#img_name = "fig11.jpeg"
#fig = px.scatter(sdf, x="time_saved", y="net_benefit",symbol="situation",color="situation")
#fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")

#img_name = "fig12.jpeg"
#fig = px.scatter(sdf, x="accepted", y="net_benefit",symbol="situation",color="situation")
#fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")

#img_name = "fig13.jpeg"
#fig = px.scatter(sdf, x="pop", y="net_benefit",symbol="situation",color="situation")
#fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")

#img_name = "fig14.jpeg"
#fig = px.scatter(sdf, x="induced", y="net_benefit",symbol="situation",color="situation")
#fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")

img_name = "fig21.jpeg"
fig = px.box(sdf, x="time_saved", y="net_benefit",color="situation",notched=True
             ,labels={"situation":"Variant","net_benefit":"Net benefit [CHF]",
                      "time_saved":"Reduction of planning time in years"})
fig.add_annotation(x= 31.041,y=2800000000,text="A",showarrow=False,yshift=10)
fig.add_annotation(x= 29.041,y=2250000000,text="B",showarrow=False,yshift=10)
fig.add_annotation(x= 27.041,y=1550000000,text="C",showarrow=False,yshift=10)
fig.add_annotation(x= 25.041,y=700000000,text="G",showarrow=False,yshift=10)
fig.add_annotation(x= 6.041,y=2600000000,text="D",showarrow=False,yshift=10)
fig.add_annotation(x= 4.041,y=1750000000,text="E",showarrow=False,yshift=10)
fig.add_annotation(x= 2.041,y=1100000000,text="F",showarrow=False,yshift=10)
fig.add_annotation(x= 0.041,y=0,text="H",showarrow=False,yshift=10)
fig.update_traces(width=0.5)
fig.update_layout(
    font=dict(
        family="Open Sans",
        size=22
    )
)
#fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")
fig.write_image("images/"+ts+img_name+"."+format1,width=1000, height=800, scale=1,format=format1)
image = Image.open("images/"+ts+img_name+"."+format1)
image.save(
            "images/"+ts+img_name+"."+format2,
            dpi=(600, 600),
            compression=None,
            artist="Your_Name_Here",
        )
#fig.show()
#img_name = "fig22.jpeg"
#fig = px.density_contour(sdf, x="accepted", y="net_benefit",color="situation")
#fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")

#img_name = "fig23.jpeg"
#fig = px.density_contour(sdf, x="pop", y="net_benefit",color="situation")
#fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")

#img_name = "fig24.jpeg"
#fig = px.density_contour(sdf, x="induced", y="net_benefit",color="situation")
#fig.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")

#ts="2023-7-14_21h27_"
#sdf = pd.read_csv("results/"+ts+"resultsSA.csv",sep=";",decimal=",")

#ts="2023-7-14_21h27_"

img_name = "fig32.jpeg"
fig32 = px.scatter(sdf, x="accepted", y="net_benefit",symbol="situation",color="situation", opacity=0.02)#, trendline="lowess")
fig32.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")
#fig32.show()

img_name = "fig33.jpeg"
fig33 = px.scatter(sdf, x="pop", y="net_benefit",symbol="situation",color="situation", opacity=0.02)#, trendline="lowess")
fig33.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")
#fig33.show()

img_name = "fig34.jpeg"
fig34 = px.scatter(sdf, x="induced", y="net_benefit",symbol="situation",color="situation", opacity=0.02)#, trendline="lowess")
fig34.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")
#fig34.show()

img_name = "fig42.jpeg"
fig42 = px.scatter(sdf, x="accepted", y="net_benefit",color="situation", symbol="situation", trendline="lowess")
fig42.data = [t for t in fig42.data if t.mode == "lines"]
fig42.add_annotation(x= 2041,y=450000000,text="A",showarrow=False,yshift=10)
fig42.add_annotation(x= 2041,y=410000000,text="D",showarrow=False,yshift=10)
fig42.add_annotation(x= 2041,y=333000000,text="B",showarrow=False,yshift=10)
fig42.add_annotation(x= 2041,y=275000000,text="E",showarrow=False,yshift=10)
fig42.add_annotation(x= 2041,y=198000000,text="C",showarrow=False,yshift=10)
fig42.add_annotation(x= 2041,y=135000000,text="F",showarrow=False,yshift=10)
fig42.add_annotation(x= 2041,y=65000000,text="G",showarrow=False,yshift=10)
fig42.add_annotation(x= 2041,y=0,text="H",showarrow=False,yshift=10)
fig42.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")


img_name = "fig43.jpeg"
fig43 = px.scatter(sdf, x="pop", y="net_benefit",color="situation", symbol="situation", trendline="lowess")
fig43.data = [t for t in fig43.data if t.mode == "lines"]
fig43.add_annotation(x= 2.041,y=500000000,text="A",showarrow=False,yshift=10)
fig43.add_annotation(x= 2.041,y=430000000,text="D",showarrow=False,yshift=10)
fig43.add_annotation(x= 2.041,y=360000000,text="B",showarrow=False,yshift=10)
fig43.add_annotation(x= 2.041,y=298000000,text="E",showarrow=False,yshift=10)
fig43.add_annotation(x= 2.041,y=212000000,text="C",showarrow=False,yshift=10)
fig43.add_annotation(x= 2.041,y=145000000,text="F",showarrow=False,yshift=10)
fig43.add_annotation(x= 2.041,y=80000000,text="G",showarrow=False,yshift=10)
fig43.add_annotation(x= 2.041,y=0,text="H",showarrow=False,yshift=10)
fig43.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")

img_name = "fig44.jpeg"
fig44 = px.scatter(sdf, x="induced", y="net_benefit",color="situation", symbol="situation", trendline="lowess")
fig44.data = [t for t in fig44.data if t.mode == "lines"]
fig44.write_image("images/"+ts+img_name,width=800, height=600, scale=6,format="jpeg")







fig32_traces = []
fig33_traces = []
fig34_traces = []
fig42_traces = []
fig43_traces = []
fig44_traces = []
for trace in range(len(fig32["data"])):
    fig32["data"][trace]['showlegend'] = False
    fig32_traces.append(fig32["data"][trace])
for trace in range(len(fig42["data"])):
    ############ The major modification. Manually set 'showlegend' attribute to False. ############
    fig42["data"][trace]['showlegend'] = True
    fig42_traces.append(fig42["data"][trace])
for trace in range(len(fig33["data"])):
    fig33_traces.append(fig33["data"][trace])
    fig33["data"][trace]['showlegend'] = False
for trace in range(len(fig43["data"])):
    ############ The major modification. Manually set 'showlegend' attribute to False. ############
    fig43["data"][trace]['showlegend'] = False
    fig43_traces.append(fig43["data"][trace])
for trace in range(len(fig34["data"])):
    fig34_traces.append(fig34["data"][trace])
    fig34["data"][trace]['showlegend'] = False
for trace in range(len(fig44["data"])):
    ############ The major modification. Manually set 'showlegend' attribute to False. ############
    fig44["data"][trace]['showlegend'] = False
    fig44_traces.append(fig44["data"][trace])

# Create a 3x2 subplot
this_figure = sp.make_subplots(rows=2, cols=3, column_titles=[ '(a) User growth over 80 years','(b) Year of construction', '(c) Demand due to construction'],
                               row_titles=['Scatter','LOWESS curve'],
                               y_title="Net benefit [CHF]")
this_figure.update_layout(height=600, width=900,  title_font_size=25)

# Get the Express fig broken down as traces and add the traces to the proper plot within the subplot
for traces in fig32_traces:
    this_figure.add_trace(traces, col=2, row=1)
for traces in fig33_traces:
    this_figure.add_trace(traces, col=1, row=1)
for traces in fig34_traces:
    this_figure.add_trace(traces, col=3, row=1)
for traces in fig42_traces:
    this_figure.add_trace(traces, col=2, row=2)
for traces in fig43_traces:
    this_figure.add_trace(traces, col=1, row=2)
this_figure.add_annotation(col=1,row=2,x= 2.041,y=1600000000,text="A",showarrow=False,yshift=10)
this_figure.add_annotation(col=1,row=2,x= 2.041,y=1320000000,text="D",showarrow=False,yshift=10)
this_figure.add_annotation(col=1,row=2,x= 2.121,y=1050000000,text="B",showarrow=False,yshift=10)
this_figure.add_annotation(col=1,row=2,x= 2.041,y=900000000,text="E",showarrow=False,yshift=10)
this_figure.add_annotation(col=1,row=2,x= 2.121,y=600000000,text="C",showarrow=False,yshift=10)
this_figure.add_annotation(col=1,row=2,x= 2.041,y=470000000,text="F",showarrow=False,yshift=10)
this_figure.add_annotation(col=1,row=2,x= 2.041,y=260000000,text="G",showarrow=False,yshift=10)
this_figure.add_annotation(col=1,row=2,x= 2.041,y=0,text="H",showarrow=False,yshift=10)

for traces in fig44_traces:
    this_figure.add_trace(traces, col=3, row=2)

img_name = "subplot"
this_figure.show()
this_figure.write_image("images/"+ts+img_name,width=900, height=600, scale=6,format="jpeg")
print('done')

#fig32.show()

#
now = datetime.now()
ts_fin = str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"_"+str(now.hour)+"h"+str(now.minute)+"_"

print("This run started at")
print(ts)
print("This run finished at")
print(ts_fin)