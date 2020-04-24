#!/usr/bin/env python
# coding: utf-8

# In[273]:


import time
import statistics
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


# In[274]:


SymbolList = [['AAPL','2018-01-01','2019-01-01'],['HPQ','2018-01-01','2019-01-01'],['GOOG','2018-01-01','2019-01-01']]


# In[ ]:





# In[275]:


import pandas_datareader as pdr
from datetime import datetime


# In[276]:


#     ListD = []
#     def ListDates():
#         start_date= '2016-01-01'
#         end_date= '2018-01-01'
#         symbol='AAPL'
#         stock_dates=a
#         for t in range(1,len(stock_dates)):
#             ListD.append(stock_dates[t][2])
#         return ''


# In[277]:


# ListDates


# In[278]:


i=0
ListTotReturns=[]
while i< len(SymbolList):
    symbol = SymbolList[i][0]
    start_date= SymbolList[i][1]
    end_date= SymbolList[i][2]
    stock_list=pdr.get_data_yahoo(symbol,start_date,end_date)
    stock_list['log_ret']=np.log(stock_list['Close'])-np.log(stock_list['Close'].shift(1))
    stock_list.drop(stock_list.index[0],inplace=True)

    Listreturns=stock_list['log_ret'].tolist()


    ListTotReturns.append(Listreturns)
    i+=1


# Profit and Loss Computation 1

# In[279]:


ModList=[]
for n in range(len(ListTotReturns[0])):
    p=0
    while p< len(ListTotReturns):
        ModList.append([ListTotReturns[p][n]])
        p+=1
ModListTotReturns= np.array(ModList,float)
ModListTotReturns=ModListTotReturns.reshape(int(len(Listreturns)),int(len(SymbolList)))


# In[280]:


len(ListTotReturns)


# #portfolio weights assignment

# In[281]:


PortfolioW=[0.5,0.25,0.25]
Weights=np.array(PortfolioW,float)


# Profit and loss computation 2

# In[282]:


ProfitandLossD=np.dot(ModListTotReturns,Weights)
PLDsorted=np.sort(ProfitandLossD)


# Plot histogram

# In[283]:


plt.figure(figsize=(10,6))
ax=plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks(fontsize=14)
plt.yticks(range(5,101,5),fontsize=14)
plt.title('Portfolio Profit & Loss',fontsize=20)
plt.xlabel('Log-Returns',fontsize=16)
plt.ylabel('Count',fontsize=16)
plt.hist(PLDsorted,color='g',bins=20)


# Historical VaR and CVaR

# In[284]:


PLVaR0=PLDsorted[0]
PLVaR1=PLDsorted[1]
print('The P&L VaR (99%) is ')
print ('%.4f%%' % (100 * PLVaR1))


# In[285]:


VaRs =[PLVaR0,PLVaR1]
PLCVaR = statistics .mean(VaRs)
print('The P&L CVaR (99%) is ') 
print ('%.4f%%' % (100 * PLCVaR))


# Compute the correlation between the series.

# In[286]:


Series=np.array(ListTotReturns,float)
print('The correlation Matrix is: ')
CorrMatrix=np.corrcoef(ma.masked_invalid(Series))
CorrMatrix


# In[287]:


PCovMatrix = np.cov(Series)
print ('The Covariance Matrix is : ')
print (PCovMatrix)


# In[288]:


print ('The Portfolio Mean is : ')
Pmean = statistics .mean(ProfitandLossD)
print('%.3f%%'% (100 * Pmean))


# In[289]:


PVariance=np.dot(Weights,np.dot(PCovMatrix,Weights.transpose()))


# In[290]:


print('The Portfolio Variance is:')
print('%.3f%%'% (100 * PVariance))


# In[291]:


print ('The Portfolio Volatility is : ')
PVol = np.sqrt(PVariance)
print('%.3f%%'% (100 * PVol))


# Parametric VaR Computation

# In[292]:


def ParametricMeasureVaR(ProfitandLossD):
    print('Parametric VaR Computation')
    t=eval(input('Enter the time period: '))
    alpha=eval(input('Enter the confidence level(ex:2.5,1.65): '))
    Nalpha=-alpha
    T=sp.sqrt(t)
    ask=input('Do you want the EWMA Volatility or the Historical(y)? ')
    if ask=='y':
        ParametricVaR = PVol*Nalpha*T
        return "%.4f%%" % (100 * ParametricVaR)
    else:
        PandLD=[b**2 for b in ProfitandLossD]
        Lambda=0.94
        EWMAVariance=[]
        for i in range(len(PandLD)):
            EWMAVariance= (1-Lambda)*(Lambda**i)*PandLD[i]
            EWMAVariance.append(EWMAVariance)
        EWMAVariance = sum(EWMAVariance)
        EWMAVol = np.sqrt(EWMAVariance)
        ParametricVaR = EWMAVol * Nalpha *T
        return '%.4%%'% (100 * ParametricVaR)
    return ''


# In[293]:


def AskforParametricVaR():
    a = input('Do you want to compute the parametric VaR? ')
    if a == 'y':
        print(ParametricMeasureVaR(ProfitandLossD))
    else:
        print('No Parametric VaR required for this session.')
    return ''
print(AskforParametricVaR())


# Plot the Portfolio serues and profit and loss returns

# In[294]:


def PlotSeriesandPLD(SymbolList,Series,ProfitandLossD):
    i=0
    while i < len(SymbolList):
        FirstPlot = plt.plot(Series[i],'g')
        plt.xlabel('Observations')
        plt.ylabel('Log-Returns')
        plt.title(SymbolList[i][0])
        plt.grid(True)
        plt.show()
        print(FirstPlot)
        i+=1
    
    PortfolioPlot = plt.plot(ProfitandLossD,'r')
    plt.xlabel('Observations')
    plt.ylabel('Log-Returns')
    plt.title('P&L')
    plt.grid(True)
    plt.show()
    print(PortfolioPlot)
       


# In[295]:


PlotSeriesandPLD(SymbolList,Series,ProfitandLossD)


# Marginal VaR and CVaR

# In[296]:


def MarginalMeasures_s(SymbolList,PortfolioW,ListTotReturns):
    ticker = input('Which ticker do you want to delete? ')
    canceledticker= [ticker]
    SymbolList_s = [i for i in SymbolList if all(x not in canceledticker for x in i)]
    canceledweight = input('Which weight do you want to delete')
    canceledweight = int ( canceledweight )
    PortfolioW_s = PortfolioW
    ListTotReturns_s = ListTotReturns
    for i in range(len(PortfolioW_s)):
        if i==canceledweight:
            del PortfolioW[i]
            del ListTotReturns_s[i]
            print('correctly deleted','\n')
            ModPortfolioW_s = []
            for i in range(len(PortfolioW_s)):
                ModW = PortfolioW_s[i]/float(sum(PortfolioW_s))
                ModPortfolioW_s.append(ModW)
            break
        else:
            print('Searching')
    print('New SymbolList_s and ModPortfolioW_s are:')
    print(SymbolList_s,'\n',ModPortfolioW_s)
    ModList_s = []
    for n in range(len(ListTotReturns_s[0])):
        p=0
        while p < len(ListTotReturns_s):
            ModList_s.append([ListTotReturns_s[p][n]])
            p+=1
    ModListTotReturns_s=np.array(ModList_s,float)
    ModListTotReturns_s=ModListTotReturns_s.reshape(int(len(Listreturns)),int(len(SymbolList_s)))       
    Weights_s = np.array(ModPortfolioW_s,float)
    ProfitandLossD_s = np.dot(ModListTotReturns_s,Weights_s)
    PLDsorted_s = np.sort(ProfitandLossD_s)
    
    PLVaR0_s =PLDsorted_s[0]
    PLVaR1_s = PLDsorted_s[1]
    
    VaRs_s = [PLVaR0_s,PLVaR1_s]
    PLCVaR_s = statistics.mean(VaRs_s)
    ResultVaR = PLVaR1_s - PLVaR1
    ResultCVaR = PLCVaR_s - PLVaR1
    print('The Marginal VaR (99%) of '+ ticker + ' is ' + '%.3f%%' % (100 * ResultVaR))
    print('The Marginal CVaR (99%) of '+ ticker + ' is ' + '%.3f%%' % (100 * ResultCVaR))
    return ''
    
    
    
    
    


# In[297]:


MarginalMeasures_s(SymbolList,PortfolioW,ListTotReturns)


# In[ ]:





# In[ ]:




