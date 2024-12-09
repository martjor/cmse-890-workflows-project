import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

def plot_sir(df,*args,**kwargs):
    '''Plots the sir curves
    '''
    plt.plot('S',data=df)
    plt.plot('I',data=df)
    plt.plot('R',data=df)
        
qois_sir = {
    "peak" : lambda df: df['I'].max(),
    "time-to-peak" : lambda df: df['I'].idxmax(),
    "epidemic-size" : lambda df: df['R'].iloc[-1]
}