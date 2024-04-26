from distutils.dist import DistributionMetadata
from math import floor
from operator import index
from pickle import TRUE
import random
import numpy as np
from numpy.random import RandomState
import pandas as pd
import matplotlib.pyplot as plt

def plot_density(input, dt, ndt):
    """
    Plots the probability density of the output from the solve_numerical function.

    Parameters:
    -----------
    input : list
        A list of three arrays containing the data to be plotted.

    Returns:
    --------
    None
        The function only plots the data.

    Example:
    --------
    >>> plot_density(object)
    """
    
    ndt_step = int(ndt/dt)
    ndt_zeros = np.zeros(ndt_step) 
    nO = len(input[0])
    nX = len(input[1])
    nU = len(input[2])
    y0 = np.concatenate((ndt_zeros, input[0]))
    y1 = np.concatenate((ndt_zeros, input[1]))
    y2 = input[2]
    x0 = np.arange(0, nO+ndt_step, 1)
    x1 = np.arange(0, nX+ndt_step, 1)
    x2 = np.arange(0, nU, 1)
    ymax = max([y0.max(), y1.max()])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ax[0].plot(x0, y0, color='red', marker='o')
    ax[0].set_title('Positive boundary')
    ax[0].set_xlabel('Time step')
    ax[0].set_ylabel('Probability density')
    ax[0].set_ylim([0, ymax])
    ax[0].grid()
    ax[1].plot(x1, y1, color ='red', marker ='o')
    ax[1].set_title('Negative boundary')
    ax[1].set_xlabel('Time step')
    ax[1].set_ylim([0, ymax])
    ax[1].grid()
    
    ax[2].plot(y2, x2, color ='red', marker ='o')
    ax[2].set_title('The remaining density \nat the last time step')
    ax[2].set_ylabel('Evidence step')
    ax[2].grid()
    return fig
    
def calculate_cdf(input):
    """
    Calculates the cumulative distribution function of the input data.

    Parameters:
    -----------
    input : list
        A list of three arrays containing the data to be plotted.

    Returns:
    --------
    array-like
        An array containing the cumulative distribution function of the input data.

    Example:
    --------
    >>> calculate_cdf(input)
    """
    y0 = input[0]
    y1 = input[1]
    y = np.column_stack((y0, y1))
    yy = y.sum(axis=1)
    cdf = np.cumsum(yy)
    return cdf

def calculate_LL(s_type, RT, R, estimated_pdf, dt, ndt, max_time, print_warning = False):
    """
    Calculates the negative 2, log-likelihood of the input data.

    Parameters:
    -----------
    s_type: bool'
        s _type = 0 , if the senario is text-based.
        s _type = 1 , if the senario is video-based.
    RT : array-like
        An array containing the reaction times.
    R : array-like
        An array containing the responses.
    estimated_pdf : list
        A list of two arrays containing the estimated probability density functions.
    dt : float
        The time step.
    ndt : float
        non-decision time.
    max_time: float
        max_time for response time.
    print_warning : bool, optional
        If True, prints a warning message when likelihood contains 0. Default is False.

    Returns:
    --------
    float
        The log-likelihood of the input data.

    Example:
    --------
    >>> calculate_LL(RT, R, [pdf_0, pdf_1], 0.1)
    """



    n = len(RT)
    ndt =  np.exp(ndt)
    DT = RT-ndt
    
    
   
               
    index = DT/dt
    likelihood = np.zeros(n)
    '''
    for each log-ll, a log(P_U)<0 is minused, which in fact adding a constant. The larger log(P_U) is, the smaller the constant is add
    When inverse the log-ll to get -2*ll, the larger P_U is, the smaller constant is minused, hence the -2NLL is harder to minimize. Hence, P_U is a penality here. 
    '''
    for i in range(n):
        if R[i] == 0:
            probability = estimated_pdf[0]/(1-np.sum(estimated_pdf[2]))
        else:
            probability = estimated_pdf[1]/(1-np.sum(estimated_pdf[2]))
            
        
        # used to be int(index[i])  
        int_index = floor(index[i])
        
        # make sure RT-ndt positive
        if int_index<=0: 
           likelihood[i] = 0
        else:
            likelihood[i] = probability[int_index]
        

        likelihood[i] = probability[int_index]

    # consider the boundary the ndt.
    idx_0 = np.where(likelihood == 0)
    if (len(idx_0) > 0): 
        if print_warning: 
            print("Warning: likelihood contains 0")
        likelihood[idx_0] = [1e-30]*len(idx_0)
    
    LL = np.log(likelihood)
    return LL.sum()*(-2)



def prob_estimated(s_type,dt,input):
    """
    return: list of probs based on estimated parameters
    """
    
    result = input
    
    total_probability = np.sum(result[0]) + np.sum(result[1]) + np.sum(result[2])
    positive_probability = np.sum(result[0]) 
    negative_probability = np.sum(result[1]) 
    remaining_probability = np.sum(result[2])
    

    res = [round(total_probability, 3), round(positive_probability, 3), round(negative_probability, 3),round(remaining_probability, 3)]
    return res

def prob_obs(df):
    """
    return: list of observed choice proportion based on df
    """
    #proportions = df['TrueR'].value_counts(normalize=True)
    proportions = df['R'].value_counts(normalize=True)
    res = [round(proportions[0], 3),round(proportions[1], 3)]
    return res


def simulate(input, df, dt, ndt, max_time,s_type):
    result = input
    ndt = np.exp(ndt)
    
    total_probability = np.sum(result[0]) + np.sum(result[1]) + np.sum(result[2])
    positive_probability = np.sum(result[0]) 
    negative_probability = np.sum(result[1]) 
    remaining_probability = np.sum(result[2]) 
    print("Total probability: ", total_probability)
    print("Probability of hitting the positive boundary", round(positive_probability, 3))
    print("Probability of hitting the negative boundary", round(negative_probability, 3))
    print("Remaining probability at the last time step", round(remaining_probability, 3))

 
    proportions = df['R'].value_counts(normalize=True)
    print("The data showed the probability of hitting the positive boundary", round(proportions[0], 3))
    print("The data showed the probability of hitting the negative boundary", round(proportions[1], 3))
    
    y0 = result[0]
    y1 = result[1]
    y = np.column_stack((y0, y1))
    yy = y.sum(axis=1)
    cdf = np.cumsum(yy)
    print(cdf.shape)
    n = len(df)
    high_value = 1
    
 
    random_draws = np.random.uniform(low=0.0, high=high_value, size=n)
    index = np.searchsorted(cdf, random_draws)
    
    # Correct the discretization error in the random draws
    for i in range(n):
        if random_draws[i] > cdf.max():
            index[i] = index[i] - 1
    

    ## Define RT based on whether close to stopline
    RT = dt * index +ndt

    p = y0[index]
    q = y1[index]
    R = np.zeros(n)
    for i in range(n):
        pp = p[i] / (p[i]+q[i])
        qq = q[i] / (p[i]+q[i])
        prob = np.array( [pp/(pp+qq), qq/(pp+qq)]).reshape(2,)
        R[i] = np.random.choice([0, 1], size=1, p = prob)[0]
    
    return [R, RT]

def simulate_simple(input, n, dt, ndt):
    result = input
    
    cdf = calculate_cdf(input)
    random_draws = np.random.uniform(low=0.0, high=1.0, size=n)
    index = np.searchsorted(cdf, random_draws)
    
    # Correct the discretization error in the random draws
    for i in range(n):
        if random_draws[i] > cdf.max():
            index[i] = index[i] - 1
    RT = dt * index + ndt             
    
    y0 = input[0]
    y1 = input[1]
    p = y0[index]
    q = y1[index]
    R = np.zeros(n)
    for i in range(n):
        pp = p[i] / (p[i]+q[i])
        qq = q[i] / (p[i]+q[i])
        prob = np.array( [pp/(pp+qq), qq/(pp+qq)]).reshape(2,)
        R[i] = np.random.choice([0, 1], size=1, p = prob)[0]
    
    return [R, RT]


def df_draw (inputdf):
    '''
    return two dataframe:train df and test df, test df contains a random trail for each subject
    -----------------------------------------
    inputdf: dataframe, {"s","RT","R","sex","age"}, where "s" is the subject ID.
    '''
    df = inputdf
    #test = df.groupby("s").sample(n=1, random_state=1)
    test = df.sample(frac = 1.0).groupby('s').head(1)
    train = df.drop(index = test.index)
    return (train,test)
   

def df_draw2 (inputdf):
    '''
    return a resample dataset for the original one. For each subject, 6 trials is sampled.
    -----------------------------------------
    inputdf: dataframe, {"s","RT","R","sex","age"}, where "s" is the subject ID.
    '''
    df = inputdf
    # for pure gender condition n = 6, otherwise n = 18
    df_new = df.groupby("s").sample(n= 6, replace = True)
    
 
    return  df_new


def df_draw3 (inputdf):
    '''
    return one dataframe: contains 6 trail for each subject.
    For those whose valid trail number isn't sufficient, will repreat part of them.
    -----------------------------------------
    inputdf: dataframe, {"s","RT","R","sex","age"}, where "s" is the subject ID.
    '''
    df = pd.DataFrame(inputdf)
    #test = df.groupby("s").sample(n=1, random_state=1)
    group_type = df.groupby("s")
    group_num = df["s"].unique()
    
    for i in group_num:
        df_per = group_type.get_group(i)
  
        if df_per.shape[0]!=6:
            test = df_per.sample(n =6-df_per.shape[0], replace = True).head()
            
            df = pd.concat([df,test])
            
 
    return df
