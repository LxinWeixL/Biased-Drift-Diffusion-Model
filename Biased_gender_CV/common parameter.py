# coding: cp1252
from asyncio import Condition

from numpy.linalg import cond
import Model
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from IPython.display import Image
import Model_utility
import os
from multiprocessing import Pool
from timeit import default_timer as timer
from joblib import Parallel, delayed






#########################################
os.chdir('C:/Users/e0729936/source/repos/Recovery_drift_bias_ndt/Biased_gender_CV')
###########################################

# load the whole dataset
df_gender_t0 = pd.read_csv('data/newdata/gender_text.csv')
df_gender_v0 = pd.read_csv('data/newdata/gender_video.csv')


def objective_function(x, *args):
    '''
    x:  array-like
        parameters for estimation.
    *args: (hyper,s_type,subgroup, RT, R, filename). tuple-like, filename won't be used in CV 
        hyper: array-like, [b0,s0,dx,dt]
        s_type: bool, text=0,video=1.   
        condition: int type:
            condition = 0, ndt is the same, but x0 is free.
            condition = 1, ndt is the same, xo is the same but unnecessary to be 0.
            condition = 2, ndt is the same, but x0 = 0.
            condition = 3, ndt = 0, x0 = 0 for group 0, x0 is free for group 1.
            condition = 4, ndt = 0, x0 is free for group 0, x0 = 0 for group 1.
    '''
    
    hyper, s_type, condition, RT_0, R_0, RT_1, R_1  = args
    # with common ndt and common x0, but different v0
    v0_0 = x[0]
    b0 = hyper[0]
    s0 = hyper[1]
    dx = hyper[2]
    dt = hyper[3]
    # video and text data set has different max_time
    if s_type==0:
         max_time = 18.0
    else: 
         max_time = 8.0
    method = "implicit"
    

    if condition ==0:
        v0_1 = x[3]
        x0_0 = x[1]
        x0_1 = x[4]
        ndt = x[2]
        bias_0 = "point"
        bias_1 = "point"
    elif condition==1: 
        v0_1 = x[3]
        x0_0 = x[1]
        x0_1 = x[1]
        ndt = x[2]
        bias_0 = "point"
        bias_1 = "point"
    elif condition==2:
        v0_1 = x[2]
        x0_0 = 0
        x0_1 = 0
        ndt = x[1]
        bias_0 = "centre"
        bias_1 = "centre"
    elif condition ==3:
        v0_1 = x[1]
        x0_0 = 0
        x0_1 = x[2]
        ndt = -np.Inf
        bias_0 = "centre"
        bias_1 = "point"
        
    else:
        # condition ==4
        v0_1 = x[2]
        x0_0 = x[1]
        x0_1 = 0
        ndt =  -np.Inf
        bias_0 = "point"
        bias_1 = "centre"


   
    estimated_pdf_0 = Model.test_solve_numerical(method, bias_0, v0_0, b0, s0, x0_0, dx, dt, max_time)
    estimated_pdf_1 = Model.test_solve_numerical(method, bias_1, v0_1, b0, s0, x0_1, dx, dt, max_time)
    res_0 = Model_utility.calculate_LL(RT_0, R_0, estimated_pdf_0, dt, ndt,max_time)
    res_1 = Model_utility.calculate_LL(RT_1, R_1, estimated_pdf_1, dt, ndt,max_time)

    # Write the value of res to a file

    LL_sum = res_0+res_1
    return LL_sum



def minimize_func(args):
    n_maxiter = 150
    hyper, s_type, condition, RT_0, R_0, RT_1, R_1  = args
    # random select initial value
    if condition ==0:
        initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -0.2, 0.2,1),np.random.uniform( -1, 0,1),
                         np.random.uniform( -1, 1,1),np.random.uniform( -0.2, 0.2,1)]
        initial_guess = np.array(initial_guess).reshape(5,)
    elif condition ==1:
        initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -0.2, 0.2,1),np.random.uniform( -1, 0,1),
                         np.random.uniform( -1, 1,1)]
        initial_guess = np.array(initial_guess).reshape(4,)
    elif condition ==2:  
        initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -1, 0,1),np.random.uniform( -1, 1,1)]
        initial_guess = np.array(initial_guess).reshape(3,)
    elif condition ==3:
        initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -1, 1,1),np.random.uniform(-0.2, 0.2,1)]
        initial_guess = np.array(initial_guess).reshape(3,)
    else:
        initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform(-0.2, 0.2,1),np.random.uniform( -1, 1,1)]
        initial_guess = np.array(initial_guess).reshape(3,)
    
  
    return minimize(objective_function,initial_guess, args=args, method="BFGS", options={'maxiter': n_maxiter})
    # ’ 'Nelder-Mead',BFGS


#####################################################
# random split training dataset and test dataset





def each_loop(bound, s0, df_gender_t, df_gender_v,condition,para):
    '''
    for each loop, based on the given dataset and hyperparameter, return estimated parameter and test dataset -2LL
    ------------------------------
    bound: double
        the decision threshold.
    s0: double
        the noise standard deviation
     df_gender_t: dataframe
     df_gender_v: dataframe
     condition: int type
     para: string.
    '''
    start = timer()
    # define hyper parameter set
    hyper = [bound, s0, 0.01,0.01]
    
    # generate training and test dataset

    
    '''
    
    grouped_t = df_gender_t.groupby(para)
    #df_gender_train_t_0, df_gender_test_t_0 = Model_utility.df_draw(grouped_t.get_group(0))
    #df_gender_train_t_1, df_gender_test_t_1 = Model_utility.df_draw(grouped_t.get_group(1))
    
    df_gender_train_t_0 = grouped_t.get_group(0)
    df_gender_test_t_0 = df_gender_train_t_0 
    df_gender_train_t_1 = grouped_t.get_group(1)
    df_gender_test_t_1 = df_gender_train_t_1
    '''
    
    grouped_v = df_gender_v.groupby(para)
    #df_gender_train_v_0, df_gender_test_v_0 = Model_utility.df_draw(grouped_v.get_group(0))
    #df_gender_train_v_1, df_gender_test_v_1 = Model_utility.df_draw(grouped_v.get_group(1))

    df_gender_train_v_0 = grouped_v.get_group(0)
    df_gender_test_v_0 = df_gender_train_v_0 
    df_gender_train_v_1 = grouped_v.get_group(1)
    df_gender_test_v_1 = df_gender_train_v_1
    
    
    '''
    RT_gender_train_t_0 = df_gender_train_t_0['RT1'].to_numpy()
    R_gender_train_t_0  = df_gender_train_t_0['R'].to_numpy()
    RT_gender_test_t_0 = df_gender_test_t_0['RT1'].to_numpy()
    R_gender_test_t_0  = df_gender_test_t_0['R'].to_numpy()
    RT_gender_train_t_1 = df_gender_train_t_1['RT1'].to_numpy()
    R_gender_train_t_1  = df_gender_train_t_1['R'].to_numpy()
    RT_gender_test_t_1 = df_gender_test_t_1['RT1'].to_numpy()
    R_gender_test_t_1 = df_gender_test_t_1['R'].to_numpy()
    
    '''
    
    RT_gender_train_v_0 = df_gender_train_v_0['RT1'].to_numpy()
    R_gender_train_v_0  = df_gender_train_v_0['R'].to_numpy()
    RT_gender_test_v_0 = df_gender_test_v_0['RT1'].to_numpy()
    R_gender_test_v_0  = df_gender_test_v_0['R'].to_numpy()
    RT_gender_train_v_1 = df_gender_train_v_1['RT1'].to_numpy()
    R_gender_train_v_1  = df_gender_train_v_1['R'].to_numpy()
    RT_gender_test_v_1 = df_gender_test_v_1['RT1'].to_numpy()
    R_gender_test_v_1  = df_gender_test_v_1['R'].to_numpy()
    
    # arugment for lieklihood setting
    s_type_t = 0 # s_type = 0 is text-based data, otherwise = 1
    s_type_v = 1
  


    #res0=minimize_func((hyper,s_type_t,condition, RT_gender_train_t_0, R_gender_train_t_0, RT_gender_train_t_1, R_gender_train_t_1))
    res1=minimize_func((hyper,s_type_v,condition, RT_gender_train_v_0, R_gender_train_v_0, RT_gender_train_v_1, R_gender_train_v_1))
    
    # estimated parameter 
    dx =hyper[2]
    dt = hyper[3]
    bound0 = hyper[0]
    noise0 = hyper[1]
    
    max_time_t = 18.0
    max_time_v = 8.0
    method = "implicit"
    


    '''
    drift0_t_0 = res0.x[0]
    if condition==0:
        x0_t_0 = res0.x[1]
        ndt_t = res0.x[2]
        drift0_t_1 = res0.x[3]
        x0_t_1 = res0.x[4]
        bias_0 =  "point"
        bias_1 =  "point"
    elif condition==1:
        x0_t_0 = res0.x[1]
        ndt_t = res0.x[2]
        drift0_t_1 = res0.x[3]
        x0_t_1 =  res0.x[1]
        bias_0 =  "point"
        bias_1 =  "point"
        
    elif condition==2:
        x0_t_0 = 0
        ndt_t = res0.x[1]
        drift0_t_1 = res0.x[2]
        x0_t_1 =  0
        bias_0 =  "centre"
        bias_1 =  "centre"
    elif condition==3:
        x0_t_0 = 0
        ndt_t = -np.Inf
        drift0_t_1 =  res0.x[1]
        x0_t_1 =  res0.x[2]
        bias_0 =  "centre"
        bias_1 =  "point"
    else:
        x0_t_0 = res0.x[1]
        ndt_t = -np.Inf
        drift0_t_1 =  res0.x[2]
        x0_t_1 =  0
        bias_0 =  "point" 
        bias_1 =  "centre"
        


    '''
    
    drift0_v_0 = res1.x[0]
    if condition==0:
        x0_v_0 = res1.x[1]
        ndt_v = res1.x[2]
        drift0_v_1 = res1.x[3]
        x0_v_1 = res1.x[4]
        bias_0 =  "point"
        bias_1 =  "point"
    elif condition==1:
        x0_v_0 = res1.x[1]
        ndt_v = res1.x[2]
        drift0_vt_1 = res1.x[3]
        x0_v_1 =  res1.x[1]
        bias_0 =  "point"
        bias_1 =  "point"
        
    elif condition==2:
        x0_v_0 = 0
        ndt_v = res1.x[1]
        drift0_v_1 = res1.x[2]
        x0_v_1 =  0
        bias_0 =  "centre"
        bias_1 =  "centre"
    elif condition==3:
        x0_v_0 = 0
        ndt_v = -np.Inf
        drift0_v_1 =  res1.x[1]
        x0_v_1 =  res1.x[2]
        bias_0 =  "centre"
        bias_1 =  "point"
    else:
        x0_v_0 = res1.x[1]
        ndt_v = -np.Inf
        drift0_v_1 =  res1.x[2]
        x0_v_1 =  0
        bias_0 =  "point" 
        bias_1 =  "centre"
    

    '''
    result_t_0 = Model.test_solve_numerical(method, bias_0, drift0_t_0, bound0, noise0, x0_t_0, dx, dt, max_time_t)
    result_t_1 = Model.test_solve_numerical(method, bias_1, drift0_t_1, bound0, noise0, x0_t_1, dx, dt, max_time_t)
    
    test_LL = (Model_utility.calculate_LL(RT_gender_test_t_0, R_gender_test_t_0, result_t_0, dt, ndt_t, max_time_t) +Model_utility.calculate_LL(RT_gender_test_t_1, R_gender_test_t_1, result_t_1, dt, ndt_t, max_time_t))/(len(R_gender_test_t_0)+len(R_gender_test_t_1))
    
    est_prob_0 = Model_utility.prob_estimated(result_t_0)
    obs_prob_0 = Model_utility.prob_obs(df_gender_train_t_0)
    test_prob_0 = Model_utility.prob_obs(df_gender_test_t_0)
    est_prob_1 = Model_utility.prob_estimated(result_t_1)
    obs_prob_1 = Model_utility.prob_obs(df_gender_train_t_1)
    test_prob_1 = Model_utility.prob_obs(df_gender_test_t_1) 
    
    '''

    result_v_0 = Model.test_solve_numerical(method, bias_0, drift0_v_0, bound0, noise0, x0_v_0, dx, dt, max_time_v)
    result_v_1 = Model.test_solve_numerical(method, bias_1, drift0_v_1, bound0, noise0, x0_v_1, dx, dt, max_time_v)
    
    test_LL = (Model_utility.calculate_LL(RT_gender_test_v_0, R_gender_test_v_0, result_v_0, dt, ndt_v, max_time_v) +Model_utility.calculate_LL(RT_gender_test_v_1, R_gender_test_v_1, result_v_1, dt, ndt_v, max_time_v))/(len(R_gender_test_v_0)+len(R_gender_test_v_1))
    est_prob_0 = Model_utility.prob_estimated(result_v_0)
    obs_prob_0 = Model_utility.prob_obs(df_gender_train_v_0)
    test_prob_0 = Model_utility.prob_obs(df_gender_test_v_0)
    est_prob_1 = Model_utility.prob_estimated(result_v_1)
    obs_prob_1 = Model_utility.prob_obs(df_gender_train_v_1)
    test_prob_1 = Model_utility.prob_obs(df_gender_test_v_1) 
    
    
    res_final = []
    res_final.extend(hyper)
    '''
    tmp = res0.x
    # log_ndt transfer to exp(.)
    if (condition==1)|(condition==0) :
        
        tmp[2] = np.exp(res0.x[2])
    else:
         tmp[1] = np.exp(res0.x[1])
        
    res_final.extend(tmp)
    
    res_final.append(res0.fun/(len(R_gender_train_t_0)+len(R_gender_train_t_1)))
    
    '''
    tmp = res1.x
    # log_ndt transfer to exp(.)
    if (condition==1)|(condition==0):
        
        tmp[2] = np.exp(res1.x[2])
    else:
         tmp[1] = np.exp(res1.x[1])
        
    res_final.extend(tmp)
    res_final.append(res1.fun/(len(R_gender_train_v_0)+len(R_gender_train_v_1)))
     
    res_final.append(test_LL)
    res_final.extend(est_prob_0)
    res_final.extend(obs_prob_0)
    res_final.extend(test_prob_0)
    res_final.extend(est_prob_1)
    res_final.extend(obs_prob_1)
    res_final.extend(test_prob_1)
   
    
    
    # write result
    '''
    file_route = 'data/newdata/estimate_gender_text_aggregated_' +  para +'.txt'
    with open(file_route, 'a') as f:
        f.write( '  '.join(map(str, res_final)) + '\n')    
    '''       
    file_route = 'data/newdata/estimate_gender_video_aggregated_' +  para +'.txt'
    with open(file_route, 'a') as f:
        f.write( '  '.join(map(str, res_final)) + '\n')
    
    print("1 round finished.")
    end =  timer()
    print(end - start)


start0 = timer()
bound = 5#2.5#
s0 = 2.5#1.75#
# divide dataset by sub_gender, sex = 0 is M.

#grouped_t = df_gender_t0.groupby('sex')
#grouped_t = df_gender_t0.groupby('age')
#grouped_t = df_gender_t0.groupby('theta')
#df_gender_t0_M = grouped_t.get_group(0)
#df_gender_t0_F = grouped_t.get_group(1)

#grouped_v = df_gender_v0.groupby('sex')
#grouped_v = df_gender_v0.groupby('age')
#df_gender_v0_M = grouped_v.get_group(0)
#df_gender_v0_F = grouped_v.get_group(1)

#print(df_gender_t0_F.head())

#each_loop(bound, s0, df_gender_t0_F, df_gender_v0_F)
#each_loop(bound, s0, df_gender_t0, df_gender_v0_F)


'''
for i in range(2):
    Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = grouped_t.get_group(i),df_gender_v = df_gender_v0_F) for _ in range(30))    


for i in range(2):
   Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = df_gender_t0_F,df_gender_v = grouped_v.get_group(i)) for _ in range(30))    
'''



para = ["sex","theta","alpha","age"]

condition_list = [3,4]
for k in para:
    
    Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = df_gender_t0,df_gender_v = df_gender_v0,condition = 0, para = k) for _ in range(30))    

#test1 = Model_utility.df_draw2(df_gender_t0)

#print(test1)

#test2 = Model_utility.df_draw2(df_gender_t0)

#print(test2)

#Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = df_gender_t0,df_gender_v = df_gender_v0) for _ in range(30))

end0 =  timer()
#print(len(df_gender_t0_M["RT"].to_numpy()))
#print(len(df_gender_t0_F["RT"].to_numpy()))
print(end0 - start0)

# 121655sec for 3*3*10=90, 22.5 min a run.

# check why the resample has the same -2LL value.
#each_loop(4,2)
# parallel algorithm





