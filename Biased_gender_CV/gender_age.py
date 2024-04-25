# coding: cp1252
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
df_gender_t0 = pd.read_csv('data/gender_text.csv')
df_gender_v0 = pd.read_csv('data/gender_video.csv')


def objective_function(x, *args):
    '''
    x:  array-like
        parameters for estimation.
    *args: (hyper,s_type,subgroup, RT, R, filename). tuple-like, filename won't be used in CV 
        hyper: array-like, [b0,s0,dx,dt]
        s_type: bool, text=0,video=1.    
    '''
    
    hyper, s_type,  RT, R  = args
    v0 = x[0]
    b0 = hyper[0]
    ndt = 0#x[1]#x[2]
 
    x0 = x[1] # x0 = 0 for unbiased model
    s0 = hyper[1]
    dx = hyper[2]
    dt = hyper[3]
    
    # video and text data set has different max_time
    if s_type==0:
         max_time = 15.0
    else: 
         max_time = 8.0 
   
    bias = "point"#"centre"#
    method = "implicit"

    estimated_pdf = Model.test_solve_numerical(method, bias, v0, b0, s0, x0, dx, dt, max_time)
    res = Model_utility.calculate_LL(RT, R, estimated_pdf, dt, ndt,max_time)
    
    # Write the value of res to a file

    return res



def minimize_func(args):
    n_maxiter = 150
    initial_guess = np.random.uniform( -1, 1,2)#np.random.uniform( -1, 1,3)
    # random select initial value
  
    return minimize(objective_function,initial_guess, args=args, method="BFGS", options={'maxiter': n_maxiter})
    # ’ 'Nelder-Mead',BFGS


#####################################################
# random split training dataset and test dataset





def each_loop(bound, s0, df_gender_t, df_gender_v):
    '''
    for each loop, based on the given dataset and hyperparameter, return estimated parameter and test dataset -2LL
    ------------------------------
    bound: double
        the decision threshold.
    s0: double
        the noise standard deviation
     df_gender_t: dataframe
     df_gender_v: dataframe
    '''
    start = timer()
    # define hyper parameter set
    hyper = [bound, s0, 0.01,0.01]
    
    # generate training and test dataset
    '''
    df_gender_train_t, df_gender_test_t = Model_utility.df_draw(df_gender_t)
    df_gender_train_t = df_gender_t
    df_gender_test_t = df_gender_t
    '''

    df_gender_train_v, df_gender_test_v = Model_utility.df_draw(df_gender_v)
    #sex = df_gender_train_t.iloc[1,-3] 
    #age = df_gender_train_t.iloc[1,-2]
    age = df_gender_train_v.iloc[1,-2] 

    #df_gender_train_v, df_gender_test_v=Model_utility.df_draw(df_gender_v0)
    '''
    RT_gender_train_t = df_gender_train_t['RT'].to_numpy()
    R_gender_train_t  = df_gender_train_t['R'].to_numpy()
    RT_gender_test_t = df_gender_test_t['RT'].to_numpy()
    R_gender_test_t  = df_gender_test_t['R'].to_numpy()

    '''
    
    RT_gender_train_v = df_gender_train_v['RT1'].to_numpy()
    #RT_gender_train_v = df_gender_train_v['RT'].to_numpy()
    R_gender_train_v  = df_gender_train_v['R'].to_numpy()
    #RT_gender_test_v = df_gender_test_v['RT'].to_numpy()
    RT_gender_test_v = df_gender_test_v['RT1'].to_numpy()
    R_gender_test_v  = df_gender_test_v['R'].to_numpy()
    
    # arugment for lieklihood setting
    s_type_t = 0 # s_type = 0 is text-based data, otherwise = 1
    s_type_v = 1
  


    #res0=minimize_func((hyper,s_type_t, RT_gender_train_t, R_gender_train_t))
    res1=minimize_func((hyper,s_type_v, RT_gender_train_v, R_gender_train_v))
    
    # estimated parameter 
    dx =hyper[2]
    dt = hyper[3]
    bound0 = hyper[0]
    noise0 = hyper[1]
    '''
    drift0_t = res0.x[0]
    ndt_t = 0#res0.x[-1]#res0.x[2] 
    x0_t =  res0.x[1] 
    '''
    
    drift0_v = res1.x[0]
    ndt_v = 0#res1.x[1]#res1.x[2]
    x0_v = res1.x[1] 
    


    max_time_t = 15.0
    max_time_v = 8.0

    bias = "point"#"centre"#
    method = "implicit"
    
    '''
    result1 = Model.test_solve_numerical(method, bias, drift0_t, bound0, noise0, x0_t, dx, dt, max_time_t)
    test_LL = Model_utility.calculate_LL(RT_gender_test_t, R_gender_test_t, result1, dt, ndt_t, max_time_t)/len(R_gender_test_t)
    
    est_prob = Model_utility.prob_estimated(result1)
    obs_prob = Model_utility.prob_obs(df_gender_train_t)
    test_prob = Model_utility.prob_obs(df_gender_test_t)
    res_final = [age]
    res_final.extend(hyper)
    res_final.append(res0.x[0])
    #res_final.append(res0.x[1])
    res_final.append(np.exp(res0.x[-1]))
    res_final.append(res0.fun/len(R_gender_train_t))
    res_final.append(test_LL)
    res_final.extend(est_prob)
    res_final.extend(obs_prob)
    res_final.extend(test_prob)
    '''
    
    result_v = Model.test_solve_numerical(method, bias, drift0_v, bound0, noise0, x0_v, dx, dt, max_time_v)
    test_LL_v = Model_utility.calculate_LL(RT_gender_test_v, R_gender_test_v, result_v, dt, ndt_v, max_time_v)/len(R_gender_test_v)
    
    est_prob_v = Model_utility.prob_estimated(result_v)
    obs_prob_v = Model_utility.prob_obs(df_gender_train_v)
    test_prob_v = Model_utility.prob_obs(df_gender_test_v)
    res_final_v = [age]
    res_final_v.extend(hyper)
    res_final_v.extend(res1.x)
    #res_final_v.append(res1.x[0])
    #res_final_v.append(np.exp(res1.x[-1]))
    #res_final_v.append(np.exp(res1.x[2]))
    res_final_v.append(res1.fun/len(R_gender_train_v))
    res_final_v.append(test_LL_v)
    res_final_v.extend(est_prob_v)
    res_final_v.extend(obs_prob_v)
    res_final_v.extend(test_prob_v)
    
    
    # write result
    '''
    with open('data/estimate_gender_text_sageX0.txt', 'a') as f:
        f.write( '  '.join(map(str, res_final)) + '\n')    
    '''       

    with open('data/estimate_gender_video_sage_ndt.txt', 'a') as f:
        f.write( '  '.join(map(str, res_final_v)) + '\n')
    
    print("1 round finished.")
    end =  timer()
    print(end - start)
   
''' for hyperparameter searching
bound_list = [5.5,6]
for i in bound_list:
    for j in range(10):
        each_loop(i, i*0.5)
'''       


start0 = timer()
bound = 5.5
s0 = 2.75
# divide dataset by sub_age, sex = 0 is Young. sex = 1 is Old

grouped_t = df_gender_t0.groupby('age')
df_gender_t0_Y = grouped_t.get_group(0)
df_gender_t0_O = grouped_t.get_group(1)

grouped_v = df_gender_v0.groupby('age')
df_gender_v0_Y = grouped_v.get_group(0)
df_gender_v0_O = grouped_v.get_group(1)

#print(df_gender_t0_F.head())

#each_loop(bound, s0, df_gender_t0_F, df_gender_v0_F)
#each_loop(bound, s0, df_gender_t0, df_gender_v0_F)
'''
for i in range(2):
    Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = grouped_t.get_group(i),df_gender_v = df_gender_v0_O) for _ in range(30))    
'''




for i in range(2):
    Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = df_gender_t0_O,df_gender_v = grouped_v.get_group(i)) for _ in range(30))    


end0 =  timer()

print(end0 - start0)

# 121655sec for 3*3*10=90, 22.5 min a run.

# check why the resample has the same -2LL value.
#each_loop(4,2)
# parallel algorithm





