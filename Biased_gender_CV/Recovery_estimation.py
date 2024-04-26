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


df_gender_t0 = pd.read_csv('data/gender_text_copy.csv')
df_gender_v0 = pd.read_csv('data/gender_video_copy.csv')

def objective_function(x, *args):
    '''
    x:  array-like
        parameters for estimation.
    *args: (hyper,s_type,condition, subgroup, RT, R, filename). tuple-like, filename won't be used in CV 
        hyper: array-like, [b0,s0,dx,dt]
        s_type: bool, text=0,video=1.   
        condition = 0, three free parameter
        condition = 1, x0 =0
        conidtion =2, ndt = 0
        condition = 3, only drift rate is free., x0 =0, ndt = 0.
    '''
    
    hyper, s_type, condition, RT, R  = args
    
    v0 = x[0]
    dx = hyper[2]
    dt = hyper[3]
    b0 = hyper[0]
    s0 = hyper[1]
    method = "implicit"
    
    # video and text data set has different max_time
    if s_type==0:
         max_time = 15 
         b0 = hyper[0]
         s0 = hyper[1]
    else: 
         max_time = 8.0     
         b0 = hyper[4]
         s0 = hyper[5]
    

    if condition == 0 :
        ndt = x[2]
        x0 = x[1]
        bias = "point"
    elif condition ==1:
        ndt = x[1]
        x0 = 0
        bias = "centre"
    elif condition==2:
        ndt = -np.Inf
        x0 = x[1]
        bias = "point"
    else:
        ndt = -np.Inf
        x0 = 0
        bias = "centre"

    estimated_pdf = Model.test_solve_numerical(method, bias, v0, b0, s0, x0, dx, dt, max_time)
    res = Model_utility.calculate_LL(s_type, RT, R, estimated_pdf, dt, ndt,max_time)
    
    # Write the value of res to a file

    return res



def minimize_func(args):
    hyper, s_type, condition, RT, R  = args
    
    if s_type:
        limit = -0.2
    else:
        limit = 0.7
 
    if condition==0:
        #np.random.uniform( -0.2, 0.2,1)
        initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -0.2, 0.2,1),np.random.uniform( -1,limit,1)]#,np.random.uniform( -1, 1,1)]
        initial_guess = np.array(initial_guess).reshape(3,)
    elif condition==1:
        initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -1, limit,1)]
        initial_guess = np.array(initial_guess).reshape(2,)
    elif condition==2:
        initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -0.2, 0.2,1)]
        initial_guess = np.array(initial_guess).reshape(2,)
    else:
        initial_guess = np.random.uniform( -1, 1,1)
        initial_guess = np.array(initial_guess).reshape(1,)
       
    # random select initial value
  
    return minimize(objective_function,initial_guess, args=args, method="Nelder-Mead")
    


#####################################################
def each_loop(bound_t, s0_t,bound_v, s0_v, df_gender_t, df_gender_v,condition):
    '''
    for each loop, based on the given dataset and hyperparameter, return estimated parameter and test dataset -2LL
    ------------------------------
    bound: double
        the decision threshold.
    s0: double
        the noise standard deviation
     df_gender_t: dataframe
     df_gender_v: dataframe
    
     condition: an int type. 
        condition = 0, three free parameters; 
        condition = 1, x0 =0;
        condition = 2, ndt = 0; 
        condition = 3, only drift is free.
    '''
    
    start = timer()
    # define hyper parameter set
    hyper = [bound_t, s0_t,  0.01,0.01, bound_v, s0_v]
    
    
    df_gender_train_t = Model_utility.df_draw3(df_gender_t)
    df_gender_train_v = Model_utility.df_draw3(df_gender_v)
  
    df_gender_test_v = df_gender_train_v
    df_gender_test_t = df_gender_train_t

    
    
    
    
    RT_gender_train_t = df_gender_train_t['RT'].to_numpy()
    R_gender_train_t  = df_gender_train_t['R'].to_numpy()
    RT_gender_test_t = df_gender_test_t['RT'].to_numpy()
    R_gender_test_t  = df_gender_test_t['R'].to_numpy()

    
    
    RT_gender_train_v = df_gender_train_v['RT1'].to_numpy()
    R_gender_train_v  = df_gender_train_v['R'].to_numpy()
    RT_gender_test_v = df_gender_test_v['RT1'].to_numpy()
    R_gender_test_v  = df_gender_test_v['R'].to_numpy()
    
    # argument for likelihood setting
    s_type_t = 0 # s_type = 0 is text-based data, otherwise = 1
    s_type_v = 1
  


    res0=minimize_func((hyper,s_type_t, condition, RT_gender_train_t, R_gender_train_t))
    res1=minimize_func((hyper,s_type_v,condition, RT_gender_train_v, R_gender_train_v))
    
    # estimated parameter 
    dx =hyper[2]
    dt = hyper[3]
    bound0_t = hyper[0]
    noise0_t = hyper[1]
    
    bound0_v = hyper[4]
    noise0_v = hyper[5]
    method = "implicit"
    
    
    drift0_t = res0.x[0]
    
    if condition==0:
        ndt_t = res0.x[2]
        x0_t =  res0.x[1]
        bias =  "point"
    elif condition==1:
        ndt_t = res0.x[1]
        x0_t = 0
        bias = "centre"
    elif condition==2:
        ndt_t = -np.Inf
        x0_t =  res0.x[1]
        bias =  "point"
    else:
        bias = "centre"
    
    
    
    drift0_v = res1.x[0]
    
    if condition==0:
        ndt_v = res1.x[2]
        x0_v =  res1.x[1]
        bias =  "point"
    elif condition==1:
        ndt_v = res1.x[1]
        x0_v = 0
        bias = "centre"
    elif condition==2:
        ndt_v = -np.Inf
        x0_v =  res1.x[1]
        bias =  "point"
    else:
        bias = "centre"
        ndt_v = -np.Inf
        x0_v =  0
    
    
    
    


    max_time_t = 15.0
    max_time_v = 8.0

    
    
    result1 = Model.test_solve_numerical(method, bias, drift0_t, bound0_t, noise0_t, x0_t, dx, dt, max_time_t)
    test_LL = Model_utility.calculate_LL(s_type_t,RT_gender_test_t, R_gender_test_t, result1, dt, ndt_t, max_time_t)/len(R_gender_test_t)
    
    est_prob = Model_utility.prob_estimated(s_type_t,dt,result1)
    obs_prob = Model_utility.prob_obs(df_gender_train_t)
    test_prob = Model_utility.prob_obs(df_gender_test_t)
    res_final = []
    res_final.extend(hyper[0:4])
   
    if condition==0:
        res_final.extend(res0.x[:-1]) 
        res_final.append(np.exp(res0.x[-1]))
    elif condition==1:
        res_final.append(res0.x[0])   
        res_final.append(np.exp(res0.x[-1]))
    elif condition==2:
        res_final.extend(res0.x) 
    else:
        res_final.append(res0.x)
  
    res_final.append(res0.fun/len(R_gender_train_t))
    res_final.append(test_LL)
    res_final.extend(est_prob)
    res_final.extend(obs_prob)
    res_final.extend(test_prob)
    res_final.append(res0.success)
    
    
    result_v = Model.test_solve_numerical(method, bias, drift0_v, bound0_v, noise0_v, x0_v, dx, dt, max_time_v)
    test_LL_v = Model_utility.calculate_LL(s_type = s_type_v, RT = RT_gender_test_v, R = R_gender_test_v, estimated_pdf = result_v, dt=dt, ndt=ndt_v, max_time = max_time_v)/len(R_gender_test_v)
    
    est_prob_v = Model_utility.prob_estimated(s_type_v,dt, result_v)
    obs_prob_v = Model_utility.prob_obs(df_gender_train_v)
    test_prob_v = Model_utility.prob_obs(df_gender_test_v)
    res_final_v = []
    res_final_v.extend([hyper[4],hyper[5],hyper[2],hyper[3]])
    if condition==0:
        res_final_v.extend(res1.x[:-1])
        res_final_v.append(np.exp(res1.x[-1]))
    elif condition==1:
        res_final_v.append(res1.x[0])   
        res_final_v.append(np.exp(res1.x[-1]))
    elif condition==2:
        res_final_v.extend(res1.x)
    else:
        res_final_v.append(res1.x[0])
    res_final_v.append(res1.fun/len(R_gender_train_v))
    res_final_v.append(test_LL_v)
    res_final_v.extend(est_prob_v)
    res_final_v.extend(obs_prob_v)
    res_final_v.extend(test_prob_v)
    res_final_v.append(res1.success)
    
    # write result

    file_route = 'data/newdata/estimate_gender_text_mle_' + '.txt'
    with open(file_route, 'a') as f:
        f.write( '  '.join(map(str, res_final)) + '\n')    
        
           
    
    file_route = 'data/newdata/estimate_gender_video_mle_' + '.txt'
    with open(file_route, 'a') as f:
        f.write( '  '.join(map(str, res_final_v)) + '\n')
    
    print("1 round finished.")
    end =  timer()
    print(end - start)
   


start0 = timer()


# After determining the hyper-parameter pair (bound,s0)
bound_t = 3.5
bound_v =3.5
s0_t = 1.75
s0_v = 1.75

condition = [0] # all three parameters are free

# repeat 30 iterations for each dataset
Parallel(n_jobs=-3)(delayed(each_loop)(bound_t=bound_t, s0_t = s0_t, bound_v=bound_v, s0_v = s0_v, df_gender_t = df_gender_t0,df_gender_v = df_gender_v0) for _ in range(30))    


end0 =  timer()



print(end0 - start0)






