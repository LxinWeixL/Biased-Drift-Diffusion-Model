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
df_gender_t0 = pd.read_csv('data/gender_text_copy2.csv')
df_gender_v0 = pd.read_csv('data/gender_video_copy2.csv')


def objective_function(x, *args):
    '''
    x:  array-like
        parameters for estimation.
    *args: (hyper,s_type,subgroup, RT, R, filename). tuple-like, filename won't be used in CV 
        hyper: array-like, [b0_t,s0_t,dx,dt,b0_v,s0_v]
        s_type: bool, text=0,video=1.
        condition: int type.
            condition=0, all is free.
            condition= 1,x0 = 0.
            
    '''
    
    hyper, s_type, condition, RT, R  = args
   
    
    v0 = x[0]
    
    dx = hyper[2]
    dt = hyper[3]
    


    method = "implicit"
    # video and text data set has different max_time
    if s_type==0:
         max_time = 15.0
         b0 = hyper[0]
         s0 = hyper[1]
    else: 
         max_time = 8.0 
         b0 = hyper[4]
         s0 = hyper[5]
         
    if condition:
        # X0 = 0
        ndt = x[1]
        x0 = 0
        bias = "centre"
    else:
        ndt = x[2]#
        x0 = x[1]
        bias = "point"
        
    estimated_pdf = Model.test_solve_numerical(method, bias, v0, b0, s0, x0, dx, dt, max_time)
    res = Model_utility.calculate_LL(s_type, RT, R, estimated_pdf, dt, ndt,max_time)
    
    # Write the value of res to a file

    return res



def minimize_func(args):
    #n_maxiter = 150!!!!!!!!!!
    hyper, s_type, condition, RT, R  = args
    # random select initial value
    if condition:
        if s_type:
            initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -1, -0.2,1)]
        else:
            initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -1, 0.7,1)]
        initial_guess = np.array(initial_guess).reshape(2,)
    else:
        if s_type:
            initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -0.2, 0.2,1),np.random.uniform( -1, -0.2,1)]
        else:
            initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -0.2, 0.2,1),np.random.uniform( -1, 0.7,1)]
            
        initial_guess = np.array(initial_guess).reshape(3,)
    
    return minimize(objective_function,initial_guess, args=args, method='Nelder-Mead')#, options={'maxiter': n_maxiter})
    # ’ ,"BFGS"


#####################################################
# random split training dataset and test dataset





def each_loop(bound_t, s0_t,bound_v, s0_v, df_gender_t, df_gender_v,para,condition,subgroup):
    '''
    for each loop, based on the given dataset and hyperparameter, return estimated parameter and test dataset -2LL
    ------------------------------
    bound: double
        the decision threshold.
    s0: double
        the noise standard deviation
     df_gender_t: dataframe
     df_gender_v: dataframe
     para: string type
     condition: int type. 
            condition=0, all is free.
            condition= 1,x0 = 0.
    subgroup: int, the index of excluded subgroup
    '''
    start = timer()
    # define hyper parameter set
    hyper = [bound_t, s0_t, 0.01,0.01,bound_v,s0_v]
    # arugment for lieklihood setting
    s_type_t = 0 # s_type = 0 is text-based data, otherwise = 1
    s_type_v = 1
    
    max_time_t = 15.0
    max_time_v = 8.0
    method = "implicit"
    # generate training dataset
    ''''''
    df_gender_train_t = Model_utility.df_draw2(df_gender_t) 
    beta_s = df_gender_train_t[para].to_numpy()[1]#subgroup# 
    #sex = df_gender_train_t["sex"].to_numpy()[1]
    
    #RT_gender_train_t = df_gender_train_t['RT1'].to_numpy()
    RT_gender_train_t = df_gender_train_t['RT'].to_numpy()
    R_gender_train_t  = df_gender_train_t['R'].to_numpy()
    res0= minimize_func((hyper,s_type_t, condition, RT_gender_train_t, R_gender_train_t))
    

    
    
    df_gender_train_v = Model_utility.df_draw2(df_gender_v) 
    beta_s = df_gender_train_v[para].to_numpy()[1]#subgroup#
    #sex = df_gender_train_v["sex"].to_numpy()[1]
    RT_gender_train_v = df_gender_train_v['RT1'].to_numpy()
    R_gender_train_v  = df_gender_train_v['R'].to_numpy()
 
    res1=minimize_func((hyper,s_type_v, condition, RT_gender_train_v, R_gender_train_v))
    
    
    # estimated parameter 
    dx =hyper[2]
    dt = hyper[3]
    bound0_t = hyper[0]
    noise0_t = hyper[1]
    bound0_v = hyper[4]
    noise0_v = hyper[5]
    
    ''''''
    
    if condition:
        # x0 = 0

        drift0_t = res0.x[0]
        ndt_t = res0.x[1]
        x0_t = 0 
        bias =  "centre"
    else:
        drift0_t = res0.x[0]
        ndt_t = res0.x[2]
        x0_t = res0.x[1] 
        bias =  "point"
        

    
    
    if condition:
        # x0 = 0

        drift0_v = res1.x[0]
        ndt_v = res1.x[1]
        x0_v = 0 
        bias =  "centre"
    else:
        drift0_v = res1.x[0]
        ndt_v = res1.x[2]
        x0_v = res1.x[1] 
        bias =  "point"
    
    

    
    result1 = Model.test_solve_numerical(method, bias, drift0_t, bound0_t, noise0_t, x0_t, dx, dt, max_time_t)
    
    est_prob = Model_utility.prob_estimated(s_type_t,dt,result1)
    obs_prob = Model_utility.prob_obs(df_gender_train_t)
    
    res_final = [beta_s]#,sex]
    #res_final = [sex]
    #res_final.extend(hyper)
    res_final.extend(hyper[0:4])
    tmp = res0.x
    if condition:
        tmp[1] = np.exp(res0.x[1])
    else:
        tmp[2] = np.exp(res0.x[2])
    
    res_final.extend(tmp)
    
    
    res_final.append(res0.fun/len(R_gender_train_t))
  
    res_final.extend(est_prob)
    res_final.extend(obs_prob)
    
    
    result_v = Model.test_solve_numerical(method, bias, drift0_v, bound0_v, noise0_v, x0_v, dx, dt, max_time_v)
    
    est_prob_v = Model_utility.prob_estimated(s_type_v,dt,result_v)
    obs_prob_v = Model_utility.prob_obs(df_gender_train_v)
    res_final_v = [beta_s]#,sex]
    #res_final_v.extend(hyper)
    res_final_v.extend([hyper[4],hyper[5],hyper[2],hyper[3]])
    tmp = res1.x
    if condition:
        tmp[1] = np.exp(res1.x[1])
    else:
        tmp[2] = np.exp(res1.x[2])
    
    res_final_v.extend(tmp)
   
    res_final_v.append(res1.fun/len(R_gender_train_v))
    
    res_final_v.extend(est_prob_v)
    res_final_v.extend(obs_prob_v)
    
    
    
    # write result
    ''''''
    file_path = 'data/estimate_gender_text2_bs_'+para+'.txt'
    #file_path = 'data/estimate_gender_text_bs_sex.txt'
    with open(file_path, 'a') as f:
        f.write( '  '.join(map(str, res_final)) + '\n')   
        

           
    file_path = 'data/estimate_gender_video2_bs_'+para+'.txt'
    #file_path = 'data/estimate_gender_video_bs_sex.txt'
    with open(file_path, 'a') as f:
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


# divide dataset by sub_gender, sex = 0 is M.
grouped_t = df_gender_t0.groupby('sex')
df_gender_t0_M = grouped_t.get_group(0)
df_gender_t0_F = grouped_t.get_group(1)
grouped_v = df_gender_v0.groupby('sex')
df_gender_v0_M = grouped_v.get_group(0)
df_gender_v0_F = grouped_v.get_group(1)

#print(df_gender_t0_F.head())

#each_loop(bound, s0, df_gender_t0_F, df_gender_v0_F)
#each_loop(bound, s0, df_gender_t0, df_gender_v0_F)
'''
for i in range(2):
    Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = grouped_t.get_group(i),df_gender_v = df_gender_v0_F) for _ in range(30))    
'''
'''
grouped_t = df_gender_t0.groupby('beta_s')
grouped_v = df_gender_v0.groupby('beta_s')
'''


bound_t = 3.5# 5#
s0_t = 1.75#2.25#
bound_v = 3.5
s0_v = 1.75
#condition = [0,1]
condition = [0]


#for k in range(2):
#    Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = grouped_t.get_group(k),df_gender_v = grouped_v.get_group(k),para = "beta_s4", condition = 0, subgroup = 0) for _ in range(30))    

para = ["beta","theta"]

for i in para:
    grouped_t = df_gender_t0.groupby(i)
    grouped_v = df_gender_v0.groupby(i)
    for k in range(2):
            # to do: the subgroup index should be inversed
            Parallel(n_jobs=-2)(delayed(each_loop)(bound_t=bound_t, s0_t = s0_t,bound_v=bound_v, s0_v = s0_v, df_gender_t = grouped_t.get_group(k),df_gender_v = grouped_v.get_group(k),para = i, condition = 0, subgroup = -1) for _ in range(150))    

''' used for beta subgroup


grouped_t = df_gender_t0.groupby('beta_s2')
grouped_v = df_gender_v0.groupby('beta_s2')
for j in condition:
    # 10
    for i in np.arange(9, 20,1):
        # reduce the slected group, therefore, the subgroup contains the highest beta-bands to lowest.
        sample_index_t = grouped_t.get_group(i).index
        #sample_index_t = grouped_t.get_group(1).index #just for no index of text
        train_t = df_gender_t0.drop(index = sample_index_t)
        #sample_index_v = grouped_v.get_group(i).index
        sample_index_v = grouped_v.get_group(i).index
        train_v = df_gender_v0.drop(index = sample_index_v)
        
        # divide the subgroup into two groups by gender
        grouped_t_1 = train_t.groupby('sex')
        grouped_v_1 = train_v.groupby('sex')
        #grouped_t_1[['s']].nunique()
        #print(grouped_t_1[['s']].nunique(),grouped_v_1[['s']].nunique())

        for k in range(2):
            # to do: the subgroup index should be inversed
            Parallel(n_jobs=-4)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = grouped_t_1.get_group(k),df_gender_v = grouped_v_1.get_group(k),para = "beta_s2", condition = j, subgroup = np.abs(19-i)) for _ in range(30))    

'''



'''

Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = df_gender_t0_F,df_gender_v = grouped_v.get_group(1)) for _ in range(30))    

'''


end0 =  timer()
print(len(df_gender_t0_M["RT"].to_numpy()))
print(len(df_gender_t0_F["RT"].to_numpy()))
print(end0 - start0)







