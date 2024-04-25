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
'''
df_gender_t0 = pd.read_csv('data/newdata/gender_text.csv')
df_gender_v0 = pd.read_csv('data/newdata/gender_video.csv')

'''

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
         max_time = 15 #15.0 #
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
    #n_maxiter = 150
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
        #initial_guess = np.array(initial_guess).reshape(3,)
    # random select initial value
  
    return minimize(objective_function,initial_guess, args=args, method="Nelder-Mead")#, options={'maxiter': n_maxiter})
    # ’ 'Nelder-Mead', #Nelder-Mead


#####################################################
# random split training dataset and test dataset





def each_loop(bound_t, s0_t,bound_v, s0_v, df_gender_t, df_gender_v,para,condition):
    '''
    for each loop, based on the given dataset and hyperparameter, return estimated parameter and test dataset -2LL
    ------------------------------
    bound: double
        the decision threshold.
    s0: double
        the noise standard deviation
     df_gender_t: dataframe
     df_gender_v: dataframe
     para: a string
     condition: an int type. 
        condition = 0, three free parameters; 
        condition = 1, x0 =0;
        condition = 2, ndt = 0; 
        condition = 3, only drift is free.
    '''
    
    start = timer()
    # define hyper parameter set
    hyper = [bound_t, s0_t,  0.01,0.01, bound_v, s0_v]
    
    # generate training and test dataset
    #df_gender_train_t, df_gender_test_t = Model_utility.df_draw(df_gender_t)
    #df_gender_train_v, df_gender_test_v = Model_utility.df_draw(df_gender_v)
    
    df_gender_train_t = Model_utility.df_draw3(df_gender_t)
    df_gender_train_v = Model_utility.df_draw3(df_gender_v)
    #file_t = 'data/newdata/text_train' +  str(iter_i) +'.csv'
    #file_v = 'data/newdata/video_train' +  str(iter_i) +'.csv'
    #df_gender_train_t.to_csv(file_t)
    #df_gender_train_v.to_csv(file_v)
    #df_gender_train_v = df_gender_v
    #df_gender_test_v = df_gender_v
    df_gender_test_v = df_gender_train_v
    sex = df_gender_train_v[para].to_numpy()[1]
   
    
    
    #df_gender_train_t = df_gender_t
    #df_gender_test_t = df_gender_t
    df_gender_test_t = df_gender_train_t
    #sex = df_gender_train_t[para].to_numpy()[1]
    
    
    
    
    RT_gender_train_t = df_gender_train_t['RT'].to_numpy()
    R_gender_train_t  = df_gender_train_t['R'].to_numpy()
    RT_gender_test_t = df_gender_test_t['RT'].to_numpy()
    R_gender_test_t  = df_gender_test_t['R'].to_numpy()

    
    
    RT_gender_train_v = df_gender_train_v['RT1'].to_numpy()
    R_gender_train_v  = df_gender_train_v['R'].to_numpy()
    RT_gender_test_v = df_gender_test_v['RT1'].to_numpy()
    R_gender_test_v  = df_gender_test_v['R'].to_numpy()
    
    # arugment for lieklihood setting
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
    res_final = [sex]#[sex]
    res_final.extend(hyper[0:4])
    #res_final = hyper[0:4]
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
    res_final_v = [sex]#[sex]
    #res_final_v.extend(hyper)
    #res_final_v = [hyper[4],hyper[5],hyper[2],hyper[3]]
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
    
    #file_route = 'data/estimate_gender_text_all_mle4.txt'
    #file_route = 'data/estimate_gender_text_all.txt'
    file_route = 'data/newdata/estimate_gender_text_mle2_' +  para +'.txt'
    with open(file_route, 'a') as f:
        f.write( '  '.join(map(str, res_final)) + '\n')    
        
           
    #file_route = 'data/estimate_gender_video_all.txt'
    file_route = 'data/newdata/estimate_gender_video_mle2_' +  para +'.txt'
    #file_route = 'data/estimate_gender_video_all_mle4.txt'
    with open(file_route, 'a') as f:
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

bound_t = 3.5#5
bound_v =3.5#5#
s0_t = 1.75
s0_v = 1.75# 2.5#
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

para = ["sex","beta","alpha","theta"]
#para = ["theta"]
#condition = [0,1,2,3]
condition = [0]


for m in condition:
    for k in para:
        grouped_t = df_gender_t0.groupby(k)
        #grouped_t = df_gender_t0.groupby('theta')
        df_gender_t0_M = grouped_t.get_group(0)
        df_gender_t0_F = grouped_t.get_group(1)

 
        grouped_v = df_gender_v0.groupby(k)
        df_gender_v0_M = grouped_v.get_group(0)
        df_gender_v0_F = grouped_v.get_group(1)   
        #print(len(df_gender_v0_M["RT1"].to_numpy()))
        #print(len(df_gender_v0_F["RT1"].to_numpy()))
        #print(len(df_gender_t0_M["R"].to_numpy()))
        #print(len(df_gender_t0_F["R"].to_numpy()))
        '''
        if k=="beta":
            Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = grouped_v.get_group(0),df_gender_v = grouped_v.get_group(0), para = k, condition = m) for _ in range(1))    
            Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = grouped_v.get_group(1),df_gender_v = grouped_v.get_group(1), para = k, condition = m) for _ in range(30))    
        '''   
        for i in range(2): 
            #if k!="beta":  
            Parallel(n_jobs=-3)(delayed(each_loop)(bound_t=bound_t, s0_t = s0_t, bound_v=bound_v, s0_v = s0_v,df_gender_t = grouped_t.get_group(i),df_gender_v = grouped_v.get_group(i), para = k, condition = m) for _ in range(30))    
           
        #Parallel(n_jobs=-3)(delayed(each_loop)(bound_t=bound_t, s0_t = s0_t, bound_v=bound_v, s0_v = s0_v, df_gender_t = grouped_t, df_gender_v = grouped_, para = k, condition = m) for _ in range(30))    

 
 

#Parallel(n_jobs=-3)(delayed(each_loop)(bound_t = bound_t, s0_t = s0_t,bound_v = bound_v, s0_v = s0_v, df_gender_t = df_gender_t0,df_gender_v = df_gender_v0, para = "sex", condition = 0,iter_i = _) for _ in range(30))    


'''

for i in range(2):
   Parallel(n_jobs=-2)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = df_gender_t0_F,df_gender_v = grouped_v.get_group(i)) for _ in range(30))    



         
Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = df_gender_t0_F,df_gender_v = grouped_v.get_group(1)) for _ in range(30))    

'''


#Parallel(n_jobs=-3)(delayed(each_loop)(bound=bound, s0 = s0, df_gender_t = df_gender_t0,df_gender_v = df_gender_v0) for _ in range(30))

end0 =  timer()

#Model_utility.df_draw3(df_gender_v0)

print(end0 - start0)

# 121655sec for 3*3*10=90, 22.5 min a run.

# check why the resample has the same -2LL value.
#each_loop(4,2)
# parallel algorithm




