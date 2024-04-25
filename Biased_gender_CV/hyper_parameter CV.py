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



start0 = timer()


#########################################
os.chdir('C:/Users/e0729936/source/repos/Recovery_drift_bias_ndt/Biased_gender_CV')
###########################################

# load the whole dataset
df_gender_t0 = pd.read_csv('data/newdata/gender_text_copy.csv')

#df_gender_v0 = pd.read_csv('data/newdata/gender_video.csv')
df_gender_v0 = pd.read_csv('data/gender_video_copy.csv')

def objective_function(x, *args):
    '''
    x:  array-like
        parameters for estimation.
    *args: (hyper,s_type, RT, R, filename). tuple-like, filename won't be used in CV 
        hyper: array-like, [b0,s0,dx,dt]
        s_type: bool, text=0,video=1.
        
    '''
    
    hyper, s_type, RT, R  = args
    

    v0 = x[0]
    ndt = x[2]
    x0 = x[1] # x0 = 0 for unbiased model
    
    
    dx = hyper[2]
    dt = hyper[3]
    
    # video and text data set has different max_time
    if s_type==0:
         max_time = 15.0
         b0 = hyper[0]
         s0 = hyper[1]
    else: 
         max_time = 8.0 
         b0 = hyper[4]
         s0 = hyper[5]
   
    bias = "point"#"centre"#
    method = "implicit"

    estimated_pdf = Model.test_solve_numerical(method, bias, v0, b0, s0, x0, dx, dt, max_time)
    res = Model_utility.calculate_LL(s_type,RT, R, estimated_pdf, dt, ndt, max_time)
    
    # Write the value of res to a file

    return res



def minimize_func(args):
    hyper, s_type, RT, R  = args
    #n_maxiter = 150
    # initial_guess = [0.5,0.5,0.5]
    # initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -0.2, 0.2,1),np.random.uniform( -1, 0,1)]
    if s_type:
        upperlimit = 0
    else:
        upperlimit = 1
    initial_guess = [np.random.uniform( -1, 1,1),np.random.uniform( -0.2, 0.2,1),np.random.uniform( -1, upperlimit,1)]
    initial_guess = np.array(initial_guess).reshape(3,)
    
    #np.random.uniform( -1, 1,3) #np.random.uniform( -1, 1,2) # random initial value
    return minimize(objective_function,initial_guess, args=args, method='Nelder-Mead')#, options={'maxiter': n_maxiter})


#####################################################
# random split training dataset and test dataset





def each_loop(bound, s0):
    '''
    for each loop, based on the given dataset and hyperparameter, return estimated parameter and test dataset -2LL
    ------------------------------
    bound: double
        the decision threshold.
    s0: double
        the noise standard deviation
    '''
    start = timer()
    # define hyper parameter set
    hyper = [bound_t, s0_t, 0.01,0.01,bound_v, s0_v]
    
    # generate training and test dataset
    df_gender_train_t, df_gender_test_t= Model_utility.df_draw(df_gender_t0)
    df_gender_train_v, df_gender_test_v = Model_utility.df_draw(df_gender_v0)
    #df_gender_train_v = df_gender_v0
    #df_gender_test_v =  df_gender_train_v
    
    RT_gender_train_t = df_gender_train_t['RT1'].to_numpy()
    R_gender_train_t  = df_gender_train_t['R'].to_numpy()
    RT_gender_test_t = df_gender_test_t['RT1'].to_numpy()
    R_gender_test_t  = df_gender_test_t['R'].to_numpy()
    
    
    RT_gender_train_v = df_gender_train_v['RT1'].to_numpy()
    R_gender_train_v  = df_gender_train_v['R'].to_numpy()
    RT_gender_test_v = df_gender_test_v['RT1'].to_numpy()
    R_gender_test_v  = df_gender_test_v['R'].to_numpy()
    
    # arugment for lieklihood setting
    s_type_t = 0 # s_type = 0 is text-based data, otherwise = 1
    s_type_v = 1
  


    res0=minimize_func((hyper,s_type_t, RT_gender_train_t, R_gender_train_t))
    res1=minimize_func((hyper,s_type_v,RT_gender_train_v, R_gender_train_v))
    
    # estimated parameter 
    dx =hyper[2]
    dt = hyper[3]
    
    
    
    drift0_t = res0.x[0]
    bound0_t = hyper[0]
    noise0_t = hyper[1]
    ndt_t = res0.x[2]#res0.x[1]# 
    x0_t =  res0.x[1]#0# 
    
    
    drift0_v = res1.x[0]
    bound0_v =  hyper[0]
    noise0_v = hyper[1]
    ndt_v = res1.x[2]
    x0_v = res1.x[1] 
    


    max_time_t = 15.0
    max_time_v = 8.0

    bias = "point"# "centre"#
    method = "implicit"

    result1 = Model.test_solve_numerical(method, bias, drift0_t, bound0_t, noise0_v, x0_t, dx, dt, max_time_t)
    result2 = Model.test_solve_numerical(method, bias, drift0_v, bound0_v, noise0_v, x0_v, dx, dt, max_time_v)
    
    test_LL_t = Model_utility.calculate_LL(s_type_t, RT_gender_test_t, R_gender_test_t, result1, dt, ndt_t, max_time_t)
    test_LL_v = Model_utility.calculate_LL(s_type_v, RT_gender_test_v, R_gender_test_v, result2, dt, ndt_v, max_time_v)
    '''
    prob_est = Model_utility.prob_estimated(s_type_t, dt, result1)
    prob_tr = Model_utility.prob_obs(df_gender_train_t)
    prob_ts = Model_utility.prob_obs(df_gender_test_t)
    
    '''
    prob_est = Model_utility.prob_estimated(s_type_v, dt, result2)
    prob_tr = Model_utility.prob_obs(df_gender_train_v)
    prob_ts = Model_utility.prob_obs(df_gender_test_v)
    


    res_final = hyper
    
    '''
    res_final.extend(res0.x[:-1])
    #res_final.append(res0.x[0])
    res_final.append(np.exp(res0.x[-1]))
    res_final.append(res0.fun/len(R_gender_train_t))
    res_final.append(test_LL_t/len(R_gender_test_t))
    res_final.extend(prob_est)
    res_final.extend(prob_tr)
    res_final.extend(prob_ts) 
    '''
    
    res_final.extend(res1.x[:-1])
    res_final.append(np.exp(res1.x[-1]))
    res_final.append(res1.fun/len(R_gender_train_v))
    res_final.append(test_LL_v/len(R_gender_test_v))
    res_final.extend(prob_est)
    res_final.extend(prob_tr)
    res_final.extend(prob_ts)
    
    # write result
    '''
    with open('data/estimate_gender_text_hyper_new.txt', 'a') as f:
        f.write( '  '.join(map(str, res_final)) + '\n')
    
    '''
    
    with open('data/estimate_gender_video_hyper_new.txt', 'a') as f:
        f.write( '  '.join(map(str, res_final)) + '\n')
    
    end =  timer()

    print(end - start)
    print("1 round finished.")


'''for hyperparameter searching'''
bound_list = np.arange(2, 6.5, 0.5)

 #sd = [(i-1)*0.5,i*0.5,(i+1)*0.5]
#print(bound_list)
hyper_list = np.zeros((27,2))


#hyper_list = np.zeros((8,2))
for i in range(len(bound_list)):
    hyper_list[i*3,:] = [bound_list[i],(bound_list[i]-1)*0.5]
    hyper_list[i*3+1,:] = [bound_list[i],(bound_list[i])*0.5]
    hyper_list[i*3+2,:] = [bound_list[i],(bound_list[i]+1)*0.5]

 

'''
    if i<=7:
        hyper_list[i:] = [bound_list[i],(bound_list[i]-2)*0.5]
    else:
        #
        j = i-8
        hyper_list[(3*j)+8,:] = [bound_list[i],(bound_list[i]-1)*0.5]
        hyper_list[(3*j+1)+8,:] = [bound_list[i],bound_list[i]*0.5]
        hyper_list[(3*j+2)+8,:] = [bound_list[i],(bound_list[i]+1)*0.5]
'''
print(hyper_list)

 

for i in range(30):
    Parallel(n_jobs=-3)(delayed(each_loop)( hyper_list[j,0],hyper_list[j,1]) for j in range(27)) 


#Parallel(n_jobs=-3)(delayed(each_loop)(5.5,2.75) for i in range(30)) 

end0 =  timer()

print(end0 - start0)






