from msilib.schema import File
import Model
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from IPython.display import Image
import Model_utility 
import os
from multiprocessing import Pool
from joblib import Parallel, delayed

############################################

os.chdir('C:/Users/e0729936/source/repos/Recovery_drift_bias_ndt/Biased_gender_CV')

def each_run(j):
    

    file_t = 'data/newdata/text_train'+str(j)+'.csv'
    file_v = 'data/newdata/video_train'+str(j)+'.csv'
    df_t =  pd.read_csv(file_t)
    df_v =  pd.read_csv(file_v)
    
    df_t_RT = df_t["RT"]
    df_t_R = df_t["R"]
    df_v_RT = df_v["RT1"]
    df_v_R = df_v["R"]
 
    df_t_est = pd.read_csv('data/estimate_gender_text_all_mle.txt',sep="  ",header = None)
    df_v_est = pd.read_csv('data/estimate_gender_video_all_mle.txt',sep="  ",header = None)
  
    res0 = df_t_est.iloc[j,4:7].to_numpy()
    res1 = df_v_est.iloc[j,4:7].to_numpy()
   

    dx = 0.01
    dt = 0.01
    s_type_t = 0
    s_type_v = 1



    drift0_t = res0[0]
    bound0_t = 3.5
    ndt_t = np.log(res0[2]) 
    x0_t = res0[1]
    noise0_t = 1.75




    drift0_v = res1[0]
    bound0_v =3.5
    noise0_v = 1.75
    ndt_v = np.log(res1[2]) 
    x0_v = res1[1] 




    max_time_t = 15.0
    max_time_v = 8.0
    bias = "point" #"centre" if X0 = 0

    method = "implicit"


######################################################################################
# video simulation data prepocessing
    result1 = Model.test_solve_numerical(method, bias, drift0_t, bound0_t, noise0_t, x0_t, dx, dt, max_time_t)
    sim_responses1 = Model_utility.simulate(result1,df_t, dt, ndt_t, max_time_t,s_type_t)

    simdata1 = {'RT1':  sim_responses1[1], 'R':  sim_responses1[0], 'RT_o':df_t_RT,'R_o':df_t_R}
    
    simdf_test = pd.DataFrame(simdata1)

    # remove outlier
    RT_L = np.max([np.median(simdf_test['RT1']) - 3*np.std(simdf_test['RT1']),res0[2]])
    RT_U = np.min([np.median(simdf_test['RT1']) + 3*np.std(simdf_test['RT1']),max_time_t])
    index_d = []
    for i in range(len(simdf_test['RT1'])):
        if (simdf_test['RT1'][i]<RT_L) | (simdf_test['RT1'][i] >RT_U):
            index_d.append(i)
    simdf1 = simdf_test.drop(index = index_d)     

    
    simdf1 = simdf_test
    simdf1.to_csv('data/simdata_text_all_final'+str(j)+'.csv')

  ##########################################################################
    # video simulation data prepocessing

    result2 = Model.test_solve_numerical(method, bias, drift0_v, bound0_v, noise0_v, x0_v, dx, dt, max_time_v)
    sim_responses2 = Model_utility.simulate(result2,df_v, dt, ndt_v, max_time_v,s_type_v)

    simdata2 = {'RT1':  sim_responses2[1], 'R':  sim_responses2[0],'RT_o':df_v_RT,'R_o':df_v_R}
    simdf_test = pd.DataFrame(simdata2)
    
    # remove outlier
    RT_L = np.max([np.median(simdf_test['RT1']) - 3*np.std(simdf_test['RT1']),res1[2]])
    RT_U = np.min([np.median(simdf_test['RT1']) + 3*np.std(simdf_test['RT1']),8])


    RT_U = 8.0 # video-based scene has max_time
    
    index_d = []
    for i in range(len(simdf_test['RT1'])):
        if (simdf_test['RT1'][i]<RT_L) | (simdf_test['RT1'][i] >RT_U):
            index_d.append(i)
    simdf2 = simdf_test.drop(index = index_d)  
    simdf2.to_csv('data/simdata_video_all_final'+str(j)+'.csv')
    

 




Parallel(n_jobs=-3)(delayed(each_run)(j) for j in range(30))    



