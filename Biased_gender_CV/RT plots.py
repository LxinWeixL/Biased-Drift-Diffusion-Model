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


#########################################
os.chdir('C:/Users/e0729936/source/repos/Recovery_drift_bias_ndt/Biased_gender_CV')
###########################################

#####################################################
#df_gender_t0 = pd.read_csv('data/newdata/gender_text.csv')
#df_gender_v0 = pd.read_csv('data/newdata/gender_video.csv')
'''
df_gender_t0 = pd.read_csv('data/gender_text_copy.csv')
df_gender_v0 = pd.read_csv('data/gender_video_copy.csv')

df_gender_v = df_gender_v0
df_gender_t = df_gender_t0
'''
'''
grouped_t = df_gender_t0.groupby('sex')
df_gender_t = df_gender_t0#grouped_t.get_group(0)

RT_gender_t = df_gender_t['RT1'].to_numpy()
R_gender_t  = df_gender_t['R'].to_numpy()



grouped_v = df_gender_v0.groupby('sex')
df_gender_v = grouped_v.get_group(0)#df_gender_v0 #

RT_gender_v = df_gender_v['RT1'].to_numpy()
R_gender_v  = df_gender_v['R'].to_numpy()
'''




##########################################################
#res0 = [0.546, 0.000, 1.069] # gender M
#res0 = [0.894,0.000,1.069] # gender F
#res0 = [1.020098195, 0, 1.291454906] # theta/beta L
#res0 = [0.650973279, 0, 1.291454906]  # theta/beta H
#res0 = [0.977041717, 0, 1.208202041] # alpha L
#res0 = [0.643301584, 0, 1.318368845] # alpha H
#res0 = [0.715,0.000,1.001] # alpha H
############################################################
#res0 = [0.026305094,-0.472438204,0.863992439] # all
#res0 = [-0.235296602,0.460115864,0.940593801] # all

############################################################
#res1 = [0.316, 0.000,0.609] # gender M
#res1 =[0.868,0.000,0.609] #gender F
#res1 =[1.019,0.137,1.303] # gender F all free parameters
#res1 =[0.927572842,0,0.481510524] # theta/beta L
#res1 =[0.427578618, 0, 0.481510524] # theta/beta H
#res1 =[0.625564826, 0, 0.467672188] # alpha L
#res1 =[0.549617068, 0, 0.467672188] # alpha H
#res1 =[0.590484518, 0, 0.482184516]  # alpha

###########################################################
#res1 = [0.00996744	,0	,0] # gender M, (f,0,0)
#res1 =[0.074,-0.137,0.381] # gender M, (f,f,f)

# res1 without adjustment
# res0 with adjustment
#################################################
# with adjustment
#res1 = [0.540360402,-0.049081903,0.802512832]

#res0 = [0.556279282,-0.183312264,1.132617122]
'''
Probability(est/obs/adj) [0.756,0.237,0.006]/[0.763,0.237]/ [0.761,0.239] for text
Probability(est/obs/adj  [0.709,0.213,0.079]/[0.761,0.239]/[0.770,0.230] for video
'''

def each_run(j):
    

    file_t = 'data/newdata/text_train'+str(j)+'.csv'
    file_v = 'data/newdata/video_train'+str(j)+'.csv'
    df_t =  pd.read_csv(file_t)
    df_v =  pd.read_csv(file_v)
    
    df_t_RT = df_t["RT"]
    df_t_R = df_t["R"]
    df_v_RT = df_v["RT1"]
    df_v_R = df_v["R"]
 
    df_t_est = pd.read_csv('data/estimate_gender_text_all_mle4.txt',sep="  ",header = None)
    df_v_est = pd.read_csv('data/estimate_gender_video_all_mle4.txt',sep="  ",header = None)
  
    res0 = df_t_est.iloc[j,4:7].to_numpy()
    res1 = df_v_est.iloc[j,4:7].to_numpy()
   

    dx = 0.01
    dt = 0.01
    s_type_t = 0
    s_type_v = 1

 #[0.868, 0.09, 0.043]

    drift0_t = res0[0]
    bound0_t = 3.5#5
    ndt_t = np.log(res0[2]) 
    x0_t = res0[1]
    noise0_t = 1.75#2.25




    drift0_v = res1[0]
    bound0_v =3.5
    noise0_v = 1.75
    ndt_v = np.log(res1[2]) 
    x0_v = res1[1] 




    max_time_t = 15.0
    max_time_v = 8.0
    bias = "point"#"centre"##

    method = "implicit"




    result1 = Model.test_solve_numerical(method, bias, drift0_t, bound0_t, noise0_t, x0_t, dx, dt, max_time_t)
    sim_responses1 = Model_utility.simulate(result1,df_t, dt, ndt_t, max_time_t,s_type_t)

    simdata1 = {'RT1':  sim_responses1[1], 'R':  sim_responses1[0], 'RT_o':df_t_RT,'R_o':df_t_R}
    
    simdf_test = pd.DataFrame(simdata1)

    
    RT_L = np.max([np.median(simdf_test['RT1']) - 3*np.std(simdf_test['RT1']),res0[2]])
    RT_U = np.min([np.median(simdf_test['RT1']) + 3*np.std(simdf_test['RT1']),max_time_t])
    index_d = []
    for i in range(len(simdf_test['RT1'])):
        if (simdf_test['RT1'][i]<RT_L) | (simdf_test['RT1'][i] >RT_U):
            index_d.append(i)

    simdf1 = simdf_test.drop(index = index_d)     
    #simdf1 = pd.concat([simdf1,df_t],axis = 1)
    
    simdf1 = simdf_test
    simdf1.to_csv('data/simdata_text_all_final'+str(j)+'.csv')
    #simdf1=simdf_test
    '''
    file_path_st = 'data/simdata_text_all_final.csv'
    with open(file_path_st, 'a') as f:
        f.writelines(simdf1)
    '''   
    #simdf1.to_csv('data/simdata_text_all_final.csv', index=False)


    #simdf1 = pd.read_csv('data/simdata_text_all_adj2.csv')



    result2 = Model.test_solve_numerical(method, bias, drift0_v, bound0_v, noise0_v, x0_v, dx, dt, max_time_v)
    sim_responses2 = Model_utility.simulate(result2,df_v, dt, ndt_v, max_time_v,s_type_v)

    simdata2 = {'RT1':  sim_responses2[1], 'R':  sim_responses2[0],'RT_o':df_v_RT,'R_o':df_v_R}
    simdf_test = pd.DataFrame(simdata2)
    
    
    RT_L = np.max([np.median(simdf_test['RT1']) - 3*np.std(simdf_test['RT1']),res1[2]])
    RT_U = np.min([np.median(simdf_test['RT1']) + 3*np.std(simdf_test['RT1']),8])

    #RT_L = 0.5
    #RT_U = 8.0
    
    index_d = []
    for i in range(len(simdf_test['RT1'])):
        if (simdf_test['RT1'][i]<RT_L) | (simdf_test['RT1'][i] >RT_U):
            index_d.append(i)
    simdf2 = simdf_test.drop(index = index_d)  
    
    
    simdf2.to_csv('data/simdata_video_all_final'+str(j)+'.csv')
    #df2_final = pd.concat([df2_final,simdf2])
    #file_path_st = 'data/simdata_video_all_final.csv'
    


    #simdf2.to_csv('data/simdata_video_all_adj2.csv', index=False)

    print("finished")
    #simdf2 = pd.read_csv('data/simdata_video_all_adj2.csv')




Parallel(n_jobs=-3)(delayed(each_run)(j) for j in range(30))    

    #res1 = df_t_est[]
#%%



#Drift* drift, Noise* noise, Bound* bound, Bias* bias, double dx, double dt, double max_time,
#        double drift0, double noise0, double b0)

'''
import seaborn as sns

def plot_rt_density(s_type,df1, df2):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    # Create a boolean mask for rows where 'R' is 0
    mask0 = df1['R'] == 0
    mask1 = df2['R'] == 0

    # Create a boolean mask for rows where 'R' is 1
    mask2 = df1['R'] == 1
    mask3 = df2['R'] == 1

    # Use the boolean masks to extract the 'RT' values for each category of 'R'
    if s_type:
        RT0 = df1.loc[mask0, 'RT1']#df1.loc[mask0, 'RT1']
        RT1 = df2.loc[mask1, 'RT1']
        RT2 = df1.loc[mask2, 'RT1']#df1.loc[mask2, 'RT1']
        RT3 = df2.loc[mask3, 'RT1']
    else :
        RT0 = df1.loc[mask0, 'RT']#df1.loc[mask0, 'RT1']
        RT1 = df2.loc[mask1, 'RT1']
        RT2 = df1.loc[mask2, 'RT']#df1.loc[mask2, 'RT1']
        RT3 = df2.loc[mask3, 'RT1']
    # Plot the density plots of the 'RT' values for each category of 'R'
    sns.kdeplot(RT0, ax=axs[0], label='Real dataset', color='blue')
    sns.kdeplot(RT1, ax=axs[0], label='Simulated dataset', color='red')
    sns.kdeplot(RT2, ax=axs[1], label='Real dataset', color='green')
    sns.kdeplot(RT3, ax=axs[1], label='Simulated dataset', color='orange')
    
    axs[0].set_xlabel('RT')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Save M')
    axs[0].legend()
    
    axs[1].set_xlabel('RT')
    axs[1].set_ylabel('Density')
    axs[1].set_title('Save F')
    axs[1].legend()
    #plt.savefig("plots/")
    plt.show()

def plot_rt_histogram(s_type, df1, df2, binwidth=0.3):
    # Create a boolean mask for rows where 'R' is 0
    
        
    mask0 = df1['R'] == 0
    mask1 = df2['R'] == 0

    # Create a boolean mask for rows where 'R' is 1
    mask2 = df1['R'] == 1
    mask3 = df2['R'] == 1

    # Use the boolean masks to extract the 'RT' values for each category of 'R'
    if s_type:
        
        RT0 = df1.loc[mask0, 'RT1']#df1.loc[mask0, 'RT1']
        RT2 = df1.loc[mask2, 'RT1']#df1.loc[mask2, 'RT1']
    else:
        RT0 = df1.loc[mask0, 'RT']#df1.loc[mask0, 'RT1']
        RT2 = df1.loc[mask2, 'RT']#df1.loc[mask2, 'RT1']
    RT1 = df2.loc[mask1, 'RT1']
    
    
    RT3 = df2.loc[mask3, 'RT1']
    # creat two plots, the upper one is for R = 0; the lower one is for R = 1.
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    # Plot the histograms of the 'RT' values for each category of 'R'

    sns.histplot(RT0, ax=axs[0], binwidth=binwidth, alpha=0.5, label='Real dataset R=0', color='blue',stat = "density")
    sns.histplot(RT1, ax=axs[0], binwidth=binwidth, alpha=0.5, label='Simulated dataset R=0', color='red',stat = "density")
    sns.histplot(RT2, ax=axs[1], binwidth=binwidth, alpha=0.5, label='Real dataset R=1', color='green',stat = "density")
    sns.histplot(RT3, ax=axs[1], binwidth=binwidth, alpha=0.5, label='Simulated dataset R=1', color='orange',stat = "density")
    
    axs[0].set_xlabel('RT')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of RT by Category of  Save M')
    axs[0].legend()

    axs[1].set_xlabel('RT')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of RT by Category of Save F')
    axs[1].legend()

    plt.show()
    #plt.savefig("plots/")

#%%
# plot_rt_histogram(df)



plot_rt_density(s_type_t, df_gender_t, simdf1)
plot_rt_histogram(s_type_t,df_gender_t, simdf1)




plot_rt_density(s_type_v,df_gender_v, simdf2)
plot_rt_histogram(s_type_v,df_gender_v, simdf2)


#plotname = "plots/multi.pdf"
#save_multi_image(plotname)

'''