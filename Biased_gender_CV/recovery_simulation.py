import Model
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from IPython.display import Image
import Model_utility 
import os
from multiprocessing import Pool

#########################################
os.chdir('C:/Users/e0729936/source/repos/Recovery_drift_bias_ndt/Biased_gender_CV')
###########################################

def objective_function(x, *args):
    s_type, RT ,R ,filename = args
    v0 = x[0]
    #b0 = x[1] #for unbiased model
    b0 = 6
    
     
    ndt = x[2]#-np.Inf#x[1] #
    
    x0 = x[1] #0# x0 =  for unbiased model
    s0 = 3
    dx = 0.01
    dt = 0.01
    # video and text data set has different max_time

    
    if s_type==0:
         max_time = 15.0
    else: 
         max_time = 18 
   
    bias = "point"#"centre"##
    method = "implicit"

    estimated_pdf = Model.test_solve_numerical(method, bias, v0, b0, s0, x0, dx, dt, max_time)# max_time
    res = Model_utility.calculate_LL(s_type, RT, R, estimated_pdf, dt, ndt,max_time)
    
    # Write the value of res to a file
    with open(filename, 'a') as f:
        f.write(str(res) + '\n')

    return res
'''
def callback_gender_t(xk):
    with open('data/estimate_gender_text_F.txt', 'a') as f:
            f.write('   '.join(map(str, xk)) + '\n')
'''
def callback_gender_v(xk):
    with open('data/estimate_gender_video_all(6,3).txt', 'a') as f:
            f.write('   '.join(map(str, xk)) + '\n')

#####################################################
df_gender_t = pd.read_csv('data/newdata/gender_text.csv')
df_gender_v = pd.read_csv('data/newdata/gender_video.csv')

print(len(df_gender_v["R"].to_numpy()))


'''
grouped_t = df_gender_t.groupby('sex')
df_gender_t0_M = grouped_t.get_group(0)
df_gender_t0_F = grouped_t.get_group(1)
df_gender_t = df_gender_t0_F

'''

RT_gender_t = df_gender_t['RT1'].to_numpy()
R_gender_t  = df_gender_t['R'].to_numpy()
RT_gender_v = df_gender_v['RT1'].to_numpy()
R_gender_v  = df_gender_v['R'].to_numpy()
s_type_t = 0
s_type_v = 1


n_maxiter = 150
initial_guess =  [np.random.uniform( -1, 1,1),np.random.uniform( -0.2, 0.2,1),np.random.uniform( -1, 0,1)]
initial_guess = np.array(initial_guess).reshape(3,)
#initial_guess = [0,0]
 #[0.680114947,np.log(1.31842917)]  # Initial guess for the parameters
filename_gender_t = 'data/likelihood_gender_text_F.txt'
filename_gender_v = 'data/likelihood_gender_video_all(6,3).txt'

def minimize_func(args):
    return minimize(objective_function,initial_guess, args=args[0], method='Nelder-Mead', options={'maxiter': n_maxiter}, callback=args[1])

#''


#######################################################

# s_type = 0 is text-based data, otherwise = 1;

print("Starting now.")
#res0=minimize_func( ((s_type_t, RT_gender_t, R_gender_t, filename_gender_t), callback_gender_t) )





print("half is finished.")
res1=minimize_func( ((s_type_v,RT_gender_v, R_gender_v, filename_gender_v), callback_gender_v) )

print("estimation  is finished.")
#######################################################

# text
# 0.44830743798948364   -0.17568152158099892   1.1525396211980485

#video
# 0.45824627790993   0.5202165634444422   1.8683354166577433


def plot_line_chart(data):
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)
    names = df.columns
    # Create a figure and axes
    fig, axes = plt.subplots(nrows=df.shape[1], ncols=1, figsize=(5, 5))
    
    # Plot each column in a separate panel
    for i, col in enumerate(df.columns):
        df[col].plot(ax=axes[i], title=col)
    
    # Show the plot
    plt.tight_layout()
    #return fig
'''



est_file_path_text = 'data/estimate_gender_text_F.txt'
est_df_text = pd.read_csv(est_file_path_text, sep='   ', header=None, engine='python')

plot_line_chart(est_df_text)

'''

est_file_path_video = 'data/estimate_gender_video_all(6,3).txt'
est_df_video = pd.read_csv(est_file_path_video, sep='   ', header=None, engine='python')

plot_line_chart(est_df_video)



'''
#.savefig("plots/text_convergent.png")
est_file_path_video = 'data/estimate_gender_video.txt'
est_df_video = pd.read_csv(est_file_path_video, sep='   ', header=None, engine='python')
plot_line_chart(est_df_video)
'''

#.savefig("plots/video_convergent.png")

##########################################################
#res0 = [0.946118316,0.048325461,1.007655785]
#res0 = [0.680114947,-0.067170469,1.31842917]

dx = 0.01
dt = 0.01
'''

drift0_t = res0.x[0]#res0[0]#
bound0_t = 5.5
#bound0_t =res0.x[1] #unbiased
ndt_t =  res0.x[-1] #np.log(res0[-1])#res0.x[1] # 
#x0_t = 0 #unbiased
x0_t =  res0.x[1] #0 #res0[1]#0#
'''

drift0_v = res1.x[0]
bound0_v = 6
#bound0_v = res1.x[1] #3.0
ndt_v = res1.x[2]#-np.Inf#
x0_v = res1.x[1] 


noise0 = 3
max_time_t = 15.0
max_time_v = 18
bias = "point"#"centre"##

method = "implicit"


'''
result1 = Model.test_solve_numerical(method, bias, drift0_t, bound0_t, noise0, x0_t, dx, dt, max_time_t)
sim_responses1 = Model_utility.simulate(result1,df_gender_t, dt, ndt_t, max_time_t,s_type_t)
#%%
simdata1 = {'RT':  sim_responses1[1], 'R':  sim_responses1[0]}
simdf1 = pd.DataFrame(simdata1)
simdf1.to_csv('data/simdata_text_F.csv', index=False)
print("simulation is done half.")
Model_utility.plot_density(result1, dt, ndt_t)
'''




result2 = Model.test_solve_numerical(method, bias, drift0_v, bound0_v, noise0, x0_v, dx, dt, max_time_v)
sim_responses2 = Model_utility.simulate(result2,df_gender_v, dt, ndt_v, max_time_v,s_type_v)

simdata2 = {'RT1':  sim_responses2[1], 'R':  sim_responses2[0]}
simdf2 = pd.DataFrame(simdata2)
simdf2.to_csv('data/simdata_video_all(6,3).csv', index=False)


simdf2 = pd.read_csv('data/simdata_video_all(6,3).csv')

#Model_utility.plot_density(result2, dt, ndt_v)

#%%



#Drift* drift, Noise* noise, Bound* bound, Bias* bias, double dx, double dt, double max_time,
#        double drift0, double noise0, double b0)


import seaborn as sns

def plot_rt_density(df1, df2):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    # Create a boolean mask for rows where 'R' is 0
    mask0 = df1['R'] == 0
    mask1 = df2['R'] == 0

    # Create a boolean mask for rows where 'R' is 1
    mask2 = df1['R'] == 1
    mask3 = df2['R'] == 1

    # Use the boolean masks to extract the 'RT' values for each category of 'R'
    RT0 = df1.loc[mask0, 'RT1']
    RT1 = df2.loc[mask1, 'RT1']
    RT2 = df1.loc[mask2, 'RT1']
    RT3 = df2.loc[mask3, 'RT1']

    # Plot the density plots of the 'RT' values for each category of 'R'
    sns.kdeplot(RT0, ax=axs[0], label='Real dataset R=0', color='blue')
    sns.kdeplot(RT1, ax=axs[0], label='Simulated dataset R=0', color='red')
    sns.kdeplot(RT2, ax=axs[1], label='Real dataset R=1', color='green')
    sns.kdeplot(RT3, ax=axs[1], label='Simulated dataset R=1', color='orange')
    
    axs[0].set_xlabel('RT')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Density Plot of RT by Category of R (R==0)')
    axs[0].legend()
    
    axs[1].set_xlabel('RT')
    axs[1].set_ylabel('Density')
    axs[1].set_title('Density Plot of RT by Category of R (R==1)')
    axs[1].legend()
    #plt.savefig("plots/")
    plt.show()

def plot_rt_histogram(df1, df2, binwidth=0.3):
    # Create a boolean mask for rows where 'R' is 0
    mask0 = df1['R'] == 0
    mask1 = df2['R'] == 0

    # Create a boolean mask for rows where 'R' is 1
    mask2 = df1['R'] == 1
    mask3 = df2['R'] == 1

    # Use the boolean masks to extract the 'RT' values for each category of 'R'
    RT0 = df1.loc[mask0, 'RT1']
    RT1 = df2.loc[mask1, 'RT1']
    RT2 = df1.loc[mask2, 'RT1']
    RT3 = df2.loc[mask3, 'RT1']
    # creat two plots, the upper one is for R = 0; the lower one is for R = 1.
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    # Plot the histograms of the 'RT' values for each category of 'R'

    sns.histplot(RT0, ax=axs[0], binwidth=binwidth, alpha=0.5, label='Real dataset R=0', color='blue')
    sns.histplot(RT1, ax=axs[0], binwidth=binwidth, alpha=0.5, label='Simulated dataset R=0', color='red')
    sns.histplot(RT2, ax=axs[1], binwidth=binwidth, alpha=0.5, label='Real dataset R=1', color='green')
    sns.histplot(RT3, ax=axs[1], binwidth=binwidth, alpha=0.5, label='Simulated dataset R=1', color='orange')
    
    axs[0].set_xlabel('RT')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of RT by Category of R=0')
    axs[0].legend()

    axs[1].set_xlabel('RT')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of RT by Category of R=1')
    axs[1].legend()

    plt.show()
    #plt.savefig("plots/")

#%%
# plot_rt_histogram(df)
'''
plot_rt_density(df_gender_t, simdf1)
plot_rt_histogram(df_gender_t, simdf1)



'''
plot_rt_density(df_gender_v, simdf2)
plot_rt_histogram(df_gender_v, simdf2)


#plotname = "plots/multi.pdf"
#save_multi_image(plotname)
