import Model
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from IPython.display import Image
import Model_utility


#############################################################
bound0 = 3.0
dx = 0.01
dt = 0.01
max_time = 14.0
xs = Model.test_x_domain(bound0, dx, dt, max_time)

drift0 = 1.05
noise0 = 2.5
ndt = 0.30 # 300 ms non-decision time
method = "implicit"
bias = "point"
x0 = -1.0

#if __name__ == "__main__":

  

#############################################################
# The probability of the system at the time step 0.
res = Model.test_get_starting_pdf(bias, drift0, bound0, noise0, x0, dx, dt, max_time)
# print(res)

print("The highest density at the time step 0 is at ", np.where(res == 1)[0], "\n")
print("The discrete evidence steps are,", len(res))


#############################################################
x = res
y = np.arange(-bound0, bound0, (2*bound0) / len(res))
plt.plot(x, y)
plt.xlabel('Probability density')
plt.ylabel('Evidence space')
plt.grid()
plt.tight_layout()
plt.savefig("plots/1.png")
drift0 = 1.05
noise0 = 2.5
bound0 = 3.0
dx = 0.01
dt = 0.01
max_time = 14.0

ndt = 0.80
dt = 0.01
method = "implicit"

x0 = -1.0
bias = "point"

result = Model.test_solve_numerical(method, bias, drift0, bound0, noise0, x0, dx, dt, max_time)


Model_utility.plot_density(result, dt, ndt).savefig('plots/2.png')


##################################################
n = 512
sim_emp_responses = Model_utility.simulate_simple(result, n, dt, ndt)
R = sim_emp_responses[0]
RT = sim_emp_responses[1]

# We assumed this simulation data as the true data and aimed to recover 
# the paraemter generated the data. 
data = {'RT': RT , 'R': R}
df = pd.DataFrame(data)
df.to_csv('data/df_recovery.csv', index=False)

##################################################
def objective_function(x):
    drift0 = x[0]
    start_point = x[1]
    ndt = x[2]

    bound0 = 3.0  # Fixed bound0 at 3.0
    noise0 = 2.5
    dx = 0.01
    dt = 0.01
    max_time = 14.0

    bias = "point"
    method = "implicit"
    
    estimated_pdf = Model.test_solve_numerical(method, bias, drift0, bound0, noise0, start_point, dx, dt, max_time)

    # RT and R were passed on as global variables. I have not found a way to
    # pass them on as arguments to the objective function. 
    # TODO: look up the scipy.minimize document to see if there is a way to
    # that.
    res = Model_utility.calculate_LL(RT, R, estimated_pdf, dt, ndt)
    # Write the value of res to a file, so we can track the progress of 
    # the optimization to see if it converges to a stable value.
    with open('data/likelihood_output.txt', 'a') as f:
        f.write(str(res) + '\n')

    return res
    
def callback(xk):
    # ".join(map(str, xk))" is to remove the square brackets from the array.
    with open('data/estimate_output.txt', 'a') as f:
        f.write('   '.join(map(str, xk)) + '\n')
    # Print the estimate to the screen so we can also see its progress.
    # Comment it out, if you run the code on a cluster.
    print(xk)
    

############################################################
# In this simple case, it usually converges before 100 iterations
# Initial guess for the parameters. I renamed it to avoid using the same 
# variable name as the starting point.
initial_guess = [0.5, 0.5, 0.5]  
res = minimize(objective_function, initial_guess, method='Nelder-Mead', options={'maxiter': 150}, callback=callback)


############################################################
print(res)

###########################################################
file_path = 'data/estimate_output.txt'
file_like = 'data/likelihood_output.txt'
est_df = pd.read_csv(file_path, sep='   ', header=None, engine='python')
lik_df = pd.read_csv(file_like, sep='   ', header=None, engine='python')

est_df['likelihood'] = lik_df
est_df.columns = ['Drift', 'Bias', 'Non-decision time', 'likelihood']
est_df.head()

############################################################
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
    #plt.show()
    # Xinwei's edition
    plt.savefig("plots/line_chart.png")

# Xinwei's edition 
plot_line_chart(est_df)

######################################################################

bias = "point"
x0 = res.x[1]

drift0 = res.x[0]
ndt = res.x[2] 

dx = 0.01
dt = 0.01

bound0 = 3.0  # Fixed bound0 at 3.0
noise0 = 2.5
max_time = 14.0

method = "implicit"
result = Model.test_solve_numerical(method, bias, drift0, bound0, noise0, x0, dx, dt, max_time)
sim_responses = Model_utility.simulate(result, df, dt, ndt)

simdata = {'RT':  sim_responses[1], 'R':  sim_responses[0]}
simdf = pd.DataFrame(simdata)
simdf.to_csv('data/simdata.csv', index=False)

#######################################################################

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
    RT0 = df1.loc[mask0, 'RT']
    RT1 = df2.loc[mask1, 'RT']
    RT2 = df1.loc[mask2, 'RT']
    RT3 = df2.loc[mask3, 'RT']

    # Plot the density plots of the 'RT' values for each category of 'R'
    sns.kdeplot(RT0, ax=axs[0], label='df1 R=0', color='blue')
    sns.kdeplot(RT1, ax=axs[0], label='df2 R=0', color='red')
    sns.kdeplot(RT2, ax=axs[1], label='df1 R=1', color='green')
    sns.kdeplot(RT3, ax=axs[1], label='df2 R=1', color='orange')
    
    axs[0].set_xlabel('RT')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Density Plot of RT by Category of R (R==0)')
    axs[0].legend()
    
    axs[1].set_xlabel('RT')
    axs[1].set_ylabel('Density')
    axs[1].set_title('Density Plot of RT by Category of R (R==1)')
    axs[1].legend()
    
    #plt.show()
    # Xinwei's edition
    plt.savefig("plots/rt_density")

def plot_rt_histogram(df1, df2, binsize=50):
    # Create a boolean mask for rows where 'R' is 0
    mask0 = df1['R'] == 0
    mask1 = df2['R'] == 0

    # Create a boolean mask for rows where 'R' is 1
    mask2 = df1['R'] == 1
    mask3 = df2['R'] == 1

    # Use the boolean masks to extract the 'RT' values for each category of 'R'
    RT0 = df1.loc[mask0, 'RT']
    RT1 = df2.loc[mask1, 'RT']
    RT2 = df1.loc[mask2, 'RT']
    RT3 = df2.loc[mask3, 'RT']

    # Plot the histograms of the 'RT' values for each category of 'R'
    plt.hist(RT0, bins=binsize, alpha=0.5, label='df1 R=0', color='blue')
    plt.hist(RT1, bins=binsize, alpha=0.5, label='df2 R=0', color='red')
    plt.hist(RT2, bins=binsize, alpha=0.5, label='df1 R=1', color='green')
    plt.hist(RT3, bins=binsize, alpha=0.5, label='df2 R=1', color='orange')
    
    plt.xlabel('RT')
    plt.ylabel('Frequency')
    plt.title('Histogram of RT by Category of R')
    plt.legend()
    #plt.show()
    # Xinwei's edition
    plt.savefig("plots/rt_hist.png")
    
################################################################################

plot_rt_density(df, simdf)
plot_rt_histogram(df, simdf)

#################################################################################

#  1.0915791263638157   -0.9839271780233605   0.80893928452614
# true value: 1.05,-1,0.3