import numpy as np
import matplotlib.pyplot as plt

def plot_density(input, dt, ndt):
    """
    Plots the probability density of the output from the solve_numerical function.

    Parameters:
    -----------
    input : list
        A list of three arrays containing the data to be plotted.

    Returns:
    --------
    None
        The function only plots the data.

    Example:
    --------
    >>> plot_density(object)
    """
    
    ndt_step = int(ndt/dt)
    ndt_zeros = np.zeros(ndt_step) 
    nO = len(input[0])
    nX = len(input[1])
    nU = len(input[2])
    y0 = np.concatenate((ndt_zeros, input[0]))
    y1 = np.concatenate((ndt_zeros, input[1]))
    y2 = input[2]
    x0 = np.arange(0, nO+ndt_step, 1)
    x1 = np.arange(0, nX+ndt_step, 1)
    x2 = np.arange(0, nU, 1)
    ymax = max([y0.max(), y1.max()])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ax[0].plot(x0, y0, color='red', marker='o')
    ax[0].set_title('Positive boundary')
    ax[0].set_xlabel('Time step')
    ax[0].set_ylabel('Probability density')
    ax[0].set_ylim([0, ymax])
    ax[0].grid()
    ax[1].plot(x1, y1, color ='red', marker ='o')
    ax[1].set_title('Negative boundary')
    ax[1].set_xlabel('Time step')
    ax[1].set_ylim([0, ymax])
    ax[1].grid()
    
    ax[2].plot(y2, x2, color ='red', marker ='o')
    ax[2].set_title('The remaining density \nat the last time step')
    ax[2].set_ylabel('Evidence step')
    ax[2].grid()
    return fig
    
def calculate_cdf(input):
    """
    Calculates the cumulative distribution function of the input data.

    Parameters:
    -----------
    input : list
        A list of three arrays containing the data to be plotted.

    Returns:
    --------
    array-like
        An array containing the cumulative distribution function of the input data.

    Example:
    --------
    >>> calculate_cdf(input)
    """
    y0 = input[0]
    y1 = input[1]
    y = np.column_stack((y0, y1))
    yy = y.sum(axis=1)
    cdf = np.cumsum(yy)
    return cdf

def calculate_LL(RT, R, estimated_pdf, dt, ndt, max_time, print_warning = False):
    """
    Calculates the negative 2, log-likelihood of the input data.

    Parameters:
    -----------
    RT : array-like
        An array containing the reaction times.
    R : array-like
        An array containing the responses.
    estimated_pdf : list
        A list of two arrays containing the estimated probability density functions.
    dt : float
        The time step.
    ndt : float
        non-decision time.
    max_time: float
        max_time for response time.
    print_warning : bool, optional
        If True, prints a warning message when likelihood contains 0. Default is False.

    Returns:
    --------
    float
        The log-likelihood of the input data.

    Example:
    --------
    >>> calculate_LL(RT, R, [pdf_0, pdf_1], 0.1)
    """



    n = len(RT) 
    DT = RT- ndt
    
    ''' only uased for video based scene
    for i in range(n):
        if RT[i]+ndt >= max_time:
            DT[i] = RT[i]
        else:
            DT[i] = max(RT[i]-ndt,0)
    '''           
    index = DT/dt
    likelihood = np.zeros(n)
    

    for i in range(n):
        if R[i] == 0:
            probability = estimated_pdf[0]
        else:
            probability = estimated_pdf[1]
          
        int_index = int(index[i])
        
        # make sure RT-ndt positive
        if int_index<=0: 
           likelihood[i]=0
        else:
            likelihood[i] = probability[int_index]
        

        #likelihood[i] = probability[int_index]

    # consider the boundary the ndt.
    idx_0 = np.where(likelihood == 0)
    if (len(idx_0) > 0):
        if print_warning: 
            print("Warning: likelihood contains 0")
        likelihood[idx_0] = 1e-10
        
    LL = np.log(likelihood)
    return -2*LL.sum()


def simulate(input, df, dt, ndt, max_time):
    result = input
    
    total_probability = np.sum(result[0]) + np.sum(result[1]) + np.sum(result[2])
    positive_probability = np.sum(result[0]) 
    negative_probability = np.sum(result[1]) 
    remaining_probability = np.sum(result[2])
    print("Total probability: ", total_probability)
    print("Probability of hitting the positive boundary", round(positive_probability, 3))
    print("Probability of hitting the negative boundary", round(negative_probability, 3))
    print("Remaining probability at the last time step", round(remaining_probability, 3))

    #proportions = df['TrueR'].value_counts(normalize=True)
    proportions = df['R'].value_counts(normalize=True)
    print("The data showed the probability of hitting the positive boundary", round(proportions[0], 3))
    print("The data showed the probability of hitting the negative boundary", round(proportions[1], 3))
    
    y0 = result[0]
    y1 = result[1]
    y = np.column_stack((y0, y1))
    yy = y.sum(axis=1)
    cdf = np.cumsum(yy)
    
    n = len(df)
    random_draws = np.random.uniform(low=0.0, high=1.0, size=n)
    index = np.searchsorted(cdf, random_draws)
    
    # Correct the discretization error in the random draws
    for i in range(n):
        if random_draws[i] > cdf.max():
            index[i] = index[i] - 1
    

    ## Define RT based on whether close to stopline
    RT = dt * index + ndt
    
    ''' only for video-based Scene
    for i in range(n):
        if RT[i]>max_time-ndt:
            RT[i] = RT[i]
        else:
            RT[i] = RT[i]+ndt
    '''      
    p = y0[index]
    q = y1[index]
    R = np.zeros(n)
    for i in range(n):
        pp = p[i] / (p[i]+q[i])
        qq = q[i] / (p[i]+q[i])
        prob = np.array( [pp/(pp+qq), qq/(pp+qq)]).reshape(2,)
        R[i] = np.random.choice([0, 1], size=1, p = prob)[0]
    
    return [R, RT]

def simulate_simple(input, n, dt, ndt):
    result = input
    
    cdf = calculate_cdf(input)
    random_draws = np.random.uniform(low=0.0, high=1.0, size=n)
    index = np.searchsorted(cdf, random_draws)
    
    # Correct the discretization error in the random draws
    for i in range(n):
        if random_draws[i] > cdf.max():
            index[i] = index[i] - 1
    RT = dt * index + ndt             
    
    y0 = input[0]
    y1 = input[1]
    p = y0[index]
    q = y1[index]
    R = np.zeros(n)
    for i in range(n):
        pp = p[i] / (p[i]+q[i])
        qq = q[i] / (p[i]+q[i])
        prob = np.array( [pp/(pp+qq), qq/(pp+qq)]).reshape(2,)
        R[i] = np.random.choice([0, 1], size=1, p = prob)[0]
    
    return [R, RT]

