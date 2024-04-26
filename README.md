# Biased Drift Diffusion Model

## Overall

This repository shares all the related codes used for the Biased Drift Diffusion Model parameter estimation and inference on paper _Does the Information Presentation Format Affect Moral Decisions? Evidence from Neural Responses and Behavioral Model._ The subjects are facing the binary choice between crashing female or male pedestrians when an Autonomous Vehicle is out of control. The survey is conducted in two different representation formats: picture-based and video-based scenes respectively. The data documented the **response choice (R)** and **Response time (RT)** for 40 participants under 6 different scenarios over pedestrians age in two types of scenes. 

The drift-diffusion model (DDM) is applied here to disentangle the influence of subjects ’ initial preferences and their interaction with the survey on the formation of choice biases. DDM inherently models the decision-making process, allowing joint modeling of the subjects’ response times and choice outcomes [^1]. Two DDM parameters are relevant for this study. One key parameter, the starting point, signifies the initial preference toward a particular choice option. Another parameter, the drift rate, denotes the speed of evidence accumulation, reflecting the subject's interaction with the scene. By fitting DDMs to the data obtained from male and female subjects in both the picture-based and video-based scenes, we examine the variation in these parameters and consequent changes in choice outcomes. Furthermore, we investigate the connections between DDM parameters and features of brain activities, shedding light on their association with attentional and emotional aspects. Thus, we ensure the robustness of our findings by triangulating correlations between choice biases, DDM parameters, and brain activities.

## Content of the repository
### The Recovery_drift_bias_ndt.sln 
The solution for both the DDM recovery experiment and empirical DDM estimation. 

### **Model:** 
This file contains the source codes written in C++ for solving stochastic differential equations of DDM. Please check the algorithm in Suppl. Algorithm of probability density approximation of DDM model of our paper for more details.

This file is used to compile the shared library (.dll) via Microsoft Visual C++ for _Recovery_drift_bias_ndt.sln_ using Windows 10.
The detailed procedure for compiling .dll can be checked [tutorial for .dll compiling under Windows System](https://learn.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-dynamic-link-library-cpp?view=msvc-170.)

Before compiling the .dll, [Armadillo](https://arma.sourceforge.net/), [Boost](https://www.boost.org/), [SuperLU](https://portal.nersc.gov/project/sparse/superlu/), and other potential dependent C++ libraries are required to download and add their file locations to `additional dependencies` while compiling the dynamic shared library Model.dll.

Last but not last, after successful compiling Model.dll, please set this shared dynamic library to as the dependence path to the following python projects.
### Recovery_drift_bias_ndt:

For readers who want to apply a biased DDM model to their dataset, this file is a simple example to replicate and then customize the DDM model.

- `main.py`: The main Python file for data simulation (based on the given parameters), estimation, and visualization.
- `Model_utility.py`: A utility file includes functions of simulation, likelihood calculations, and so on, which is imported in `main.py`.
- data:
  - `df_recovery.csv`: generated response choice (R) and corresponding response time (RT) data with a given true parameter set.
  - `estimate_output.txt`: The output for MLE estimations for all iterations.
  - `likelihood_output.txt`: The output for -2*log_likelihood for each iteration.
  - `simdata.csv`: The simulated R&RT data under MLE.
    
### Biased_gender_CV

This project includes all codes related to DDM models in our paper.

- `Model_utility.py`: A utility file includes functions of simulation, likelihood calculations, and so on, which is imported in the following .py files.
  Specially, unlike `Model_utility.py` in `Recovery_drift_bias_ndt` project, there includes three sampling function: `df_draw()`, `df_draw2()`, and `df_draw3()` facilitating the hyperparameter selection, MLE estimation and reference, respectively.

-  

  




  





## Reference:

[^1]: [Ratcliff, R., Smith, P. L., Brown, S. D., & McKoon, G. (2016). Diffusion decision model: Current issues and history. Trends in cognitive sciences, 20(4), 260-281]([https://pages.github.com/](https://linkinghub.elsevier.com/retrieve/pii/S1364661316000255))
