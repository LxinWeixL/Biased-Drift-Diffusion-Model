# Biased Drift Diffusion Model

## Overall

This repository shares all the related codes used for the Biased Drift Diffusion Model parameter estimation and inference on paper _Does the Information Presentation Format Affect Moral Decisions? Evidence from Neural Responses and Behavioral Model._ The subjects are facing the binary choice between crashing female or male pedestrians when an Autonomous Vehicle is out of control. The survey is conducted in two different representation formats: picture-based and video-based scenes respectively. The data documented the **response choice (R)** and **Response time (RT)** for 40 participants under 6 different scenarios over pedestrians age in two types of scenes. 

The drift-diffusion model (DDM) is applied here to disentangle the influence of subjects ’ initial preferences and their interaction with the survey on the formation of choice biases. DDM inherently models the decision-making process, allowing joint modeling of the subjects’ response times and choice outcomes [^1]. Two DDM parameters are relevant for this study. One key parameter, the starting point, signifies the initial preference toward a particular choice option. Another parameter, the drift rate, denotes the speed of evidence accumulation, reflecting the subject's interaction with the scene. By fitting DDMs to the data obtained from male and female subjects in both the picture-based and video-based scenes, we examine the variation in these parameters and consequent changes in choice outcomes. Furthermore, we investigate the connections between DDM parameters and features of brain activities, shedding light on their association with attentional and emotional aspects. Thus, we ensure the robustness of our findings by triangulating correlations between choice biases, DDM parameters, and brain activities.

## Content of the repository
### The Biased_DDM.sln 
The solution for both the DDM recovery experiment and empirical DDM estimation. 

### **Model:** 
This file contains the source codes written in C++ for solving stochastic differential equations of DDM. Please check the algorithm in Suppl. Algorithm of probability density approximation of DDM model of our paper for more details.

This file is used to compile the shared library (.dll) via Microsoft Visual C++ for _Biased_DDM.sln_ using Windows 10.
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
- data:
  - `gender_text_copy.csv`: The dataframe of picture-based scene includes:
    - Subject ID:`s`
    - Response Time: `RT` in sec.
    - Display format: `M`
    - Scenario Type: `Scenario_Type`
    - Response: `R`. R=1 when crashing the male; otherwise, R = 0.
    - Subject gender: `sex`. sex = 0, when subject is male.
    - Subject age: `age`. age = 0, when subject is younger or equal to 24 years old.
    - theta-bands subgroup: `theta`. theta= 0 if the subject has a lower theta-band frequency compared to the group median.
    - beta-bands subgroup: `theta`. beta= 0 if the subject has a lower beta-band frequency compared to the group median.
    - alpha-bands subgroup: `theta`. alpha= 0 if the subject has a lower alpha-band frequency compared to the group median.
 - `gender_video_copy.csv`: The dataframe of video-based scene includes: other columns are the same as `gender_text_copy.csv`, except the following.
    - Overall Response Time: `RT` in sec.
    - Deliberation Time: `RT1` in sec. 
  
- `Model_utility.py`: A utility file includes functions of simulation, likelihood calculations, and so on, which is imported in `hyper_parameter CV.py`,`gender_subject.py`, and `bootstrap estimation.py`.
  Specially, unlike `Model_utility.py` in `Recovery_drift_bias_ndt` project, there includes three sampling function: `df_draw()`, `df_draw2()`, and `df_draw3()` facilitating the hyperparameter selection, MLE estimation and reference, respectively.

- `CV for hyperparameter.py`:
  This code is used to search for the best hyper-parameter pair based on Cross-validation.
  - Input: `gender_text_copy.csv` and `gender_video_copy.csv` in `.../Biased_gender_CV/data/`.
  - sampling method: `df_draw()` in `Model_utility.py`. This is a subject-level Leave-one-out sampling method to ensure the each subject has one observation in test dataset.
  - Output: MLE of parameters, `-2*loglikelihood`, `observed choice proportion`, and `predicted choice proportion` of both in-sample and ou-of-sample datasets for each hyper-parameter pair candidate.
- `Bootstrap estimation.py`:
  This code is used to estimate the MLE distributions of subgroups(partition by gender, theta, and so on) by the bootstrap method.
  - Input: `gender_text_copy.csv` and `gender_video_copy.csv` in `.../Biased_gender_CV/data/`.
  - sampling method: `df_draw2()` in `Model_utility.py`. This resampling method ensures the resampling dataset has the same number of observations (n=6) for each subject, hence let the overall average proportion be the estimator of the mean proportion at the individual level. 
  - Output: MLE of parameters, `-2*loglikelihood`, `observed choice proportion`, and `predicted choice proportion` for bootstrapped datasets for each subgroup.
 
- `Recovery_estimation.py` + `Recovery_simulation.py`:
  This code is used for showing the good **data recovery (individual-level choice proportion and RT distribution)** by Biased-DDM. Only if the number of observations for each subject is equal, the individual-level choice proportion mean is equivalent to the average proportion for all observations. Hence, a dynamic conditional imputing method is used to balance the dataset.

  - Input: `gender_text_copy.csv` and `gender_video_copy.csv` in `.../Biased_gender_CV/data/`.
  - sampling method: `df_draw3()` in `Model_utility.py`. This imputing method ensures the estimated choice proportion is unbiased to the individual-level choice proportion mean of the whole population. 
  - Output: MLE of parameters, `-2*loglikelihood`, `observed choice proportion`, and `predicted choice proportion` for imputed datasets for each subgroup.

  




  





## Reference:

[^1]: [Ratcliff, R., Smith, P. L., Brown, S. D., & McKoon, G. (2016). Diffusion decision model: Current issues and history. Trends in cognitive sciences, 20(4), 260-281]([https://pages.github.com/](https://linkinghub.elsevier.com/retrieve/pii/S1364661316000255))
