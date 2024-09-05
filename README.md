# Biased Drift Diffusion Model

## Overall

This repository shares all the related codes used for the Biased Drift Diffusion Model parameter estimation and inference on paper _Video-Based Experiments Better Capture Societal Biases in AI’s Ethical Decisions!_ The subjects are facing the binary choice between crashing female or male pedestrians when an Autonomous Vehicle is out of control. The survey is conducted in two different representation formats: picture-based and video-based scenes respectively. The data documented the **Response choice (R)** and **Response Time (RT)** for 40 participants under 6 different scenarios over pedestrians age in two types of scenes. 

The drift-diffusion model (DDM) is applied here to disentangle the influence of subjects ’ initial preferences and their interaction with the survey on the formation of choice biases. DDM inherently models the decision-making process, allowing joint modeling of the subjects’ response times and choice outcomes [^1]. Two DDM parameters are relevant for this study. One key parameter, the starting point, signifies the initial preference toward a particular choice option. Another parameter, the drift rate, denotes the speed of evidence accumulation, reflecting the subject's interaction with the scene. By fitting DDMs to the data obtained from male and female subjects in both the picture-based and video-based scenes, we examine the variation in these parameters and consequent changes in choice outcomes. Furthermore, we investigate the connections between DDM parameters and features of brain activities, shedding light on their association with attentional and emotional aspects. Thus, we ensure the robustness of our findings by triangulating correlations between choice biases, DDM parameters, and brain activities.

##  Repository Details

### The Biased_DDM.sln 
The solution for both the DDM recovery experiment and empirical DDM estimation. Please clone all files except readme and license in this repository such that the solution works.

### **[Model](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/tree/master/Model):** 
This file contains the source codes written in C++ for solving stochastic differential equations of DDM. Please check the algorithm in Suppl. Algorithm of probability density approximation of DDM model of our paper for more details.

This file is used to compile the shared library (.dll) via Microsoft Visual C++ for _Biased_DDM.sln_ using Windows 10.
The detailed procedure for compiling .dll can be checked [tutorial for .dll compiling under Windows System](https://learn.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-dynamic-link-library-cpp?view=msvc-170.)

Before compiling the .dll, [Armadillo](https://arma.sourceforge.net/), [Boost](https://www.boost.org/), [SuperLU](https://portal.nersc.gov/project/sparse/superlu/), and other potential dependent C++ libraries are required to download and add their file locations to `additional dependencies` while compiling the dynamic shared library Model.dll.

Last but not last, after successful compiling Model.dll, please set this shared dynamic library as the dependence path to the following python projects.
### [Recovery_drift_bias_ndt](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/tree/master/Recovery_drift_bias_ndt):

For readers who want to apply a biased DDM model to their dataset, this file is a simple example to replicate and then customize the DDM model.

- [main.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Recovery_drift_bias_ndt/main.py): The main Python file for data simulation (based on the given parameters), estimation, and visualization.
- [Model_utility.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Recovery_drift_bias_ndt/Model_utility.py): A utility file includes functions of simulation, likelihood calculations, and so on, which is imported in `main.py`.
- [data](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/tree/master/Recovery_drift_bias_ndt/data):
  - `df_recovery.csv`: generated response choice (R) and corresponding response time (RT) data with a given true parameter set.
  - `estimate_output.txt`: The output for MLE estimations for all iterations.
  - `likelihood_output.txt`: The output for -2*log_likelihood for each iteration.
  - `simdata.csv`: The simulated R&RT data under MLE.
    
### [Biased_gender_CV](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/tree/master/Biased_gender_CV)

This project includes all codes related to DDM models in our paper.
- **[data](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/tree/master/Biased_gender_CV/data):**
  - **[gender_text_copy.csv](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/data/gender_text_copy.csv)**: The dataframe of picture-based scene includes:
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
   - **[gender_video_copy.csv](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/data/gender_video_copy.csv)**: The dataframe of video-based scene includes: other columns are the same as `gender_text_copy.csv`, except the following.
      - Overall Response Time: `RT` in sec.
      - Deliberation Time: `RT1` in sec.
        
  
> Note that in the video-based scene, Deliberation Time `RT1` is used for the biased-DDM instead of overall Response Time `RT`. The latter includes the time elapse from the start of the scene to the appearance of pedestrians, which shouldn't be considered as decision time. 
 
> The term **"text-based" scene** in the codes or dataset is equivalent to the **"picture-based" scene** in the paper.
>  
- [Model_utility.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Model_utility.py): A utility file includes functions of simulation, likelihood calculations, and so on, which is imported in [CV for hyperparameter.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/CV%20for%20hyperparameter.py),[Subgroup_estimation.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Subgroup%20estimation.py), and [Recovery estimation.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Recovery_estimation.py).
  Specially, unlike [Model_utility.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Recovery_drift_bias_ndt/Model_utility.py) in [Recovery_drift_bias_ndt](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/tree/master/Recovery_drift_bias_ndt) project, there include three sampling functions: `df_draw()`, `df_draw2()`, and `df_draw3()` facilitating the hyperparameter selection, MLE estimation and inference, respectively.

- **[CV for hyperparameter.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/CV%20for%20hyperparameter.py)**:
  This code is used to search for the best hyper-parameter pair based on Cross-validation.
  - Input: **[gender_text_copy.csv](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/data/gender_text_copy.csv)** and **[gender_video_copy.csv](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/data/gender_video_copy.csv)** in `.../Biased_gender_CV/data/`.
  - sampling method: `df_draw()` in [Model_utility.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Model_utility.py). This is a subject-level Leave-one-out sampling method to ensure the each subject has one observation in test dataset.
  - Output: MLE of parameters, `-2*loglikelihood`, `observed choice proportion`, and `predicted choice proportion` of both in-sample and ou-of-sample datasets for each hyper-parameter pair candidate.
- **[Subgroup estimation.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Subgroup%20estimation.py)**:
  This code is used to estimate the MLE and its bootstrapped distributions of subgroups(partition by gender, theta, and so on).
  - Input: **[gender_text_copy.csv](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/data/gender_text_copy.csv)** and **[gender_video_copy.csv](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/data/gender_video_copy.csv)** in `.../Biased_gender_CV/data/`.
  - sampling method:
    - **When estimating bootstrapped distribution:** `df_draw2()` in Model_utility.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Model_utility.py). This resampling method ensures the resampling dataset has the same number of observations (n=6) for each subject, hence let the overall average proportion be the estimator of the mean proportion at the individual level.
    - **When estimating MLE: ** Directly use the subgroup dataset as training data.
  - Output: MLE of parameters, `-2*loglikelihood`, `observed choice proportion`, and `predicted choice proportion` for each subgroup.
 
- **[Recovery_estimation.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Recovery_estimation.py)** + **[Recovery_simulation.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Recovery_simulation.py)**:
  This code is used for showing the good **data recovery (individual-level choice proportion and RT distribution)** by Biased-DDM. Only if the number of observations for each subject is equal, the individual-level choice proportion mean is equivalent to the average proportion for all observations. Hence, a dynamic conditional imputing method is used to balance the dataset.
  - **[Recovery_estimation.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Recovery_estimation.py)**:
    - Input: **[gender_text_copy.csv](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/data/gender_text_copy.csv)** and **[gender_video_copy.csv](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/data/gender_video_copy.csv)**  in `.../Biased_gender_CV/data/`.
    - sampling method: `df_draw3()` in [Model_utility.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Model_utility.py). This imputing method ensures the estimated choice proportion is unbiased to the individual-level choice proportion mean of the whole population. 
    - Output: MLE of parameters, `-2*loglikelihood`, `observed choice proportion`, and `predicted choice proportion` for imputed datasets for each subgroup.
- **[Recovery_simulation.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Recovery_simulation.py)**:
-   Input: estimated output from [Recovery_estimation.py](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/Biased_gender_CV/Recovery_estimation.py).
-   Output: simulated observations in the format of (R, RT). These outputs are used to generate proportion_all.xlsx in `.../DDM_results`.

## [DDM result](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/tree/master/DDM%20result):
This file folder stores the main results in our paper related to the biased-DDM.
- [proportion_all.xlsx](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/DDM%20result/proportion_all.xlsx): The estimated choice proportion and observed proportion for both scenes using all data for 30 iterations for data recovery.
- [DDM_result_test(adjusted).xlsx](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/DDM%20result/DDM_result_test(adjusted).xlsx): The MLE, bootstrap CI, k-s test, and overlapping area for the subgroups.
- **[pure_gender_bs_NF_150.xlsx](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/DDM%20result/pure_gender_bs_NF_150(adjusted).xlsx)**: All the estimation results for MLE and bootstrapped distributions of subgroups.
  The sheets are named by "X1_X2_X3" format, where
  - "X1" stands for scene type, text-based or video-based.
  - "X2" stands for the subgroup partition type, by"all", "gender", "alpha", "beta", and "theta". Specially, "all" stands for no partition rule used.
  - "X3" stands for whether the estimated result is for finding MLE point estimation (30 iterations, finding the estimation with maximum llk) denoted by "mle" or estimating MLE bootstrapped distribution (150 iterations) denoted by "bs".
    For example, "text_all_mle" is the sheet contains the estimation results and corresponding in-sample `-2*loglikelihood`, observed proportion, and estimated proportion for **30 repetitions based on the whole dataset of the text-based (picture-based) scene to find MLE.**
    "video_beta_bs" is the sheet contains the estimation results and corresponding in-sample `-2*loglikelihood`, observed proportion, and estimated proportion for **150 bootstrap iterations based on the subgroup dataset partitioned by beta-bands on the video-based scene to generate MLE distribution.**
- [Hyper_parameter_CV.xlsx](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/DDM%20result/Hyper_parameter_CV.xlsx): Summary output of **CV for hyperparameter.py**.
- [sampling procedures.pdf](https://github.com/LxinWeixL/Biased-Drift-Diffusion-Model/blob/master/DDM%20result/sampling%20procedures.pdf): The sampling procedures for stratified Leave-one-out sampling (p1) and stratified bootstrap sampling (p2).

  
# Ackowledgement and Contributions:

Thanks for the kind help from [Dr. Yi-shin Lin](https://github.com/yxlin) for his contribution to the .cpp source code of the DDM model (in the Model folder). See [Shinn, M., & Lam, N. (2018). pyddm (version 0.5.6).](https://github.com/mwshinn/PyDDM/) for more details. 

Based on .cpp source code, the author of this repository built the dynamic shared library in the Windows system and applied it to our empirical experiments to achieve diverse purposes in Python. Hence, for further questions on the above code and data, feel free to contact the author by [xinwei.li@u.nus.edu](xinwei.li@u.nus.edu) or start an issue in this repository. 








  





## Reference:

[^1]: [Ratcliff, R., Smith, P. L., Brown, S. D., & McKoon, G. (2016). Diffusion decision model: Current issues and history. Trends in cognitive sciences, 20(4), 260-281]([https://pages.github.com/](https://linkinghub.elsevier.com/retrieve/pii/S1364661316000255))
