# Biased Drift Diffusion Model

## Introduction

This repository shares all the related codes used for the Biased Drift Diffusion Model parameter estimation and inference on paper _Does the Information Presentation Format Affect Moral Decisions? Evidence from Neural Responses and Behavioral Model._ The subjects are facing the binary choice between crashing female or male pedestrians when an Autonomous Vehicle is out of control. The survey is conducted in two different representation formats: picture-based and video-based scenes respectively. The data documented the **response choice (R)** and **Response time (RT)** for 40 participants under 6 different scenarios over pedestrians age in two types of scenes. 

the drift-diffusion model (DDM) is applied here to disentangle the influence of subjects ’ initial preferences and their interaction with the survey on the formation of choice biases. DDM inherently models the decision-making process, allowing joint modeling of the subjects’ response times and choice outcomes [^1]. Two DDM parameters are relevant for this study. One key parameter, the starting point, signifies the initial preference toward a particular choice option. Another parameter, the drift rate, denotes the speed of evidence accumulation, reflecting the subject's interaction with the scene. By fitting DDMs to the data obtained from male and female subjects in both the picture-based and video-based scenes, we examine the variation in these parameters and consequent changes in choice outcomes. Furthermore, we investigate the connections between DDM parameters and features of brain activities, shedding light on their association with attentional and emotional aspects. Thus, we ensure the robustness of our findings by triangulating correlations between choice biases, DDM parameters, and brain activities.

## Content of the repository


## Software Prerequisite
The core codes for solving stochastic differential equations of DDM (see Appendix 



The software so far has been built and tested on Ubuntu 22.04.2. Theoretically, the shared library could be compiled to dll using Microsoft Visual C++, although we have not tried it.



## Reference:

[^1]: [Ratcliff, R., Smith, P. L., Brown, S. D., & McKoon, G. (2016). Diffusion decision model: Current issues and history. Trends in cognitive sciences, 20(4), 260-281]([https://pages.github.com/](https://linkinghub.elsevier.com/retrieve/pii/S1364661316000255))
