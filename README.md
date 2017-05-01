# STA-663-Final-Project

The LaTex code of the report can be found through the link:https://www.overleaf.com/read/ssdybrngszwq

The github repository: https://github.com/CeciliaShi/STA-663-Final-Project

Source code can be found in fastfsr/fastfsr/__init__.py

Instructions to install the package:

a. download the folder fastfsr

b. after download go into fastfsr, and create a new Jupyter Notebookand 

c. type in "!pip install ." 

d. type in "import fastfsr"


This repository contains material for the STA 663 final project.
This repository contains code, examples and data for implementing the fast false selection rate algorithm proposed by 
Dennis D.Boos, Leonard A. Stefanski, and Yujun Wu in the paper Fast FSR Variable Selection with Applications to Clinical Trials.

The data folder contains two sets of simulated data with responses from 5 models for each simulated data. 
It also contains the NCAA Data on 6 year college graduation rates. Both the simulated data and the NCAA data are referenced from
http://www4.stat.ncsu.edu/~boos/var.select/

Following the paper and the R code written by the authors, we implemented the key functions in Python. These functions include:
* fsr_fast: implementation of the Fast FSR algorithm
* fsr_fast_pv:  Fast FSR based on summary p-values from forward selection
* lasso_fit: using cross validtion to select the best lambda using Scikit-learn and returning the fitted values, model size etc from lasso
* reg_subset: regression subset selection
* gic: gives model size including intercept of min BIC model
* bic_sim: chooses from FAS using minimum BIC

We also included detailed examples of using these functions with tables and plots. The example code can be found in the github repository. The examples include:
* Examples of how to use the key functions in the package (fsr_fast), users can simply change the fsr_fast function with bic_sim or lasso_fit to run examples of BIC and Lasso

The simulation file consists of model comparison of Fast FSR, BIC and LASSO using the simulated data sets with results.

The repository also have a file Compare_run_tim.ipynb to show the possible optimizations using Cython.

