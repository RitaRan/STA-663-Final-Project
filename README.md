# STA-663-Final-Project

This repository contains material for the STA 663 final project.
This repository contains code, examples and data for implementing the fast false selection rate algorithm proposed by 
Dennis D.Boos, Leonard A. Stefanski, and Yujun Wu in the paper Fast FSR Variable Selection with Applications to Clinical Trials.

The data folder contain two sets of simulated data with responses from 5 models for each simulated data. 
It also contains the NCAA Data on 6 year college graduation rates. Both the simulated data and the NCAA data are referenced from
http://www4.stat.ncsu.edu/~boos/var.select/

Following the paper and the R code written by the authors, we implemented the key functions in Python. These functions include:
* fsr_fast: implementation of the Fast FSR algorithm
* fsr_fast_pv:  Fast FSR based on summary p-values from forward selection
* lasso_fit: using cross validtion to select the best lambda using Scikit-learn and returning the fitted values, model size etc from lasso
* reg_subset: regression subset selection
* gic: gives model size including intercept of min BIC model
* bic_sim: chooses from FAS using minimum BIC

We also included detailed examples of using these functions with tables and plots. 

The simulation file consists of model comparison of Fast FSR, BIC and LASSO using the simulated data sets with results.
