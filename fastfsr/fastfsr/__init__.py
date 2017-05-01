import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.linear_model as sl
import scipy.stats as st
import cython
import time
from tqdm import *
from progress.bar import Bar

def reg_subset(x, y):
    
    lm = sl.LinearRegression()
    
    (n,m) = x.shape
    in_ = []
    out_ = list(range(m))
    rss = np.zeros(m+1)
    rss[0] = sum((y-np.mean(y))**2)
    if(m>=n): ml = n-5
    else: ml = m 

    for pi in range(m):
        rss_find = []
        for i in out_:
            fit_X = pd.DataFrame(x.ix[:, in_ + [i]])
            lm.fit(fit_X, y)
            pred = lm.predict(fit_X)
            rss_find.append(sum((pred-y)**2))
        min_idx = np.argmin(rss_find)
        min_var = out_[min_idx]
        rss[pi+1] = np.min(rss_find)
        in_.append(min_var)
        del out_[min_idx]
    
    in_ = np.array(in_)
    in_var = x.columns[in_]
    vm = np.array(range(ml))
    pv_org = 1 - st.f.cdf((rss[vm] - rss[vm+1])*(n-(vm+2))/rss[vm+1],
                     1,n-(vm+2))
    return (in_,np.array(in_var),rss, pv_org)

def lasso_fit(x, y):
    
    #Define the alpha values to test
    alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]
    
    #Initialize the dataframe to store coefficients
    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,x.shape[1]+1)]
    ind = [str(alpha_lasso[i]) for i in range(0,10)]
    coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)
    
    for i in range(10):
        ls = sl.Lasso(alpha=alpha_lasso[i],normalize=True, max_iter=1e5)
        ls.fit(x,y)
        y_pred = ls.predict(x)
        #Return the result in pre-defined format
        rss = sum((y_pred-y)**2)
        ret = [rss]
        ret.extend([ls.intercept_])
        ret.extend(ls.coef_)
        coef_matrix_lasso.iloc[i,] = ret
    
    exist = np.sum(coef_matrix_lasso.ix[:,2:]==0, axis = 1)!=x.shape[1]
    if(sum(exist)==0):
        size = 0
        lm = sl.LinearRegression()
        intercept = pd.DataFrame(np.ones(x.shape[0]))
        lm.fit(intercept, y)
        fitted = lm.predict(intercept)
    else:
        alpha = pd.to_numeric(coef_matrix_lasso.index[exist][-1])
        ls = sl.Lasso(alpha=alpha,normalize=True, max_iter=1e5)
        ls.fit(x,y)
        fitted = ls.predict(x)
        index = np.array(range(len(ls.coef_)))[ls.coef_!=0]
        size = len(index)
        
    # get residuals
    residual = sum((y-fitted)**2)
    df_residual = x.shape[0] - size - 1
    
    return {'fitted':fitted, 'residual':residual, 'df_residual':df_residual, 
            'size':size, 'index': index} 

def fsr_fast_pv(pv_orig, m, gam0 = 0.05, digits = 4, printout = True, plot = True):
    m1 = len(pv_orig)
    ng = m1+1
    (pvm,alpha,alpha2) = helper(pv_orig, m1)
    S = np.zeros(ng)
    for j in range(1, ng):
        S[j] = sum(pvm <= alpha[j])

    # calculate gamma hat
    (ghat2,ghat) = helper2(S,alpha,alpha2,m1,m,ng)
    zp = pd.DataFrame({'a': np.concatenate([alpha, alpha2]), 'g': np.concatenate([ghat, ghat2])})
    zp.sort_values(by =['a', 'g'], ascending = [True, False], inplace = True)
    
    # largest gamma hat and index
    gamma_max = np.argmax(zp['g'])
    
    alpha_max = zp['a'][gamma_max]

    # model size with ghat just below gam0
    ind = np.logical_and(ghat <= gam0, alpha <= alpha_max)*1
    Sind = S[np.max(np.where(ind > 0))]
    
    # calculate alpha_F
    alpha_fast = (1 + Sind)*gam0/(m - Sind)
    
    # size of model no intercept
    size1 = sum(pvm <= alpha_fast)
    
    # generate plot
    if plot==True:
        plt.plot(zp['a'], zp['g'], marker = 'o', markersize = 6)
        plt.ylabel('Estimated Gamma')
        plt.xlabel('Alpha')
        pass

    df1 = pd.DataFrame({'pval': pv_orig, 'pvmax': pvm, 'ghigh': ghat2, 'glow': ghat[1:ng]}, columns = ['pval', 'pvmax', 'ghigh', 'glow'])
    df2 = pd.DataFrame({'m1': m1, 'm': m, 'gam0': gam0, 'size': size1, 'alphamax': alpha_max, 'alpha_fast': alpha_fast}, columns = ['m1', 'm', 'gam0', 'size', 'alphamax', 'alpha_fast'], index=[0])
    if printout == True:
        print(df1,df2)
    return(np.round(df1, digits), np.round(df2, digits), ghat)

def lasso_fit(x, y):
    
    #Define the alpha values to test
    alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]
    
    #Initialize the dataframe to store coefficients
    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,x.shape[1]+1)]
    ind = [str(alpha_lasso[i]) for i in range(0,10)]
    coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)
    
    for i in range(10):
        ls = sl.Lasso(alpha=alpha_lasso[i],normalize=True, max_iter=1e5)
        ls.fit(x,y)
        y_pred = ls.predict(x)
        #Return the result in pre-defined format
        rss = sum((y_pred-y)**2)
        ret = [rss]
        ret.extend([ls.intercept_])
        ret.extend(ls.coef_)
        coef_matrix_lasso.iloc[i,] = ret
    
    exist = np.sum(coef_matrix_lasso.ix[:,2:]==0, axis = 1)!=x.shape[1]
    if(sum(exist)==0):
        size = 0
        lm = sl.LinearRegression()
        intercept = pd.DataFrame(np.ones(x.shape[0]))
        lm.fit(intercept, y)
        fitted = lm.predict(intercept)
    else:
        alpha = pd.to_numeric(coef_matrix_lasso.index[exist][-1])
        ls = sl.Lasso(alpha=alpha,normalize=True, max_iter=1e5)
        ls.fit(x,y)
        fitted = ls.predict(x)
        index = np.array(range(len(ls.coef_)))[ls.coef_!=0]
        size = len(index)
        
    # get residuals
    residual = sum((y-fitted)**2)
    df_residual = x.shape[0] - size - 1
    
    return {'fitted':fitted, 'residual':residual, 'df_residual':df_residual, 
            'size':size, 'index': index}      

def gic(rss, m, n, z):
    """gic gives model size including intercept of min BIC model"""
    t1 = z*np.arange(2, m + 3) + n*np.log(rss/n) + n + 2*n*np.log(np.sqrt(2*np.pi))
    t2 = np.argmin(t1) + 1
    return(t2)

def bic_sim(x, y):
    """chooses from FAS using minimum BIC"""
    lm = sl.LinearRegression()
    (n,m) = x.shape
    out_x = reg_subset(x, y)
    rss = out_x[2]
    vorder = out_x[0]
    if m > n:
        rss = rss[0:60]
        m = 60
    # model size including intercept
    bic = gic(rss, m, n, np.log(n))
    if bic > 1:
        index = vorder[0:bic - 1]
        x = x.ix[:,index]
        x_ind = index
    else:
        x_ind = 0
    if bic == 1:
        intercept = pd.DataFrame(np.ones(n))
        mod = lm.fit(intercept, y)  
        fitted = lm.predict(intercept)
        residual = sum((y - fitted)**2)
        df_residual = n - (bic - 1) - 1
    else:
        mod = lm.fit(x, y)
        fitted = lm.predict(x)
        residual = sum((y - fitted)**2)
        df_residual = n - (bic - 1) - 1
    return dict({'fitted': fitted, 'residual': residual, 'df_residual': df_residual, 'size': bic - 1, 'index': x_ind})