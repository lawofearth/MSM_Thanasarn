import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
# import MSM_util
from MSM_util import *

GLD = pd.read_excel('data_GVZ_GLD.xlsx')
date_GLD = GLD.iloc[:,3]
GLD = GLD.loc[:,'GLD']
GLD_d = GLD - np.mean(GLD)
# plt.plot(date_GLD,GLD_d)
data = GLD_d

kbar = 5
startingvals = []
LB = [1, 1, 0.001, 0.0001]
UB = [50, 1.99, 0.99999, 5]

# set up atemplate for a transition matrix
A_template = MSM_util.T_mat_temp(kbar)

# Grid search for starating values
# TODO: (try sklearn.GridsearchCV) or map lambda instead of double for-loop
input_param, LLS = MSM_starting_values(data, startingvals, kbar, A_template)

# Minimize multivariate

b = input_param[0]
m0 = input_param[1]
gamma_k = input_param[2]
sigma = input_param[3]


print(LLS)

LL = MSM_util.MSM_likelihood_new(b, m0, gamma_k, sigma, kbar, data, A_template)

# MSM.MSM_likelihood(input_param, kbar, data, A_template, estim_flag)
print(LL)