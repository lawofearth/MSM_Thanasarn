import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import MSM

GLD = pd.read_excel('data_GVZ_GLD.xlsx')
date_GLD = GLD.iloc[:,3]
GLD = GLD.loc[:,'GLD']
GLD_d = GLD - np.mean(GLD)
kbar = 5
# plt.plot(date_GLD,GLD_d)
data = GLD_d

input_param = [1.5, 1.4292, 0.1, 383.4037]
# [6, 1, 1, 5]
# [50, 1.99,0.945585657754, 5]
# [1.5, 1.4292, 0.1, 383.4037]
b = input_param[0]
m0 = input_param[1]
gamma_k = input_param[2]
sigma = input_param[3]

estim_flag = []
A_template = MSM.T_mat_temp(kbar)

LL, LLs = MSM.MSM_likelihood_new(b, m0, gamma_k, sigma, kbar, data, A_template)

# MSM.MSM_likelihood(input_param, kbar, data, A_template, estim_flag)
print(LL)