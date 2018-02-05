from lmfit import model, minimize, Minimizer, Parameters, Parameter, report_fit
from MSM_util import *
import pandas as pd
import numpy as np

GLD = pd.read_excel('data_GVZ_GLD.xlsx')
date_GLD = GLD.iloc[:,3]
GLD = GLD.loc[:,'GLD']
GLD_d = GLD - np.mean(GLD)
kbar = 5
data = GLD_d


A_template = T_mat_temp(kbar)

input_param = [1.5, 1.5, 0.1, 383.4037]
m0 = input_param[1]
b = input_param[0]
gamma_k = input_param[2]
sigma = input_param[3]

# create a set of Parameters
params = Parameters()
params.add('m0', value=1.9, min=1.2, max=1.8)


# do fit, here with leastsq model
minner = Minimizer(MSM_likelihood_new, params, fcn_args=(b, gamma_k, sigma, kbar, data, A_template))
result = minner.minimize()
