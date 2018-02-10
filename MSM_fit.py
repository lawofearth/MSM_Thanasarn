import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
# import MSM_util
from MSM_util import *
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import sys, scipy, numpy

# print(scipy.__version__, numpy.__version__, sys.version_info)

GLD = pd.read_excel('data_GVZ_GLD.xlsx')
date_GLD = GLD.iloc[:,3]
GLD = GLD.loc[:,'GLD']
GLD_d = GLD - np.mean(GLD)
# plt.plot(date_GLD,GLD_d)
data = GLD_d

kbar = 5
startingvals = []
LB = [1, 1, 0.001, 0.0001]
UB = [1.99, 50, 0.99999, 5]


# Grid search for starating values
# TODO: (try sklearn.GridsearchCV) or map lambda instead of double for-loop
input_param, LLS = MSM_starting_values(data, startingvals, kbar)

# create a set of Parameters
params = Parameters()
params.add('m0',   value= input_param[0],  min=LB[0], max = UB[0])
params.add('b', value= input_param[1],  min=LB[1], max = UB[1])
params.add('gamma_k', input_param[2],  min=LB[2], max = UB[2])
params.add('sigma', value= input_param[3],  min=LB[3])

minner = Minimizer(MSM_likelihood_new, params, fcn_args=(kbar, data))
result = minner.minimize(method='slsqp')
print(result.params)

for element in result.params : print(element)

print(result)


