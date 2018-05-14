import numpy as np
import pandas as pd
from scipy.stats import chi2
from pandas import DataFrame as df
from numpy import matlib
from scipy.optimize import fminbound, minimize, fsolve
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date
#import winsound
from pylab import rcParams
import pickle
from datetime import date
from time import time
import statsmodels.api as sm


def t_mat_temp(kbar):
    """
    Build template for kbar
    :param kbar:
    :return:
    """
    A = np.fromfunction(lambda i, j: i ^ j, (2**kbar, 2**kbar), dtype=int)
    A_temp = A.astype(float)

    return A_temp


def MSM_starting_values(data, startingvals, kbar, A_in, distribution, dof, send_out):
    """
    find starting values for params
    :param data:
    :param startingvals:
    :param kbar:
    :return:
    """

    # print('No starting values entered: Using grid-search')
    # A grid search used to find the best set of starting values in the
    # event none are supplied by the user

    if bool(startingvals):
        dum = startingvals[1]
        b = [1.5, 1.5 + (dum - 1.5) / 3, 1.5 + 2 * (dum - 1.5) / 3, dum]
        lb = len(b)

        dum = startingvals[2]
        g = [0.01, dum - (dum - 0.01) / 2, (0.99 - dum) / 2 + dum, .99]
        lg = len(g)
        sigma = startingvals[3]

    else:
        b = [1.5, 3, 6, 20]
        lb = len(b)
        g = [.01, .5, .99]
        lg = len(g)
        sigma = np.std(data, ddof=1)  # * np.sqrt(252)

    LL_storage = df(columns=['LL', 'b', 'g', 'm', 'sigma'])
    m0_lower = 1.2
    m0_upper = 1.8

    index = 1

    # I know that adding one by one row on DF is quiet inefficient
    # I will try to use lambda, map instead of double for loop if I have a time

    for i in range(0, lb):
        for j in range(0, lg):
            a_m0 = fminbound(MSM_likelihood_new, 1.2, 1.8,
                             args=(b[i], g[j], sigma, kbar, data, A_in, distribution, dof, send_out),
                             xtol=1e-05, maxfun=500, full_output=True)  # , disp=3)
            LL_storage.loc[len(LL_storage), 'LL'] = a_m0[1]
            LL_storage.loc[len(LL_storage) - 1, 'b'] = b[i]
            LL_storage.loc[len(LL_storage) - 1, 'g'] = g[j]
            LL_storage.loc[len(LL_storage) - 1, 'm'] = a_m0[0]

    LL_storage = LL_storage.sort_values(by=['LL'], ascending=True)
    LL_storage = LL_storage.reset_index(drop=True)
    #     print(LL_storage)
    startingvals = [0., 0., 0., sigma]
    startingvals[1] = LL_storage.loc[0, 'b']
    startingvals[0] = LL_storage.loc[0, 'm']
    startingvals[2] = LL_storage.loc[0, 'g']

    return startingvals, LL_storage


def bitget(number, position):
    bi_number = bin(number)
    bi_number = bi_number[2:]
    if len(bi_number) >= position:

        fn_output = bi_number[-position]
    else:
        fn_output = 0

    return fn_output


def transition_mat(A_temp, b, gamma_k, kbar):

    A = A_temp.copy()

    gamma = np.zeros((kbar, 1))
    gamma[0] = 1 - (1 - gamma_k) ** (1 / (b ** (kbar - 1)))

    for i in range(1, kbar):
        gamma[i, 0] = 1 - (1 - gamma[0]) ** (b ** i)

    gamma = gamma / 2
    gamma = np.append(gamma, gamma, axis=1)
    gamma[:, 0] = 1 - gamma[:, 0]

    # print(gamma[:,1])
    # print(gamma[-1,1])

    kbar1 = kbar + 1
    kbar2 = 2 ** kbar

    prob = np.ones((kbar2, 1))

    for i in range(0, kbar2):
        for m in range(1, kbar + 1):
            prob[i, 0] = prob[i, 0] * gamma[kbar1 - m - 1, int(bitget(i, m))]

    for i in range(0, 2 ** (kbar - 1)):
        for j in range(i, 2 ** (kbar - 1)):
            A[kbar2 - i - 1, j] = prob[kbar2 - int(A[i, j]) - 1, 0]
            A[kbar2 - j - 1, i] = A[kbar2 - i - 1, j]
            A[j, kbar2 - i - 1] = A[kbar2 - i - 1, j]
            A[i, kbar2 - j - 1] = A[kbar2 - i - 1, j]

            A[i, j] = prob[int(A[i, j]), 0]
            A[j, i] = A[i, j]
            A[kbar2 - i - 1, kbar2 - j - 1] = A[i, j]
            A[kbar2 - j - 1, kbar2 - i - 1] = A[i, j]
    return A, gamma


def gofm(m0, kbar):

    m1 = 2 - m0
    kbar2 = 2 ** kbar
    g_m1 = list(range(0, kbar2))
    for i in range(0, kbar2):
        g = 1
        for j in range(0, kbar):  # not req -1
            if g_m1[i] & 2 ** j != 0:
                g = g * m1
            else:
                g = g * m0
        g_m1[i] = g

    g_m = np.sqrt(g_m1)
    return g_m


def MSM_likelihood_new(*args, A_in=1, distribution='Normal', dof=2, send_out='vol'):

    """
    calculate LL for 3 cases (depend on number of input)
    :param args:
    :return:
    """

    if len(args) == 7:
        # optimize -> minimizer
        print("len arg = 7")
        inp = args[0]
        kbar = args[1]
        data = args[2]
        m0 = inp['m0'].value
        b = inp['b'].value
        gamma_k = inp['gamma_k'].value
        sigma = inp['sigma'].value
        A_in = args[3]
        distribution = args[4]
        dof = args[5]
        send_out = args[6]


    elif len(args) == 10:
        # choose starting val -> fminbound
        print("len arg = 10")
        m0 = args[0]
        b = args[1]
        gamma_k = args[2]
        sigma = args[3]
        kbar = args[4]
        data = args[5]
        A_in = args[6]
        distribution = args[7]
        dof = args[8]
        send_out = args[9]

    elif len(args) == 4:
        # for outcome
        print("len arg =4")
        inp = args[0]
        kbar = args[1]
        data = args[2]
        m0 = inp['m0'].value
        b = inp['b'].value
        gamma_k = inp['gamma_k'].value
        sigma = inp['sigma'].value

    k2 = 2 ** kbar
    A_temp = A_in.copy()
    A, gamma = transition_mat(A_temp, b, gamma_k, kbar)

    g_m = gofm(m0, kbar)
    T = len(data)
    pi_mat = np.zeros((T + 1, k2))
    LLs = np.zeros((T, 1))
    pi_mat[0, :] = (1 / k2) * np.ones((1, k2))

    # Likelihood Algorithm

    # g_m is binomial measure ?
    g_m = np.array(g_m).reshape((1, len(g_m)))
    # s is matrix of all volatility (sigma) state ?
    s = np.matlib.repmat(sigma * g_m, T, 1)

    data2 = np.array(data).reshape((len(data), 1))

    if distribution == 'Normal':
        pa = (2 * np.pi) ** (-0.5)
        w_t = np.matlib.repmat(data2, 1, k2)
        w_t = np.divide(pa * np.e ** (-0.5 * np.power(np.divide(w_t, s), 2)), s)
        w_t = w_t + 10 ** -16

    elif distribution == 'ChiSq':
        s = s ** 2  # test
        w_t = np.matlib.repmat(data2, 1, k2)
        w_t = np.divide(chi2.pdf(np.divide(w_t, s) + 10 ** -25, dof), s)
        w_t = w_t + 10 ** -25

    for t in range(0, T):
        piA = np.matmul(pi_mat[t, :], A)
        C = np.multiply(w_t[t, :], piA)

        ft = sum(C)

        # stop div by zero if prob are too low
        if ft == 0:
            pi_mat[t + 1, :] = (1 / k2) * np.ones((1, k2))
            print("o(>\\\<)0 {!!!!!}")
        else:
            pi_mat[t + 1, :] = np.divide(C, ft)

        LLs[t] = np.log(np.dot(w_t[t, :], piA))

    LL = -sum(LLs)

    if np.isinf(LL):
        print('Log-likelihood is inf. Probably due to all zeros in pi_mat.')

    if (len(args) == 4) & (send_out == 'vol'):
        # Volatility prediction (single state)

        t_predict = args[3]
        vol = np.ones((t_predict, 1))
        vol = vol.astype(float)
        state_now = pi_mat[-1, :]

        for t in range(0, t_predict):
            vol[t, 0] = sum(np.matmul(state_now, np.linalg.matrix_power(A, t + 1))) * sigma

        return vol

    elif (len(args) == 4) & (send_out == 'forecast multistep'):
        # get state for multiple forecast time

        return pi_mat, A, sigma, s

    elif send_out == 'LLs':

        return LLs

    else:
        return sum(LL)  # +1/(2*gamma[-1,1])-1


def MSM_fitdata(data, kbar, LB, UB, op_methods, startingvals, A_in, distribution, dof, send_out):
    """
    Combine MSM_likelihood_new, MSM_starting_values, T_mat_Temp
    :param data: Must be a column vector of a log return
    from latest day [index0 : 1 Jan 1974] to today [index -1 : 1 Jan 2018]
    multiply it with 100
    :param kbar:
    :param LB:
    :param UB:
    :param op_methods:
    :param startingvals:
    :return:
    """

    input_param, LLS = MSM_starting_values(data, startingvals, kbar, A_in, distribution, dof, send_out)
    # print('LL = %8.4f' % LLS.loc[0, 'LL'])

    # input_param = startingvals

    # create a set of Parameters
    params = Parameters()
    params.add('m0', value=input_param[0], min=LB[0], max=UB[0])
    params.add('b', value=input_param[1], min=LB[1], max=UB[1])
    params.add('gamma_k', input_param[2], min=LB[2], max=UB[2])
    params.add('sigma', value=input_param[3], min=LB[3], max=UB[3])

    print("==========init params=========")
    for element in params:
        print(element + " = %8.4f" % (params[element].value))

    minner = Minimizer(MSM_likelihood_new, params, fcn_args=(kbar, data, A_in, distribution, dof, send_out))
    result = minner.minimize(method=op_methods)

    print("\n\n ==========fitted results==========")
    print('optimization method = ' + op_methods)
    for element in result.params:
        print(element + " = %8.4f" % (result.params[element].value))
    print("\n")
    print('LLs = %8.4f' % (result.residual))
    print('AIC = %8.4f' % (result.aic))
    print('BIC = %8.4f' % (result.bic))

    return result