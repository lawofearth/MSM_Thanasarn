import numpy as np
import pandas as pd
from pandas import DataFrame as df
from numpy import matlib
from scipy.optimize import fminbound, minimize, fsolve
from lmfit import minimize, model


def MSM_likelihood(input_param, kbar, data, A_temp, estim_flag):
    if len(input_param) == 1:
        input_param = [estim_flag[0], input_param, estim_flag[1], estim_flag[2]]

    sigma = input_param[3] / np.sqrt(252)
    k2 = 2 ** kbar

    def transition_mat(A, input_param, kbar):
        b = input_param[0]
        gamma_kbar = input_param[2]

        gamma = np.zeros((kbar, 1))
        gamma[0] = 1 - (1 - gamma_kbar) ** (1 / (b ** (kbar - 1)))  # TOWARN : Equation does not conform bbut math is ok

        def bitget(number, position):

            bi_number = bin(number)
            bi_number = bi_number[2:]
            if len(bi_number) >= position:

                fn_output = bi_number[-position]
            else:
                fn_output = 0

            return fn_output

        for i in range(1, kbar):
            gamma[i, 0] = 1 - (1 - gamma[0]) ** (b ** i)

        gamma = gamma / 2
        gamma = np.append(gamma, gamma, axis=1)
        gamma[:, 0] = 1 - gamma[:, 0]

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
        return A

    def gofm(input_param, kbar):

        m0 = input_param[1]
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

    A = transition_mat(A_temp, input_param, kbar)
    g_m = gofm(input_param, kbar)
    T = len(data)
    pi_mat = np.zeros((T + 1, k2))
    LLs = np.zeros((T,1))
    pi_mat[0, :] = (1 / k2) * np.ones((1, k2))

    # Likelihood Algorithm

    pa = (2 * np.pi) ** (-0.5)
    s = np.matlib.repmat(sigma * g_m, T, 1)
    data2 = np.array(data).reshape((len(data), 1))
    w_t = np.matlib.repmat(data2, 1, k2)

    w_t = np.divide(pa * np.e ** (-0.5 * np.power(np.divide(w_t, s), 2)), s)
    w_t = w_t + 10 ** -16

    for t in range(0, T):
        piA = np.matmul(pi_mat[t, :], A)
        C = np.multiply(w_t[t, :], piA)
        ft = sum(C)

        # stop div by zero if prob are too low
        if ft == 0:
            pi_mat[t + 1, 0] = 1
        else:
            pi_mat[t + 1, :] = np.divide(C, ft)

        LLs[t] = np.log(np.dot(w_t[t, :], piA))

    LL = sum(LLs)

    if np.isinf(LL):
        print('Log-likelihood is inf. Probably due to all zeros in pi_mat.')

    return LL, LLs


def T_mat_temp(kbar):
    A = np.fromfunction(lambda i, j: i ^ j, (2**kbar, 2**kbar), dtype=int)
    A_temp = A.astype(float)
    return A_temp


def MSM_likelihood_new(m0, b, gamma_k, sigma, kbar, data, A_temp):

    sigma = sigma / np.sqrt(252)
    k2 = 2 ** kbar

    def transition_mat(A_temp, b, gamma_k, kbar):
        A = A_temp

        gamma = np.zeros((kbar, 1))
        gamma[0] = 1 - (1 - gamma_k) ** (1 / (b ** (kbar - 1)))  # TOWARN : Equation does not conform bbut math is ok

        def bitget(number, position):

            bi_number = bin(number)
            bi_number = bi_number[2:]
            if len(bi_number) >= position:

                fn_output = bi_number[-position]
            else:
                fn_output = 0

            return fn_output

        for i in range(1, kbar):
            gamma[i, 0] = 1 - (1 - gamma[0]) ** (b ** i)

        gamma = gamma / 2
        gamma = np.append(gamma, gamma, axis=1)
        gamma[:, 0] = 1 - gamma[:, 0]

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
        return A

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

    A = transition_mat(A_temp, b, gamma_k, kbar)
    g_m = gofm(m0, kbar)
    T = len(data)
    pi_mat = np.zeros((T + 1, k2))
    LLs = np.zeros((T,1))
    pi_mat[0, :] = (1 / k2) * np.ones((1, k2))

    # Likelihood Algorithm

    pa = (2 * np.pi) ** (-0.5)
    g_m = np.array(g_m).reshape((1, len(g_m)))
    s = np.matlib.repmat(sigma * g_m, T, 1)


    data2 = np.array(data).reshape((len(data), 1))
    w_t = np.matlib.repmat(data2, 1, k2)

    w_t = np.divide(pa * np.e ** (-0.5 * np.power(np.divide(w_t, s), 2)), s)
    w_t = w_t + 10 ** -16

    for t in range(0, T):
        piA = np.matmul(pi_mat[t, :], A)
        C = np.multiply(w_t[t, :], piA)
        ft = sum(C)

        # stop div by zero if prob are too low
        if ft == 0:
            pi_mat[t + 1, 0] = 1
        else:
            pi_mat[t + 1, :] = np.divide(C, ft)

        LLs[t] = np.log(np.dot(w_t[t, :], piA))

    LL = -sum(LLs)

    if np.isinf(LL):
        print('Log-likelihood is inf. Probably due to all zeros in pi_mat.')

    return LL

def MSM_likelihood_new2(m0, b, gamma_k, sigma, kbar, data):

    A_temp = T_mat_temp(kbar)
    sigma = sigma / np.sqrt(252)
    k2 = 2 ** kbar

    def transition_mat(A_temp, b, gamma_k, kbar):
        A = A_temp

        gamma = np.zeros((kbar, 1))
        gamma[0] = 1 - (1 - gamma_k) ** (1 / (b ** (kbar - 1)))  # TOWARN : Equation does not conform bbut math is ok

        def bitget(number, position):

            bi_number = bin(number)
            bi_number = bi_number[2:]
            if len(bi_number) >= position:

                fn_output = bi_number[-position]
            else:
                fn_output = 0

            return fn_output

        for i in range(1, kbar):
            gamma[i, 0] = 1 - (1 - gamma[0]) ** (b ** i)

        gamma = gamma / 2
        gamma = np.append(gamma, gamma, axis=1)
        gamma[:, 0] = 1 - gamma[:, 0]

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
        return A

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

    A = transition_mat(A_temp, b, gamma_k, kbar)
    g_m = gofm(m0, kbar)
    T = len(data)
    pi_mat = np.zeros((T + 1, k2))
    LLs = np.zeros((T,1))
    pi_mat[0, :] = (1 / k2) * np.ones((1, k2))

    # Likelihood Algorithm

    pa = (2 * np.pi) ** (-0.5)
    g_m = np.array(g_m).reshape((1, len(g_m)))
    s = np.matlib.repmat(sigma * g_m, T, 1)


    data2 = np.array(data).reshape((len(data), 1))
    w_t = np.matlib.repmat(data2, 1, k2)

    w_t = np.divide(pa * np.e ** (-0.5 * np.power(np.divide(w_t, s), 2)), s)
    w_t = w_t + 10 ** -16

    for t in range(0, T):
        piA = np.matmul(pi_mat[t, :], A)
        C = np.multiply(w_t[t, :], piA)
        ft = sum(C)

        # stop div by zero if prob are too low
        if ft == 0:
            pi_mat[t + 1, 0] = 1
        else:
            pi_mat[t + 1, :] = np.divide(C, ft)

        LLs[t] = np.log(np.dot(w_t[t, :], piA))

    LL = -sum(LLs)

    if np.isinf(LL):
        print('Log-likelihood is inf. Probably due to all zeros in pi_mat.')

    return sum(LL)
def MSM_starting_values2(data, startingvals, kbar):
    # options=optimset('fminbnd');
    xtol = 1e-5
    #     LLs = []
    #     output_parameters=[]

    # implicit  booleanness of the empty list is quite pythonic.
    if not startingvals:
        print('No starting values entered: Using grid-search')
        # A grid search used to find the best set of starting values in the
        # event none are supplied by the user
        b = [1.5, 3, 6, 20]
        lb = len(b)
        g = [.1, .5, .9]
        lg = len(g)
        sigma = np.std(data, ddof=1) * np.sqrt(252)

        LL_storage = df(columns=['LL', 'b', 'g', 'm'])
        # output_parameters=np.zeros((lb*lg,3))
        # LLs=zeros((lb*lg,1))
        m0_lower = 1.2
        m0_upper = 1.8

        index = 1

        # I know that adding one by one row on DF is quiet inefficient
        # I will try to use lambda, map instead of double for loop if I have a time

        for i in range(0, lb):
            for j in range(0, lg):
                a_m0 = fminbound(MSM_likelihood_new2, 1.2, 1.8,
                                 args=(b[i], g[j], sigma, kbar, data),
                                 xtol=1e-05, maxfun=500, full_output = True, disp=3)
                LL_storage.loc[len(LL_storage), 'LL'] = a_m0[1]
                LL_storage.loc[len(LL_storage) - 1, 'b'] = b[i]
                LL_storage.loc[len(LL_storage) - 1, 'g'] = g[j]
                LL_storage.loc[len(LL_storage) - 1, 'm'] = a_m0[0]

        LL_storage = LL_storage.sort_values(by=['LL'], ascending=True)
        startingvals = [0., 0., 0., sigma]
        startingvals[1] = LL_storage.loc[0, 'b']
        startingvals[0] = LL_storage.loc[0, 'm']
        startingvals[2] = LL_storage.loc[0, 'g']

    return startingvals, LL_storage
def MSM_likelihood_new2V2(inp, kbar, data):
    m0 = inp['m0'].value
    b = inp['b'].value
    gamma_k = inp['gamma_k'].value
    sigma = inp['sigma'].value

    A_temp = T_mat_temp(kbar)
    sigma = sigma / np.sqrt(252)
    k2 = 2 ** kbar

    def transition_mat(A_temp, b, gamma_k, kbar):
        A = A_temp

        gamma = np.zeros((kbar, 1))
        gamma[0] = 1 - (1 - gamma_k) ** (1 / (b ** (kbar - 1)))  # TOWARN : Equation does not conform bbut math is ok

        def bitget(number, position):

            bi_number = bin(number)
            bi_number = bi_number[2:]
            if len(bi_number) >= position:

                fn_output = bi_number[-position]
            else:
                fn_output = 0

            return fn_output

        for i in range(1, kbar):
            gamma[i, 0] = 1 - (1 - gamma[0]) ** (b ** i)

        gamma = gamma / 2
        gamma = np.append(gamma, gamma, axis=1)
        gamma[:, 0] = 1 - gamma[:, 0]

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
        return A

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

    A = transition_mat(A_temp, b, gamma_k, kbar)
    g_m = gofm(m0, kbar)
    T = len(data)
    pi_mat = np.zeros((T + 1, k2))
    LLs = np.zeros((T,1))
    pi_mat[0, :] = (1 / k2) * np.ones((1, k2))

    # Likelihood Algorithm

    pa = (2 * np.pi) ** (-0.5)
    g_m = np.array(g_m).reshape((1, len(g_m)))
    s = np.matlib.repmat(sigma * g_m, T, 1)


    data2 = np.array(data).reshape((len(data), 1))
    w_t = np.matlib.repmat(data2, 1, k2)

    w_t = np.divide(pa * np.e ** (-0.5 * np.power(np.divide(w_t, s), 2)), s)
    w_t = w_t + 10 ** -16

    for t in range(0, T):
        piA = np.matmul(pi_mat[t, :], A)
        C = np.multiply(w_t[t, :], piA)
        ft = sum(C)

        # stop div by zero if prob are too low
        if ft == 0:
            pi_mat[t + 1, 0] = 1
        else:
            pi_mat[t + 1, :] = np.divide(C, ft)

        LLs[t] = np.log(np.dot(w_t[t, :], piA))

    LL = -sum(LLs)

    if np.isinf(LL):
        print('Log-likelihood is inf. Probably due to all zeros in pi_mat.')

    return LL
def MSM_likelihood_new2V1(inp, kbar, data):
    m0 = inp[0]
    b = inp[1]
    gamma_k = inp[2]
    sigma = inp[3]

    A_temp = T_mat_temp(kbar)
    sigma = sigma / np.sqrt(252)
    k2 = 2 ** kbar

    def transition_mat(A_temp, b, gamma_k, kbar):
        A = A_temp

        gamma = np.zeros((kbar, 1))
        gamma[0] = 1 - (1 - gamma_k) ** (1 / (b ** (kbar - 1)))  # TOWARN : Equation does not conform bbut math is ok

        def bitget(number, position):

            bi_number = bin(number)
            bi_number = bi_number[2:]
            if len(bi_number) >= position:

                fn_output = bi_number[-position]
            else:
                fn_output = 0

            return fn_output

        for i in range(1, kbar):
            gamma[i, 0] = 1 - (1 - gamma[0]) ** (b ** i)

        gamma = gamma / 2
        gamma = np.append(gamma, gamma, axis=1)
        gamma[:, 0] = 1 - gamma[:, 0]

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
        return A

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

    A = transition_mat(A_temp, b, gamma_k, kbar)
    g_m = gofm(m0, kbar)
    T = len(data)
    pi_mat = np.zeros((T + 1, k2))
    LLs = np.zeros((T,1))
    pi_mat[0, :] = (1 / k2) * np.ones((1, k2))

    # Likelihood Algorithm

    pa = (2 * np.pi) ** (-0.5)
    g_m = np.array(g_m).reshape((1, len(g_m)))
    s = np.matlib.repmat(sigma * g_m, T, 1)


    data2 = np.array(data).reshape((len(data), 1))
    w_t = np.matlib.repmat(data2, 1, k2)

    w_t = np.divide(pa * np.e ** (-0.5 * np.power(np.divide(w_t, s), 2)), s)
    w_t = w_t + 10 ** -16

    for t in range(0, T):
        piA = np.matmul(pi_mat[t, :], A)
        C = np.multiply(w_t[t, :], piA)
        ft = sum(C)

        # stop div by zero if prob are too low
        if ft == 0:
            pi_mat[t + 1, 0] = 1
        else:
            pi_mat[t + 1, :] = np.divide(C, ft)

        LLs[t] = np.log(np.dot(w_t[t, :], piA))

    LL = -sum(LLs)

    if np.isinf(LL):
        print('Log-likelihood is inf. Probably due to all zeros in pi_mat.')

    return LL

def MSM_starting_values(data, startingvals, kbar, A_temp):
    # options=optimset('fminbnd');
    xtol = 1e-5
    #     LLs = []
    #     output_parameters=[]

    # implicit  booleanness of the empty list is quite pythonic.
    if not startingvals:
        print('No starting values entered: Using grid-search')
        # A grid search used to find the best set of starting values in the
        # event none are supplied by the user
        b = [1.5, 3, 6, 20]
        lb = len(b)
        g = [.1, .5, .9]
        lg = len(g)
        sigma = np.std(data, ddof=1) * np.sqrt(252)

        LL_storage = df(columns=['LL', 'b', 'g', 'm'])
        # output_parameters=np.zeros((lb*lg,3))
        # LLs=zeros((lb*lg,1))
        m0_lower = 1.2
        m0_upper = 1.8

        index = 1

        # I know that adding one by one row on DF is quiet inefficient
        # I will try to use lambda, map instead of double for loop if I have a time

        A = A_temp
        for i in range(0, lb):
            for j in range(0, lg):
                a_m0 = fminbound(MSM_likelihood_new, 1.2, 1.8,
                                 args=(b[i], g[j], sigma, kbar, data, A),
                                 xtol=1e-05, maxfun=500, full_output = True, disp=3)
                LL_storage.loc[len(LL_storage), 'LL'] = sum(a_m0[1])
                LL_storage.loc[len(LL_storage) - 1, 'b'] = b[i]
                LL_storage.loc[len(LL_storage) - 1, 'g'] = g[j]
                LL_storage.loc[len(LL_storage) - 1, 'm'] = a_m0[0]

        LL_storage = LL_storage.sort_values(by=['LL'], ascending=True)
        startingvals = [0., 0., 0., sigma]
        startingvals[0] = LL_storage.loc[0, 'b']
        startingvals[1] = LL_storage.loc[0, 'm']
        startingvals[2] = LL_storage.loc[0, 'g']

    return startingvals, LL_storage


def MSM_starting_values_util(data, startingvals, kbar, A_template):
    # options=optimset('fminbnd');
    xtol = 1e-5
    #     LLs = []
    #     output_parameters=[]

    # implicit  booleanness of the empty list is quite pythonic.
    if not startingvals:
        print('No starting values entered: Using grid-search')
        # A grid search used to find the best set of starting values in the
        # event none are supplied by the user
        b = [1.5, 3, 6, 20]
        lb = len(b)
        g = [.1, .5, .9]
        lg = len(g)
        sigma = np.std(data) * np.sqrt(252)

        LL_storage = df(columns=['LL', 'b', 'g', 'm'])
        # output_parameters=np.zeros((lb*lg,3))
        # LLs=zeros((lb*lg,1))
        m0_lower = 1.2
        m0_upper = 1.8

        index = 1

        # I know that adding one by one row on DF is quiet inefficient
        # I will try to use lambda, map instead of double for loop if I have a time

        mymodel = model(MSM_likelihood_new)
        for i in range(0, lb):
            for j in range(0, lg):
                params = mymodel.make_params(a=100, b=-1, c=0, d=0)
                a_m0 = fminbound(MSM_util.MSM_likelihood_new, 1.2, 1.8,
                                 args=(b[i], g[j], sigma, kbar, data, A_template),
                                 xtol=1e-03, maxfun=500, full_output = True, disp=3)
                LL_storage.loc[len(LL_storage), 'LL'] = sum(a_m0[1])
                LL_storage.loc[len(LL_storage) - 1, 'b'] = b[i]
                LL_storage.loc[len(LL_storage) - 1, 'g'] = g[j]
                LL_storage.loc[len(LL_storage) - 1, 'm'] = a_m0[0]

        LL_storage = LL_storage.sort_values(by=['LL'], ascending=False)
        startingvals = [0., 0., 0., sigma]
        startingvals[0] = LL_storage.loc[0, 'b']
        startingvals[1] = LL_storage.loc[0, 'm']
        startingvals[2] = LL_storage.loc[0, 'g']

    return startingvals, LL_storage