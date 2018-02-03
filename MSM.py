import numpy as np
import pandas as pd
from numpy import matlib

def MSM_likelihood(input_param, kbar, data, A_template, estim_flag):
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

    A = transition_mat(A_template, input_param, kbar)
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
    A_template = A.astype(float)
    return A_template

def MSM_likelihood_new(b, m0, gamma_k, sigma, kbar, data, A_template):

    input_param = np.zeros((4, 1))
    input_param[0] = b
    input_param[1] = m0
    input_param[2] = gamma_k
    input_param[3] = sigma

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

    A = transition_mat(A_template, input_param, kbar)
    g_m = gofm(input_param, kbar)
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

    LL = sum(LLs)

    if np.isinf(LL):
        print('Log-likelihood is inf. Probably due to all zeros in pi_mat.')

    return LL