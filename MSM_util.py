import numpy as np
import pandas as pd
from pandas import DataFrame as df
from numpy import matlib
from scipy.optimize import fminbound, minimize, fsolve
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date
import winsound
from pylab import rcParams


def T_mat_temp(kbar):
    """
    Build template for kbar
    :param kbar:
    :return:
    """
    A = np.fromfunction(lambda i, j: i ^ j, (2**kbar, 2**kbar), dtype=int)
    A_temp = A.astype(float)
    return A_temp


def MSM_starting_values(data, startingvals, kbar):
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
    b = [1.5, 3, 6, 20]
    lb = len(b)
    g = [.1, .5, .9]
    lg = len(g)
    sigma = np.std(data, ddof=1) #* np.sqrt(252)

    LL_storage = df(columns=['LL', 'b', 'g', 'm'])
    m0_lower = 1.2
    m0_upper = 1.8

    index = 1

    # I know that adding one by one row on DF is quiet inefficient
    # I will try to use lambda, map instead of double for loop if I have a time

    for i in range(0, lb):
        for j in range(0, lg):
            a_m0 = fminbound(MSM_likelihood_new, 1.2, 1.8,
                             args=(b[i], g[j], sigma, kbar, data),
                             xtol=1e-05, maxfun=500, full_output=True)  # , disp=3)
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


def MSM_likelihood_new(*args):
    """
    calculate LL for 3 cases (depend on number of input)
    :param args:
    :return:
    """

    if len(args) ==3 :
        # choose starting val
        inp = args[0]
        kbar = args[1]
        data = args[2]
        m0 = inp['m0'].value
        b = inp['b'].value
        gamma_k = inp['gamma_k'].value
        sigma = inp['sigma'].value

    elif len(args) == 6 :
        #
        m0 = args[0]
        b = args[1]
        gamma_k = args[2]
        sigma = args[3]
        kbar = args[4]
        data = args[5]

    elif len(args) == 4 :
        # prediction
        inp = args[0]
        kbar = args[1]
        data = args[2]
        m0 = inp['m0'].value
        b = inp['b'].value
        gamma_k = inp['gamma_k'].value
        sigma = inp['sigma'].value

    A_temp = T_mat_temp(kbar)
    k2 = 2 ** kbar

    def transition_mat(A_temp, b, gamma_k, kbar):
        A = A_temp

        gamma = np.zeros((kbar, 1))
        gamma[0] = 1 - (1 - gamma_k) ** (1 / (b ** (kbar - 1)))

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

    # g_m is binomial measure ?
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

    if len(args) == 4:

        # prediction
        t_predict = args[3]
        vol = np.ones((t_predict,1))
        vol = vol.astype(float)
        state_now = pi_mat[-1 , :]
        for t in range(0,t_predict):
            vol[t,0] = sum(np.matmul(state_now, np.linalg.matrix_power(A, t+1)))*sigma

        return vol

    else:
        return sum(LL)


def MSM_fitdata(data, kbar, LB ,UB, op_methods, startingvals):
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

    if not startingvals:
        input_param, LLS = MSM_starting_values(data, startingvals, kbar)
        # print('LL = %8.4f' % LLS.loc[0, 'LL'])
    else:
        input_param = startingvals

    # create a set of Parameters
    params = Parameters()
    params.add('m0', value=input_param[0], min=LB[0], max=UB[0])
    params.add('b', value=input_param[1], min=LB[1], max=UB[1])
    params.add('gamma_k', input_param[2], min=LB[2], max=UB[2])
    params.add('sigma', value=input_param[3], min=LB[3], max=UB[3])

    print("==========init params=========")
    for element in params:
        print(element + " = %8.4f" % (params[element].value))



    minner = Minimizer(MSM_likelihood_new, params, fcn_args=(kbar, data))
    result = minner.minimize(method=op_methods)

    # print("\n\n ==========fitted results==========")
    # print('optimization method = ' + op_methods)
    # for element in result.params:
    #     print(element + " = %8.4f" % (result.params[element].value))
    # print('LLs = %8.4f' % (result.residual))
    # print('AIC = %8.4f' % (result.aic))
    # print('BIC = %8.4f' % (result.bic))

    return result


def msm_fitseries(data, kbar, LB, UB, op_methods, startingvals, m, RV):
    """
     series of MSM_fitdata for calculate step-by-step
    :param data: 
    :param kbar: 
    :param LB: 
    :param UB: 
    :param op_methods: OPTIMIZATION METHOD
    :param startingvals: 
    :param m: numbers of days we try to estimate
    :param RV: days to calculate RV
    :return: 
    """
    output = df()
    output["RV"] = []
    output["m0"] = []
    output["b"] = []
    output["gamma_k"] = []
    output["sigma"] = []

    for i in range(0, m):
        print("round" + str(i))
        data2 = data[i:len(data) - m + i]
        result = MSM_fitdata(data2, kbar, LB, UB, op_methods, startingvals)
        re2 = MSM_likelihood_new(result.params, kbar, data2, RV)
        output.loc[i, "RV"] = sum(re2) ** 2 * 100
        for element in result.params:
            output.loc[i, element] = result.params[element].value

    return output


def msm_averageparams(output, kbar, m, data, RV, LB , UB):
    """
    AVERAGE INPUT PARAMS for test robutness of model
    :param output:
    :param kbar:
    :param m:
    :param data:
    :param RV:
    :param LB:
    :param UB:
    :return:
    """
    mean_o = output.mean()
    params = Parameters()
    params.add('m0', value=mean_o[1], min=LB[0], max=UB[0])
    params.add('b', value=mean_o[2], min=LB[1], max=UB[1])
    params.add('gamma_k', mean_o[3], min=LB[2], max=UB[2])
    params.add('sigma', value=mean_o[4], min=LB[3], max=UB[3])

    output_m = df()
    output_m["RV"] = []
    output_m["m0"] = []
    output_m["b"] = []
    output_m["gamma_k"] = []
    output_m["sigma"] = []

    for i in range(0, m):
        data2 = data[i:len(data) - m + i]
        re2 = MSM_likelihood_new(params, kbar, data2, RV)
        output_m.loc[i, "RV"] = sum(re2) ** 2 * 100
        for element in params:
            output_m.loc[i, element] = params[element].value

    return output_m


def msm_plot(GVZ, output, m):
    """
    plot quickly
    :param GVZ:
    :param output:
    :param m:
    :return:
    """

    # GVZ = xls_data['GVZ']
    GVZ_lastm = GVZ[::-1]
    GVZ_lastm = GVZ_lastm.iloc[-m:]

    plt.figure()
    actual = plt.plot(range(0, m), GVZ_lastm, label='GVZ', color='red')
    model = plt.plot(range(0, m), (output['RV']), label='MSM')

    plt.title('Predict VOL actual(GVZ) vs forecast(model)')
    plt.legend()

    plt.figure()
    plt.plot(range(0, m), output['m0'], label='m0')
    plt.title('m0')
    plt.legend()

    plt.figure()
    plt.plot(range(0, m), output['b'], label='b')
    plt.title('b')
    plt.legend()

    plt.figure()
    plt.plot(range(0, m), output['gamma_k'], label='gamma_k')
    plt.title('gamma_k')
    plt.legend()

    plt.figure()
    plt.plot(range(0, m), output['gamma_k'], label='gamma_k')
    plt.title('gamma_k')
    plt.legend()

    plt.figure()
    plt.plot(range(0, m), output['sigma'], label='sigma')
    plt.title('sigma')
    plt.legend()

    plt.show()


def linreg(x,y):
    """
    Quick Regression
    :param x:
    :param y:
    :return:
    """
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x, y)

    # The coefficients
    print('Slope : ', sum(sum(regr.coef_)))
    print('Intercept : ', sum(regr.intercept_))

    # # The mean squared error
    # print("Mean squared error: %.2f"
    #       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    return sum(sum(regr.coef_)), sum(regr.intercept_)


def msm_vary_k_cal(data, vary, kbar_start, kbar_max, LB, UB, op_methods, startingvals, m, RVn):
    """
    calculation by vary kbar to see the effect of kbar
    :param data:
    :param vary: "k"
    :param kbar_start:
    :param kbar_max:
    :param LB:
    :param UB:
    :param op_methods:
    :param startingvals:
    :param m:
    :param RVn:
    :return:
    """
    RV = df()
    m0 = df()
    b = df()
    gamma_k = df()
    sigma = df()

    for kbar in range(kbar_start, kbar_max):
        # fit series
        print("=====kbar = " + str(kbar))
        output = msm_fitseries(data, kbar, LB, UB, op_methods, startingvals, m, RVn)
        RV[vary + str(kbar)] = output["RV"]
        m0[vary + str(kbar)] = output["m0"]
        b[vary + str(kbar)] = output["b"]
        gamma_k[vary + str(kbar)] = output["gamma_k"]
        sigma[vary + str(kbar)] = output["sigma"]

    text = 'MSM_vary_kbar'
    namew = text + str(kbar_max) + "_RV" + str(RVn) + '_m' + str(m) + ".xlsx"
    writer = pd.ExcelWriter("".join((date.today().strftime('%y%m%d'), namew)))
    RV.to_excel(writer, 'RV')
    m0.to_excel(writer, 'm0')
    b.to_excel(writer, 'b')
    gamma_k.to_excel(writer, 'gamma_k')
    sigma.to_excel(writer, 'sigma')
    print("".join((date.today().strftime('%y%m%d'), namew)))
    writer.save()

    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

    return RV, m0, b, gamma_k, sigma


def msm_vary_k_plot(GVZ, m=20, *args):

    def label_plot(m, x, LA):
        for LAs in LA:
            plt.plot(range(0, m), x[LAs], label=LAs)

    if len(args) == 1:
        xls = pd.ExcelFile(args[0])
        RV = pd.read_excel(xls, 'RV')
        m0 = pd.read_excel(xls, 'm0')
        b = pd.read_excel(xls, 'b')
        gamma_k = pd.read_excel(xls, 'gamma_k')
        sigma = pd.read_excel(xls, 'sigma')

    elif len(args) == 5:
        RV = args[0]
        m0 = args[1]
        b = args[2]
        gamma_k = args[3]
        sigma = args[4]

    GVZ_lastm = GVZ.iloc[-m:]

    rcParams['figure.figsize'] = 5, 25

    LA = m0.columns.tolist()

    plt.subplot(511)
    actual = plt.plot(range(0, m), GVZ_lastm, label='GVZ', color='red')
    label_plot(m, RV, LA)
    plt.title('Predict VOL actual(GVZ) vs forecast(model)')
    plt.legend()

    plt.subplot(512)
    label_plot(m, m0, LA)
    plt.title('m0')
    plt.legend()

    plt.subplot(513)
    label_plot(m, b, LA)
    plt.title('b')
    plt.legend()

    plt.subplot(514)
    label_plot(m, gamma_k, LA)
    plt.title('gamma_k')
    plt.legend()

    plt.subplot(515)
    label_plot(m, sigma, LA)
    plt.title('sigma')
    plt.legend()

    plt.show()