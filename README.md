GIT_MSM


1.) The first part of this code will refer to Calvet and Fisher as the relying theory and the process of estimating 4 parameters of the model. I only convert the code from MATLAB into Python3.

% ------------------------------------------------------------------------- 
% Markov Switching Multifractal (MSM)
% Maximum likelihood estimation
% v1.0
% Copyright ? 2010 Multifractal-finance.com
% ------------------------------------------------------------------------- 
% 
% USAGE 
% [PARAMETERS] = MSM(DATA, K) % [PARAMETERS, LL, LLs, DIAGNOSTICS] = MSM(DATA, K, STARTING_VALUES, OPTIONS) 
% 
% INPUTS: 
% DATA - A column (or row) of mean zero data 
% KBAR - The number of frequency components 
% STARTINGVALS - [OPTIONAL] Starting values for optimization 
% [b, m0, gamma_k, sigma] 
% b - (1,inf) 
% m0 - (1,2] 
% gamma_k - (0,1) 
% sigma - [0,inf) 
% OPTIONS - {OPTIONAL} User provided options structure 
% 
% OUTPUTS: 
% PARAMETERS - A 4x1 row vector of parameters 
% [b, m0, gamma_k, sigma] 
% LL - The log-likelihood at the optimum 
% LLs - Individual daily log-likelihoods at optimum % diagnostics - Structure of optimization output information. 
% Useful for checking convergence problems 
% 
% ASSOCIATED FILES: 
% MSM_likelihood.m, MSM_parameter_check.m, MSM_starting_values.m 
% 
% REFERENCES: 
% [1] Calvet, L., Adlai Fisher (2004). "How to Forecast long-run % volatility: regime-switching and the estimation of multifractal processes". Journal of Financial Econometrics 2: 49?83. 
% [2] Calvet, L., Adlai Fisher (2008). "Multifractal Volatility: Theory, Forecasting and Pricing". Elsevier - Academic Press.. 
% -------------------------------------------------------------------------

2.) After that, I will try to forecast by using those parameters and do a model valuation both in-sample and out-of sample test. Which is the part of my Independent Study (IS) plan for Master degree In Finance at Thammasat University, Thailand.

