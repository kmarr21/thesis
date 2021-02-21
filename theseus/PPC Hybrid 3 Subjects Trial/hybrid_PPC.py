## IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from numba import njit
from pandas import DataFrame, read_csv, concat
import os, h5py, pystan
import _pickle as pickle

# DEFINE FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@njit
def inv_logit(x):
    return 1. / (1. + np.exp(-x))

def load_model(filepath):
    """Load or precomplile a StanModel object.
    Parameters
    ----------
    filepath : str
        Path to the Stan model.
    Returns
    -------
    StanModel : pystan.StanModel
        Model described in Stan’s modeling language compiled from C++ code.
    Notes
    -----
    If an extensionless filepath is supplied, looks for *.stan or *.txt files for StanCode 
    and *.pkl and *.pickle for StanModels. Otherwise requires a file with one of those four extensions.
    """

    for ext in ['.pkl','.pickle','.stan','.txt']:

        if filepath.endswith(ext):
            break
        elif os.path.isfile(filepath + ext):
            filepath += ext
            break

    if filepath.lower().endswith(('.pkl','.pickle')):

        ## Load pickle object.
        StanModel = pickle.load(open(filepath, 'rb'))

    elif filepath.lower().endswith(('.stan','.txt')):

        ## Precompile StanModel.
        StanModel = pystan.StanModel(file=filepath)

        ## Dump to pickle object.
        f = '.'.join(filepath.split('.')[:-1]) + '.pkl'
        with open(f, 'wb') as f: pickle.dump(StanModel, f)

    else:

        raise IOError('%s not correct filetype.' %filepath)

    return StanModel

## MAIN CODE SEGMENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## LOAD DATA
subjects = read_csv('HY3.csv')
data = read_csv('cleaned_Gillandata.csv')

total_data = []

for subject in subjects:

    ## EXTRACT & PREP DATA
    dd = data.loc[data['subject'] == subject]
    dd = dd.dropna()
    dd = dd[dd.outcome != -1.0]

    R = dd['outcome'].astype(int)
    Y1 = dd['stage_1_choice'].astype(int)
    Y2 = dd['stage_2_choice'].astype(int)
    S2 = dd["stage_2_state"].astype(int)
    t = dd['transition'].astype(int)
    T = len(R)

    ## FIT STAN MODEL
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Define parameters.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## I/O parameters.
    stan_model = 'hybrid'
    PATH_NAME = '/Users/kierstenmarr/Desktop/hybrid'

    ## Sampling parameters.
    warmup   = 250
    samples = 750
    chains = 3
    thin = 1
    n_jobs = 3

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Fit Stan Model.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ## Assemble data.
    dd = dict(T=T, Y1=Y1, Y2=Y2, S2=S2, R=R.astype(float))

    ## Load StanModel
    StanModel = load_model(PATH_NAME)

    ## Fit Stan model.
    StanFit = StanModel.sampling(data=dd, iter=samples, warmup=warmup, chains=chains, thin=thin, n_jobs=n_jobs, seed=44404)

    ## POSTERIOR PREDICTIVE CHECK
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Define parameters.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Extract parameters from Stan.
    StanDict = StanFit.extract()

    ## Define agent parameters.
    w = StanDict['w']
    beta_1 = StanDict['beta_1']
    beta_2 = StanDict['beta_2']
    eta_1  = StanDict['eta_1']
    eta_2  = StanDict['eta_2']

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Posterior predictive check.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    np.random.seed(47404)

    ## Initialize Q-values.
    n_samp = w.size
    MB = np.ones((n_samp,2)) * 0.5
    MF = np.ones((n_samp,2)) * 0.5
    HQ = np.ones((n_samp,2)) * 0.5
    Q = np.ones((n_samp,2,2)) * 0.5

    ## Preallocate space.
    Y1_hat  = np.zeros((n_samp,T))
    Y1_pred = np.zeros((n_samp,T))
    Y2_hat  = np.zeros((n_samp,T))
    Y2_pred = np.zeros((n_samp,T))

    for i in tqdm(range(T)):
        
        ## Stage 1 choice.
        theta = inv_logit( beta_1 * (HQ[:,1] - HQ[:,0]) )
        Y1_hat[:,i]  = np.random.binomial(1, theta)       # Simulate choice
        Y1_pred[:,i] = np.where(Y1[i], theta, 1-theta)    # Compare choice
        
        ## Stage 2 choice.
        theta = inv_logit( beta_2 * (Q[:,S2[i],1] - Q[:,S2[i],0]) )
        Y2_hat[:,i]  = np.random.binomial(1, theta)       # Simulate choice
        Y2_pred[:,i] = np.where(Y2[i], theta, 1-theta)    # Compare choice
        
        ## Update Q-values.
        Q[:,S2[i],Y2[i]] += eta_2 * (R[i] - Q[:,S2[i],Y2[i]])
        
        ## Update MB values.
        MB[:,0] = 0.7 * np.max(Q[:,0], axis=-1) + 0.3 * np.max(Q[:,1], axis=-1)
        MB[:,1] = 0.3 * np.max(Q[:,0], axis=-1) + 0.7 * np.max(Q[:,1], axis=-1)
        
        ## Update MF values.
        MF[:,Y1[i]] = eta_1 * (R[i] - MF[:,Y1[i]])
        
        ## Update hybrid values.
        HQ = w[:,np.newaxis] * MB + (1-w[:,np.newaxis]) * MF

    ## Compute stay.
    stay_hat = (Y1_hat == np.roll(Y1, 1)).astype(int)

    ## debugging
    print(stay_hat)
        
    ## Average across samples.
    Y1_hat  = Y1_hat.mean(axis=0)
    Y1_pred = Y1_pred.mean(axis=0)
    Y2_hat  = Y2_hat.mean(axis=0)
    Y2_pred = Y2_pred.mean(axis=0)
    stay_hat = stay_hat.mean(axis=0)
    subs = np.array([subject] * len(stay_hat))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Plotting.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Convert to DataFrame.
    columns = ['subject','transition','S2','R','Y1','Y1_hat','Y1_pred','Y2','Y2_hat','Y2_pred','stay_hat']
    dat = DataFrame(np.column_stack([subs,t,S2,R,Y1,Y1_hat,Y1_pred,Y2,Y2_hat,Y2_pred,stay_hat]), columns=columns)

    ## Compute analysis variables.
    dat['prev_t'] = np.roll(dat.transition, 1)
    dat['prev_r'] = np.roll(dat.R, 1)
    dat['stay'] = (dat.Y1 == np.roll(dat.Y1, 1)).astype(int)

    ## CONCATENATE
    total_data.append(dat)

    ## debugging
    print(dat)

## debugging 
print(total_data)

## SAVE
total_data = concat(total_data)

## save out
total_data.to_csv("HY3_PPC_test.csv")
