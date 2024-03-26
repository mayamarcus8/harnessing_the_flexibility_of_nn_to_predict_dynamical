import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def q_pred(df,parameters):

    # initialize predictions array
    num_of_trials = len(df)
    choice_epred = np.zeros(num_of_trials)

    # upload data of the subject/agent
    action_list = df['action'].astype(int)
    reward_list = df['reward'].astype(np.float32)
 
    # set up paramters of the agent     
    alpha = parameters[0] 
    beta = parameters[1]

    # initialize q-values and preservation
    q = np.zeros(2)

    for t in range(num_of_trials):
            
        if t%100 == 0:
            q = np.zeros(2)
        
        # set up data
        choice = action_list[t]
        reward = reward_list[t]

        # decision
        p = np.exp( beta*q ) / np.sum( np.exp( beta * q ) )  
        choice_epred[t] = p[1]

        # value update
        prediction_error = reward - q[choice] 
        q[choice] = q[choice] + alpha*prediction_error 

    # auc computation
    auc = roc_auc_score(action_list, choice_epred)
           
    return auc, choice_epred 
    
