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
    qvals = np.zeros((num_of_trials,2))

    for t in range(num_of_trials):
            
        if t%100 == 0: # start of a new block 
            # reset q-values
            qvals[t,0] = 0
            qvals[t,1] = 0
        
        # set up data
        choice = action_list[t]
        reward = reward_list[t]

        # decision
        p = np.exp( beta*qvals[t] ) / np.sum( np.exp( beta * qvals[t] ) )
        choice_epred[t] = p[1]

        # value update (unless we're on the last trial)
        if t != (num_of_trials - 1):
            prediction_error = reward - qvals[t,choice] 
            qvals[t+1,choice] = qvals[t,choice] + alpha*prediction_error
            qvals[t+1,1-choice] = qvals[t,1-choice] # the q-value of the unchosen action stays the same

    # auc computation
    auc = roc_auc_score(action_list, choice_epred)
           
    return auc, choice_epred, qvals
    
