from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from sklearn.model_selection import KFold, train_test_split
import numpy as np

def get_init_prob (states):
    init_prob = dict ()
    
    for state in  states.keys ():
         init_prob [state] =  states [state] / sum ( states.values ())

    return init_prob

def get_states (state_seq):
     states = dict (Counter (state_seq))
     return states
     


def get_emission_prob (data_obs, data_seq, states, corpus):
    emi_prob = dict ()
    
    for state in  states.keys ():
        emi_prob [state] = dict ()
        for obs in corpus.keys ():
             emi_prob [state] [obs] = 0

    for t in range (len (data_seq)):
        for w in range (len (data_seq [t])):
             emi_prob [data_seq [t] [w]] [data_obs [t] [w]] += 1
            
    for state in  states.keys ():
        for obs in corpus.keys ():
             emi_prob [state] [obs] = ( emi_prob [state] [obs] + 1) / ( states [state] + len (corpus))

    return emi_prob
    
def get_trans_prob (data_seq, states):
    trans_prob = dict ()
    
    for state_r in  states.keys ():
        trans_prob [state_r] = dict ()
        for state_c in  states.keys ():
             trans_prob [state_r] [state_c] = 0

    for sample_seq in data_seq:
        for i in range (len (sample_seq) - 1):
             trans_prob [sample_seq [i]] [sample_seq [i + 1]] += 1
            
    for state_r in  states.keys ():
        for state_c in  states.keys ():
             trans_prob [state_r] [state_c] = ( trans_prob [state_r] [state_c] + 1) / ( states [state_r] + len ( states))

    return trans_prob

def get_forward_prob (init_prob, trans_prob, emi_prob, states, obs_seq):
    T = len (obs_seq)
    state_keys = list ( states.keys ())

    alpha = list ()
    alpha.append (dict ())

    for state in state_keys:
        if  emi_prob [state].get (obs_seq [0]) != None:
            alpha [0] [state] =  init_prob [state] *  emi_prob [state] [obs_seq [0]]
        else:
            alpha [0] [state] = 0.0

    
    for t in range (1, T):
        alpha.append (dict ())
        
        for curr_state in state_keys:
            sum_prob = 0.0
            for prev_state in state_keys:
                sum_prob += (alpha [t-1][prev_state] *  trans_prob [prev_state][curr_state])    
            
            if  emi_prob [curr_state].get (obs_seq [t]) != None:
                alpha [t] [curr_state] =  emi_prob [curr_state][obs_seq [t]] * sum_prob 
            else:
                alpha [t] [curr_state] = 0.0
    
    return alpha

def get_backward_prob (init_prob, trans_prob, emi_prob, states, obs_seq):
    T = len (obs_seq)
    state_keys = list ( states.keys ())

    beta = [ dict () for i in range (T)]

    for state in state_keys:
        beta [T-1] [state] = 1

    for t in range (T-2, -1, -1):           
        for curr_state in state_keys:
            sum_prob = 0.0
            for next_state in state_keys:
                if  emi_prob [next_state].get (obs_seq [t+1]) != None:
                    sum_prob += (beta [t+1][next_state] *  trans_prob [curr_state][next_state] *  emi_prob [next_state][obs_seq [t+1]])    
            
            beta [t] [curr_state] = sum_prob 
    
    return beta
    
def viterbi (init_prob, trans_prob, emi_prob, states, obs_seq):
    start_delim = -1
    i = 0
    tree = list ()
    tree.append (dict ())
    state_keys = list ( states.keys ())

    for state in state_keys:
        if emi_prob [state].get (obs_seq [0]) != None:
            tree [0] [state] = { 'p' : init_prob [state] * emi_prob [state] [obs_seq [0]], 'prev' : start_delim  }
        else:
            tree [0] [state] = { 'p' : 0.0, 'prev' : start_delim  }

    for i in range (1, len (obs_seq)):
        tree.append (dict ())       
        
        for curr_state in state_keys:
            max_prob = 0.0
            prev = state_keys [0]
            
            for prev_state in state_keys:
                if emi_prob [curr_state].get (obs_seq [i]) != None:
                    prob = tree [i-1][prev_state]['p'] * trans_prob [prev_state] [curr_state] * emi_prob [curr_state] [obs_seq [i]]
                else:
                    prob = 0.0
                if prob >= max_prob:
                    max_prob = prob
                    prev = prev_state
            
            tree [i] [curr_state] = { 'p' : max_prob, 'prev' : prev }
    
    max_prob_state = state_keys [0]
    for state in state_keys [1:]:
        if (tree [i] [state] ['p'] > tree [i] [max_prob_state] ['p']):
            max_prob_state = state
    
    pred_state_seq = list ()
    pred_state_seq.append (max_prob_state)
    while (i > 0):
        pred_state_seq.append ( tree [i][max_prob_state]['prev'] )
        
        max_prob_state = tree [i][max_prob_state]['prev']
        i -= 1
    
    pred_state_seq.reverse ()
    
    return pred_state_seq
   
def get_temp_variables (init_prob, trans_prob, emi_prob, states, alpha, beta, obs_seq):
    T = len (obs_seq)
    state_keys = list ( states.keys ())
    
    y = [ dict () for i in range (T) ]
    
    for t in range (0, T):
        for state in state_keys:
            sum_y = 0.0
            for all_s in state_keys:
                sum_y += (alpha [t][all_s] * beta [t][all_s])
            
            if sum_y > 0:
                y [t][state] = (alpha [t][state] * beta [t][state]) / sum_y
            else:
                y [t][state] = 0.0
    
    epi = [ dict () for i in range (T) ]
            
    for t in range (0, T-1):
        for i in state_keys:
            epi [t][i] = dict ()
            
            for j in state_keys:
                sum_epi = 0.0
                
                for k in state_keys:
                    for w in state_keys:
                        if  emi_prob [w].get (obs_seq [t+1]) != None:
                            sum_epi += ( alpha [t][k] *  trans_prob [k][w] * beta [t+1][w] *  emi_prob [w] [obs_seq [t+1]] )
                
                if  emi_prob [j].get (obs_seq [t+1]) != None and sum_epi > 0:
                    epi [t][i][j] = (alpha [t][i] *  trans_prob [i][j] * beta [t+1][j] *  emi_prob [j] [obs_seq [t+1]]) / sum_epi
                else:
                    epi [t][i][j] = 0.0
                    
    return y, epi


def train (init_prob, trans_prob, emi_prob, states, x_train, y_train, epochs):
    samples = x_train.shape [0]
    state_keys = list ( states.keys ())
    

    for epoch in range (1, epochs+1):
        print ('Epoch ', epoch, end='\r')
        alpha = list ()
        beta = list ()
        y = list ()
        epi = list ()
        
        
        for r in range (0, samples):
            alpha.append ( get_forward_prob (x_train [r]))
            beta.append ( get_backward_prob (x_train [r]))
            temp1, temp2 =  get_temp_variables (alpha [r], beta [r], x_train [r])
            y.append (temp1)
            epi.append (temp2)

            
        for state in state_keys:
            init_prob [state] = 0.0
            for r in range (0, samples):
                init_prob [state] += y [r][0][state]
            init_prob [state] /= samples
        
        
        for i in state_keys:
            for j in state_keys:
                num = 0.0
                den = 0.0
                for r in range (0, samples):
                    T = len (epi [r])
                    for t in range (0, T-1):
                        num += epi [r][t][i][j]
                        den += y [r][t][i]
                
                if den > 0:
                     trans_prob [i][j] = num / den
                else:
                     trans_prob [i][j] = 0.0
                    
        
        for i in state_keys:
            for r in range (0, samples):
                T = len (y [r])
                
                for k in x_train [r]:
                    num = 0.0
                    den = 0.0
                    for t in range (0, T):
                        if x_train [t] == k:
                            num += y [r][t][i]
                        den += y [r][t][i]
                
                if den > 0:
                     emi_prob [i][k] = num / den
                else:
                     emi_prob [i][k] = 0.0
        
    return init_prob, trans_prob, emi_prob


def get_obs_seq (sentences):
    
    corpus = dict ()
    state_seq = list ()
    
    data_obs = list ()
    data_seq = list ()
    
    for sent in sentences:
        words = sent.split(' ')
        sent_tag_list = []
        sent_word_list = []
        for word in words:
            word_split = word.rsplit('/', 1)
            word, tag = word_split[0], word_split[1]
            sent_tag_list.append(tag)
            sent_word_list.append(word)
            
            if corpus.get (word) == None:
                corpus [word] = 1
            else:
                corpus [word] += 1
            state_seq.append (tag)
                
        data_obs.append (sent_word_list) # list of all words for each sentence
        data_seq.append (sent_tag_list) # list of all tags for each sentence
        
 
    return data_obs, data_seq, corpus, state_seq

def get_all_sentences(filename):
    sentences = []
    with open(filename, 'r') as infile:
        for line in infile:
            sentences.append(line.rstrip())
    print('Number of sentences in the brown dataset are', len(sentences))
    return sentences

def get_and_write_preds (data_obs, outfile, init, trans, emi, states): 

    outputfile = open (outfile, 'w')
    all_predictions = []
    for words in data_obs:
        pred_seq = viterbi (init, trans, emi, states, words)
        all_predictions.append(pred_seq)
        for i in range (len (pred_seq)):
            outputfile.write (words [i] + '\t' + pred_seq [i] + '\n')            
        outputfile.write ('\n')

    outputfile.close ()
    return all_predictions

def flatten_state_seq (data_seq):
    vec = list ()
    
    for seq in data_seq:
        vec.extend (seq)
    
    return vec

def validate_and_write (filename, outfile):
    sentences = get_all_sentences(filename)
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=117)
    data_obs, data_seq, corpus, state_seq = get_obs_seq (train_sentences)
    test_data_obs, test_data_seq, test_corpus, test_state_seq = get_obs_seq(test_sentences)
    len_corpus = len (corpus)
    states = dict (Counter (state_seq))
    state_keys = list ( states.keys ())
    
    kf = KFold (n_splits=5, shuffle=False)
    
    np_obs = np.array (data_obs)
    np_seq = np.array (data_seq)
    
    for train_index, test_index in kf.split(np_obs):
        x_train, y_train = np_obs [train_index], np_seq [train_index]
        x_test, y_test  = np_obs [test_index], np_seq [test_index]

        init = get_init_prob (states)
        trans = get_trans_prob (y_train, states)
        emi = get_emission_prob (x_train, y_train, states, corpus)
        
        pred_seq = list ()
        for t in range (0, len (x_test)):
            pred_seq.extend (viterbi (init, trans, emi, states, x_test [t]))
        
        print (precision_recall_fscore_support(flatten_state_seq (y_test), pred_seq, labels=state_keys))
        print ()

    
    init = get_init_prob (states)
    trans = get_trans_prob (y_train, states)
    emi = get_emission_prob (x_train, y_train, states, corpus)
    
    test_pred_seq = get_and_write_preds (test_data_obs, outfile, init, trans, emi, states)
    with open('results.txt', 'w') as infile:
        print(precision_recall_fscore_support(flatten_state_seq (test_data_seq), flatten_state_seq (test_pred_seq), labels=state_keys), file=infile)
    return
        
def main ():
    filename = "Brown_train.txt"
    validate_and_write (filename, "test_pred.txt")
  
    return

if __name__ == "__main__":
    main ()
    
