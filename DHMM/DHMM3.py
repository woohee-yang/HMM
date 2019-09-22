import numpy as np
import matplotlib.pyplot as plt
from bokeh import plotting as bp
import seaborn as sns
from math import log
import time
import os
import json
import pprint

def _normalized_random_distribution(rows, cols):
    rand_dist = np.random.rand(rows, cols)
    denominator = np.sum(rand_dist, axis=1).reshape(rows, -1)
    rand_dist /= denominator
    return rand_dist

def train(model, sequences, max_iter=100, delta = 0.0001, smoothing = 0):
    # model = DHMM(2, 2, init, A, B)
    seqs_len = len(sequences)

    old_likelihood = 0.0
    for seq in sequences:
        old_likelihood += -log(model.evaluate(seq))
    old_likelihood /= seqs_len

    print('likelihood : %f' %old_likelihood)

    for iter in range(max_iter):
        new_likelihood = 0.0
        for seq in sequences:
            model.fit(seq)
            new_likelihood += -log(model.evaluate(seq))
            new_likelihood /= seqs_len

        differ = new_likelihood - old_likelihood
        print('\n====================')
        print('iteration #', iter+1, ' / loglikelihood : ', new_likelihood, ' / differ : ', differ)
        if differ < delta:
            break
        
        old_likelihood = new_likelihood
    return model


def visualize_parameters(pi, A, B):
    fig = plt.figure()
    plt.title('HMM Parameters')
    plt.subplot(211, title='Initial Probability Distribution', ylabel='state')
    sns.heatmap(pi, annot=True, fmt='.3f', linewidth=6, cmap='Blues')
    plt.subplot(223, title='State Transition Matrix', xlabel='state_from', ylabel='state_to')
    sns.heatmap(A, annot=True, fmt='.3f', linewidth=6, cmap='Blues')
    plt.subplot(224, title='Emission Matrix', ylabel='state', xlabel='symbol')
    sns.heatmap(B, annot=True, fmt='.3f', linewidth=6, cmap='Blues')

    return fig

def visualize_best_state_path(bss, bsp):
    fig = plt.figure()
    plt.title('Viterbi result')
    X = range(len(bss))
    plt.plot(X, bss)
    plt.scatter(X, bss)

    for t in X:
        plt.text(t+0.01, bss[t], bsp[t], fontsize=8)
    
    return fig

def save_model_parameters(save_path, N, M, model):
    file = str(N) + '-' + str(M) + '-params.json'
    params = {
        'initial_prob': model.init_prob.tolist(),
        'transition_matrix': model.trans_prob.tolist(),
        'emission_matrix': model.emiss_prob.tolist()
    }
    jsonString = json.dumps(params)

    with open(save_path+file, 'w') as f:
        f.write(jsonString)

    return

def load_model_parameters(file):
    with open(file, 'r') as f:
        jsonString = json.loads(f.read())
        dict = jsonString
        pi = np.array(dict['initial_prob'])
        A = np.array(dict['transition_matrix'])
        B = np.array(dict['emission_matrix'])

    return pi, A, B

class DHMM():
    def __init__(self, nstates, nsymbols, init_prob=None, trans_prob=None, emiss_prob=None):
        self._N = nstates
        self._M = nsymbols

        self._init_prob = init_prob
        self._trans_prob = trans_prob
        self._emiss_prob = emiss_prob
    
    def __str__(self):
        return """initial_prob : \n{}\n
transition_matrix : \n{}\n
emission_matrix : \n{}""".format(self._init_prob, self._trans_prob, self._emiss_prob)

    @property
    def init_prob(self):
        return self._init_prob

    @property
    def trans_prob(self):
        return self._trans_prob

    @property
    def emiss_prob(self):
        return self._emiss_prob

    def _forward(self, sequence):
        """
        Parameters:
            sequence : 1D list

        Return:
            alpha : numpy ndarray

            scf : numpy ndarray
                scf = scaling factor
        """
        T = len(sequence)
        alpha = np.zeros((T, self._N))
        scf = np.zeros(T)

        alpha[0] = self._init_prob * self._emiss_prob.T[sequence[0]]
        scf[0] = np.sum(alpha[0], axis=0)
        for t in range(1, T):
            for state_to in range(self._N):
                prob = 0.0
                for state_from in range(self._N):
                    prob += alpha[t-1][state_from] * \
                        self._trans_prob[state_from][state_to]
                alpha[t][state_to] = prob * self._emiss_prob[state_to][sequence[t]]
            scf[t] = np.sum(alpha[t], axis=0)
            alpha[t] /= scf[t]

        return alpha, scf

    def _backward(self, sequence, scf):
        T = len(sequence)
        beta = np.zeros((T, self._N))

        beta[0] = np.ones(self._N)
        for t in range(1,T):
            for state_from in range(self._N):
                for state_to in range(self._N):
                     beta[t][state_from] += beta[t-1][state_to] *\
                         self._trans_prob[state_from][state_to] * self._emiss_prob[state_to][sequence[T-t]]
            beta[t] /= scf[T-1-t]
        return beta
    
    def _estimate_gammar(self, alpha, beta, scf):
        T = len(scf)
        gammar = np.zeros((T, self._N))

        for t in range(T):
            gammar[t] = (scf[t]*alpha[t]) * (scf[T-1-t]*beta[T-1-t]) / scf[T-1]
        
        return gammar

    def _estimate_csi(self, alpha, beta, scf, seq):
        T = len(seq)
        csi = np.zeros((T, self._N, self._N))

        for t in range(1,T):
            for state_from in range(self._N):
                for state_to in range(self._N):
                    csi[t][state_from][state_to] =\
                    (scf[t-1] * alpha[t-1][state_from]) *\
                    self._trans_prob[state_from][state_to] *\
                    self._emiss_prob[state_to][seq[t]] *\
                    (scf[T-1-t] * beta[T-1-t][state_to]) / scf[T-1]
        
        return csi
    
    def _update_init_prob(self, gammar):
        gammar_sum = np.sum(gammar[0], axis=0)
        self._init_prob = gammar[0] / gammar_sum
        return
    
    def _update_trans_prob(self, csi):
        csi_sum = np.sum(csi, axis=0)
        for state_from in range(self._N):
            denominator = np.sum(csi_sum[state_from], axis=0)
            for state_to in range(self._N):
                self._trans_prob[state_from][state_to] = csi_sum[state_from][state_to] / denominator
        return

    def _update_emiss_prob(self, gammar, seq):
        T = len(seq)
        gammar_sum = np.sum(gammar, axis=0)

        for state in range(self._N):
            for symbol in range(self._M):
                numerator = 0.0
                for t in range(T):
                    if symbol == seq[t]:
                        numerator += gammar[t][state]
                self._emiss_prob[state][symbol] = numerator / gammar_sum[state]
        return

    def fit(self, sequence, smoothing=0):
        T = len(sequence)
        if T == 0:
            return -1
        # E-step : estimating the posteriors of latent variables using forward-backward algorithm
        alpha, scf = self._forward(sequence)
        beta = self._backward(sequence, scf)

        gammar = self._estimate_gammar(alpha, beta, scf)
        csi = self._estimate_csi(alpha, beta, scf, sequence)

        # M-step : maximize the model parameters
        self._update_init_prob(gammar)
        self._update_trans_prob(csi)
        self._update_emiss_prob(gammar, sequence)
        return
    
    def evaluate(self, seq):
        _, scf = self._forward(seq)
        return scf[len(seq)-1]
    
    def decode(self, seq):
        print("Decoding starts.")

        T = len(seq)
        if T <= 0:
            return -1
        
        delta = np.zeros((T, self._N))
        psi = np.zeros((T, self._N))

        delta[0] = self._init_prob * self._emiss_prob.T[seq[0]]
        for t in range(1,T):
            for state_to in range(self._N):
                tmp = delta[t-1] * self._trans_prob.T[state_to]
                max_prob = np.max(tmp)
                max_state = np.argmax(tmp)

                delta[t][state_to] = max_prob * self._emiss_prob[state_to][seq[t]]
                psi[t][state_to] = max_state
        # backtracking
        best_states_seq = np.zeros(T)
        best_states_prob = np.zeros(T)

        for t in range(T-1, 1, -1):
            best_states_prob[t] = np.max(delta[t])
            best_states_seq[t] = psi[t][np.argmax(delta[t-1])]

        return best_states_seq, best_states_prob


if __name__ == '__main__':
    with open('cointossdata.txt', 'r') as fp:
        line = fp.readline()
    data = [int(i) for i in line]
    data = [data[:100]]
    nstates = 2
    nsymbols = 2

    data = [[0,0,1,0]]

    #init_prob = _normalized_random_distribution(1, nstates)
    #trans_prob = _normalized_random_distribution(nstates, nstates)
    #emiss_prob = _normalized_random_distribution(nstates, nsymbols)
    
    init_prob = np.array([0.8, 0.2])
    trans_prob = np.array([[0.6, 0.4], [0.3, 0.7]])
    emiss_prob = np.array([[0.2, 0.8], [0.5, 0.5]])

    dhmm = DHMM(nstates, nsymbols, init_prob, trans_prob, emiss_prob)
    # print(dhmm)
    dhmm = train(dhmm, data, max_iter = 10)
    bss, _, = dhmm.decode(data[0])
    print(bss)


    #best_states_seq, best_states_prob = dhmm.decode(data[0])

    save_path = './new_results/'
    folder = 'result_' + time.strftime('%Y-%m-%d-%H%M%S')
    save_path += folder + '/'
    #os.mkdir(save_path)
    
    #save_model_parameters(save_path, nstates, nsymbols, dhmm)

    #figures = []
    #figures.append(visualize_parameters(dhmm.init_prob.reshape(nstates, -1), dhmm.trans_prob, dhmm.emiss_prob))
    #figures.append(visualize_best_state_path(best_states_seq, best_states_prob))
    #i = 0
    #for fig in figures:
    #    i += 1
    #    fig.savefig(save_path + 'figure%d.png' % i)
    # plt.show()

