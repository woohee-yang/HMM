# ~~~
import numpy as np
import matplotlib.pyplot as plt
import pylab
import matplotlib._pylab_helpers
import seaborn as sns
import math
import time, os
import codecs, json
import bokeh.plotting as bp

class HMM:
    def __init__(self, N=0, M=0, data=None, load=False, params_file=None):
        self.save_path = './results/'
        folder = 'result_' + time.strftime('%Y-%m-%d-%H%M%S')
        self.save_path += folder + '/'

        self._figures = []
        self.M = M
        self.data = data
        self.T = len(data)
        if load:
            self.load_params(params_file)
            self.N = len(self.pi)
        else :
            self.N = N
            # HMM parameters
            self.pi = self._init_random_prob(1,self.N).T
            self.A = self._init_random_prob(self.N, self.N)
            self.B = self._init_random_prob(self.N, self.M)

        self.visualization_training_result()
        # for Q function
        self._init_probs()
        self.ll = 0.0

    def _init_random_prob(self, n, m):
        tmp = np.random.rand(n, m)
        deno = np.sum(tmp, axis=1).reshape(n, -1)
        tmp /= deno
        return tmp

    def _init_probs(self):
        self._gammar = np.zeros((self.T, self.N))
        self._csi = np.zeros((self.T, self.N, self.N))
        self._alpha = np.zeros((self.T, self.N))
        self._beta = np.zeros((self.T, self.N))
        self._ct = np.zeros(self.T)
        return

    def _forward_pass(self):
        for t in range(self.T):
            for i in range(self.N):
                if t == 0:
                    self._alpha[t][i] = self.pi[i] * self.B[i][self.data[t]]
                else:
                    self._alpha[t][i] = self.B[i][self.data[t]]
                    self._alpha[t][i] *= np.sum(self._alpha[t-1] * self.A.T[i])
            # scaling = nomalization
            denom = np.sum(self._alpha[t], axis = 0)
            self._alpha[t] /= denom
            self._ct[t] += denom
        return

    def _backward_pass(self):
        for t in range(self.T):
            for i in range(self.N):
                if t == 0:
                    self._beta[0][i] = 1
                else:
                    for j in range(self.N):
                        self._beta[t][i] += self._beta[t-1][j] * self.A[i][j] * self.B[j][self.data[self.T-t]]
            # scaling
            self._beta[t] /= self._ct[t]
        return

    def _calc_gammar(self):
        for t in range(self.T):
            for i in range(self.N):
                self._gammar[t][i] = (self._ct[t]*self._alpha[t][i])
                self._gammar[t][i] *= (self._ct[self.T-t-1]*self._beta[self.T-1-t][i])
        return

    def _calc_csi(self):
        for t in range(1,self.T):
            denom = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    self._csi[t][i][j] = (self._ct[t-1]*self._alpha[t-1][i]) * self.A[i][j]
                    self._csi[t][i][j] *= self.B[j][self.data[t]] * (self._ct[self.T-1-t]*self._beta[self.T-1-t][j])
        return

    def _update_params(self):
        self._update_pi()
        self._update_transition()
        self._update_emission()
        return

    def _update_pi(self):
        deno = np.sum(self._gammar[0], axis=0)
        for i in range(self.N):
            self.pi[i] = self._gammar[0][i] / deno
        return

    def _update_transition(self):
        csi = np.sum(self._csi, axis = 0)
        for i in range(self.N):
            denom = np.sum(csi[i])
            for j in range(self.N):
                self.A[i][j] = csi[i][j] / denom
        return

    def _update_emission(self):
        denom = np.sum(self._gammar, axis=0)
        for i in range(self.N):
            for j in range(self.M):
                numer = 0
                for t in range(self.T):
                    if j == data[t]:
                        numer += self._gammar[t][i]
                self.B[i][j] = numer / denom[i]
        return

    def train(self, max_iter = 100, tol = 0.00001):
        for iter in range(max_iter):
            self._init_probs()
            # E step : Calculate the posteriors ; gammar, csi using alpha-beta
            self._forward_pass()
            self._backward_pass()

            self._calc_gammar()
            self._calc_csi()

            # M step : maximize the hmm parameters
            self._update_params()

            # evaluate the current model appropriateness
            flag = self.evaluate(iter, tol)
            if flag == -1 and iter > 0 :
                print('the differnece of loglikelihood reached to tol.')
                break
            print()
        self.save_params()
        print('training is over.\n\n')

    def evaluate(self, num, tol):
        print('====================')
        # print(self._ct)
        for t in range(self.T):
            self._ct[t] = math.log(self._ct[t])
        ll_ = np.sum(self._ct)
        ll_ = -ll_
        differ = abs(ll_ - self.ll)
        print('iteration #', num+1, ' / ll : ', ll_,' / differ : ', differ)
        if differ <= tol or ll_ < self.ll:
            return -1
        else:
            self.ll = ll_
            return 0
        # return 0

    def decode(self):
        # for decoding : viterbi
        self._delta = np.zeros(self.T)
        self._psi = np.zeros(self.T)

        delta = self.pi.reshape(1, -1) * self.B.T[self.data[0]]
        delta = delta.reshape(-1)
        # normalization
        delta /= np.sum(delta)

        t = 0
        while True:
            self._delta[t] = delta[0]
            self._psi[t] = 0
            for i in range(1,self.N):
                if self._delta[t] < delta[i] :
                    self._delta[t] = delta[i]
                    self._psi[t] = i
            t = t + 1
            if(t >= self.T): break
            delta = self._delta[t-1] * self.A[int(self._psi[t-1])] * self.B.T[self.data[t]]
            # normalization
            delta /= np.sum(delta)
        return

    def visualization(self, show=False):
        self.visualization_training_result()
        self.visualization_decoding_result()
        self.save_result_figures()
        if show:
            plt.show()

    def save_result_figures(self):
        i = 0
        for figure in self._figures:
            i += 1
            figure.savefig(self.save_path + 'figure%d.png' %i)

        return

    def visualization_training_result(self, show=False):
        fig = plt.figure()
        plt.title('HMM Parameters')
        plt.subplot(211)
        sns.heatmap(self.pi, annot=True, fmt='.5f', linewidth=6, cmap='Blues')
        plt.subplot(223)
        sns.heatmap(self.A, annot=True, fmt='.5f', linewidth=6, cmap='Blues')
        plt.subplot(224)
        sns.heatmap(self.B, annot=True, fmt='.5f', linewidth=6, cmap='Blues')
        self._figures.append(fig)
        if show: plt.show()
        # plot = bp.figure(plot_width=300, plot_height=300)
        # plot.square(x=[20,20,20,40], y=[20,20,40,40], size=[10,20,30,40], color="#74ADD1",fill_color=None)
        # bp.show(plot)
        return

    def visualization_decoding_result(self, show=False):
        fig = plt.figure()
        T = len(self.data)
        X = range(T)
        plt.plot(X, self._psi)
        plt.scatter(X, self._psi)
        for t in range(T):
            plt.text(t+0.01, self._psi[t], round(self._delta[t],3), fontsize=8)
        self._figures.append(fig)

        p = bp.figure(
            title='Viterbi result',
            x_range=(-1,self.T+1),
            y_range=(-0.5,self.N-1+0.5),
            x_axis_type='linear',
            tools = 'pan, wheel_zoom, box_zoom, reset, previewsave'
        )
        p.line(
            X,
            self._psi,
            color = '#0066cc',
            legend = 'state'
        )
        p.scatter(
            X,
            self._psi,
            color = '#000000',
        )
        bp.output_file(filename=self.save_path+'bokeh_fig.html')
        bp.save(p)
        if show :
            bp.show(p)
            plt.show()
        return

    def save_params(self):
        os.mkdir(self.save_path)
        file = str(self.N)+'-'+str(self.T)+'-params.json'

        params = {
        'initial_probs' : self.pi.tolist(),
             'transition_matrix' : self.A.tolist(),
             'emission_matrix' : self.B.tolist(),
             'alpha : ' : self._alpha.tolist(),
             'beta : ' : self._beta.tolist(),
             'scaling factor : ' : self._ct.tolist(),
             'gammar : ' : self._gammar.tolist(),
             'csi : ' : self._csi.tolist()
        }
        jsonString = json.dumps(params)

        with open(self.save_path+file, 'w') as f:
            f.write(jsonString)
        return

    def load_params(self, data_file):
        with open(data_file, 'r') as f:
            jsonString = json.loads(f.read())
            dict = jsonString
            self.pi = np.array(dict['initial_probs'])
            self.A = np.array(dict['transition_matrix'])
            self.B = np.array(dict['emission_matrix'])
        return

if __name__ == "__main__":
    with open('cointossdata.txt','r') as fp:
        line = fp.readline()
    data = [int(i) for i in line]
    hmm = HMM(2,2,data[:100])
    hmm.train()
    hmm.decode()
    hmm.visualization(False)
    # hmm = HMM(2, 2, data, True, './result_2019-08-30-1756/params.json')
    # HMM(2,2,data).load_params('./result_2019-08-30-1756/params.json')
# ~~~
