import matlab
import matlab.engine
import numpy as np
import scipy
import copy
from tqdm import tqdm
import powerlaw
from scipy.stats import powerlaw
from scipy.stats import pareto
from scipy.stats import cauchy
from scipy.stats import levy
from scipy.stats import t
from scipy.stats import fisk
from scipy import special
from numpy import linalg as LA
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.sparse import spdiags
from scipy.sparse.linalg import norm as sparse_norm
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pylab import rcParams
import pickle
from matplotlib import rc
from matplotlib import ticker
import matplotlib
import matplotlib as mpl
import ast
import mpld3
import time
from math import ceil
from math import sqrt
mpld3.enable_notebook()

plt.style.use('seaborn-paper')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150

eng = matlab.engine.start_matlab()

class Params(object):
    def __init__(self, mu = [10, -5, -4, 2], m = 500, tau = 0.2, d = 500, k = 4, eps = 0.1, var = 1, nItrs = 0, mass = 0, tv = 0, fv = 0, group_size = 4, param = 1, J = 40):
        self.m = m                      #Number of Samples
        self.d = d                      #Dimention
        self.k = k                      #Sparsity
        self.eps = eps                  #Corruption Proportion
        self.mu = mu                    #True Mean
        self.var = var                  #Variancce
        self.tau = tau                  #Delta
        self.nItrs = nItrs              #Iterations
        self.mass = mass
        self.tv = tv
        self.fv = fv
        self.group_size = group_size
        self.param = param
        self.J = J
        
    def tm(self):
        tm = np.append(self.mu, np.zeros(self.d-self.k))
        if len(tm) > self.d: return tm[:self.d]
        if len(tm) < self.d: return np.append(tm, np.zeros(self.d-len(tm)))
        return tm                       #Sparse Mean


def err(a, b): return LA.norm(a-b)


def acc(a, b):

    common_set = set(a) & set (b)
    all_set = set(a) | set(b)
    acc = len(common_set)/len(all_set)

    return acc


class RunCollection(object):
    def __init__(self, func, inp):
        self.runs = []
        self.func = func
        self.inp = inp

    def run(self, trials):
        for i in tqdm(range(trials)):
            self.runs.append(self.func(*self.inp))


class ParetoModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, var, tm, param = params.m, params.d, params.var, params.tm(), params.param

        S = np.zeros((m, d))
        for i in range(m):
            for j in range(d):
                S[i][j] = var * pareto.rvs(param) * (2 * np.random.randint(0,2) - 1)
        
        S = S + tm

        return S, tm
    

class TModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, var, tm, param = params.m, params.d, params.var, params.tm(), params.param

        S = var * t.rvs(param, size = (m,d)) + tm

        return S, tm
    

class FiskModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, var, tm, param = params.m, params.d, params.var, params.tm(), params.param
        #print('tm and k')
        print(tm)
        print(params.k)
        S = np.zeros((m, d))
        for i in range(m):
            for j in range(d):
                S[i][j] = var * fisk.rvs(param) * (2 * np.random.randint(0,2) - 1)
        
        S = S + tm

        return S, tm


class LognormalModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, tm, var = params.m, params.d, params.tm(), params.var

        #S = np.random.lognormal(np.ones(d), var, (m, d))
        S = np.zeros((m, d))
        for i in range(m):
            for j in range(d):
                S[i][j] = var * np.random.lognormal() * (2 * np.random.randint(0,2) - 1)
        #print(tm)
        S = S + tm

        return S, tm
    

class DenseNoise(object):
    def __init__(self, dist):
        self.dist = dist

    def generate(self, params, S):
        eps, m = params.eps, params.m

        G = S.copy()

        L = int(m * (1 - eps))

        G[L:] += self.dist
        
        return G
    

def pre_processing(params, S):

    m = params.m
    eps = params.eps
    idx = np.arange(m)
    np.random.shuffle(idx)
    K = min(int(1.5 * ceil(eps * m) + 150),int(m/2))
    idx_split = np.array_split(idx, K)
    X_grouped = []

    for i in range(K):
        idx_tmp = idx_split[i]
        S_tmp = [S[j] for j in idx_tmp]
        X_grouped.append(list(np.mean(S_tmp, axis = 0)))

    X_grouped = np.array(X_grouped)

    params.m = K
    params.eps = ceil(eps * m) / K

    return params, X_grouped


class Top_K(object):

    def __init__(self, params):
        self.params = params

    def GD(self, S, iter_num):

        d = self.params.d
        m = self.params.m

        alpha = 1e-5

        u = alpha * np.ones(d)
        v = alpha * np.ones(d)

        eta = 0.05
        rho = 1
        max_iter = iter_num

        for t in range(max_iter):
            grad_u = np.zeros(d)
            grad_v = np.zeros(d)
            for i in range(m):
                grad_u += - \
                    np.sign(S[i, :].reshape(d) - u * u + v * v) * u
                grad_v += np.sign(S[i,
                                  :].reshape(d) - u * u + v * v) * v
            u -= eta * grad_u / m
            v -= eta * grad_v / m
            eta *= rho

        estimated_mean = u * u - v * v
        top_k_indices = []

        for i in range(len(estimated_mean)):
            if np.abs(estimated_mean[i]) >= alpha:
                top_k_indices.append(i)
        print("Prediction:", top_k_indices)
        return top_k_indices

    def alg(self, S):
        
        params, S = pre_processing(self.params, S)
        self.params = params
        pred_k = self.GD(S, 200)

        return pred_k
    

class load_data(RunCollection):

    def __init__(self, model, noise_model, params, loss, keys=[]):

        self.params = params
        self.keys = keys
        self.model = model
        self.noise_model = noise_model
        self.loss = loss
        self.inp = 0
        self.Run = 0

    def get_dataxy(self, xvar_name, xs=[]):

        results = {}

        for xvar in xs:
            if xvar_name == 'm':
                self.params.m = xvar
            elif xvar_name == 'k':
                self.params.k = xvar
                self.params.mu = np.ones(xvar) * 2
            elif xvar_name == 'd':
                self.params.d = xvar
            elif xvar_name == 'eps':
                self.params.eps = xvar

            elif xvar_name == 'param':
                self.params.param = xvar

            elif xvar_name == 'var':
                self.params.var = xvar

            elif xvar_name == 'group_size':
                self.params.group_size = xvar

            elif xvar_name == 'm_k':
                self.params.k = xvar
                self.params.m = xvar * 100
                self.params.mu = np.ones(xvar) * 2

            elif xvar_name == 'test':
                self.params.k = xvar
                self.params.mu = np.ones(xvar) * 2
                if xvar_name == 5:
                    self.params.m = 400
                if xvar_name == 25:
                    self.params.m = 10000

            elif xvar_name == 'sen':
                self.params.k = xvar

            elif xvar_name == 'acc':
                self.params.eps = xvar
                self.params.mu = np.ones(self.params.k) * 2

            S, tm = self.model.generate(self.params)
            S = self.noise_model.generate(self.params, S)

            for f in self.keys:
                inp_copy = copy.copy(self.params)
                S_copy = copy.deepcopy(S)

                func = f(inp_copy)

                pred_k = func.alg(S_copy)
                results.setdefault(f.__name__, []).append(
                    self.loss(pred_k, list(np.arange(inp_copy.k))))

        return results

    def setdata_tofile(self, filename, xvar_name, trials, xs=[]):
        start_time = time.perf_counter()
        self.setdata(xvar_name, trials, xs)
        with open(filename, 'wb') as g:
            pickle.dump(self.Run.runs, g, -1)
        
        end_time = time.perf_counter()
        runtime = end_time - start_time
        #print("runtime:", runtime)

    def setdata(self, xvar_name, trials, xs=[]):

        Runs_l_samples = RunCollection(
            self.get_dataxy, (xvar_name, xs))
        Runs_l_samples.run(trials)
        self.Run = Runs_l_samples


class plot_data(RunCollection):

    def __init__(self, model, noise_model, params, loss, keys=[]):

        self.params = params
        self.keys = keys
        self.model = model
        self.noise_model = noise_model
        self.loss = loss
        self.inp = 0
        self.Run = 0

    def readdata(self, filename):
        with open(filename, 'rb') as g:
            ans = pickle.load(g)
        return ans

    def plot_xloss(self, outputfilename, runs, title, xlabel, ylabel, xs=[], fsize=10, fpad=10, figsize=(1, 1), fontname='Arial', yscale = 'linear'):

        markers = {'RME_sp': 'o',
                   'RME_sp_L': 'v',
                   'RME': '^',
                   'ransacGaussianMean': 'D',
                   'NP_sp': 'p',
                   'Oracle': 'x',
                   'Top_K': '.',
                   'GDAlgs':'^',
                   'Top_K_Filtered': 'o',
                   'Topk_GD':'*',
                   'NP_sp_npre': 'p',
                   'RME_sp_npre': 'o', 
                   'RME_sp_L_npre': 'v', 
                   'RME_npre': '^', 
                   'GDAlgs_npre': '^',
                   'GD_nonsparse': '*',
                   'Stage2_GD': 'o',
                  'Stage2_filter': 'p',
                  'Top_K_Filtered_RME': '*'
                   }

        labels = {'NP_sp': 'NP_sp',
                  'ransacGaussianMean': 'RANSAC',
                  'RME_sp': 'Filter_sp_LQ',
                  'RME_sp_L': 'Filter_sp_L',
                  'Oracle': 'Oracle',
                  'RME': 'Filter_nsp',
                  'Top_K': 'Stage 1',
                  'Top_K_Filtered': 'Full',
                  'GDAlgs': 'Sparse GD',
                  'Topk_GD': 'Full',
                  'NP_sp_npre': 'NP_sp_npre',
                  'RME_sp_npre': 'Filter_sp_LQ_npre', 
                  'RME_sp_L_npre': 'Filter_sp_L_npre', 
                  'RME_npre': 'Filter_nsp_npre', 
                  'GDAlgs_npre': 'Sparse GD_npre',
                  'GD_nonsparse': 'GD_nonsparse',
                  'Stage2_GD': 'Stage2_GD',
                  'Stage2_filter': 'Stage2_filter',
                  'Top_K_Filtered_RME': 'Full'
                  }

        s = len(runs)
        #print(runs)
        str_keys = [key.__name__ for key in self.keys]
        #print(str_keys)
        #str_keys_time = [key.__name__ + '_time' for key in self.keys]
        #print(str_keys_time)

        for key in str_keys:
            #print(key)
            A = np.array([res[key] for res in runs])
            #print(A)
            '''
            if explicit_xs == False:
                xs = np.arange(*bounds)
            else:
                xs = xs
            '''
            mins = [np.sort(x)[int(s*0.25)] for x in A.T]
            maxs = [np.sort(x)[int(s*0.75)] for x in A.T]

            plt.fill_between(xs, mins, maxs,alpha=0.2)
            plt.plot(xs, np.median(A, axis=0),
                     label=labels[key], marker=markers[key])

        #p = copy.copy(self.params)

        rcParams['figure.figsize'] = figsize

        rc('font', family=fontname, size=fsize)
        rc('axes', labelsize='large')
        rc('legend', numpoints=1)


        plt.title(title, pad=fpad, fontsize=fsize)
        plt.xlabel(xlabel, fontsize=fsize, labelpad=fpad)
        plt.ylabel(ylabel, labelpad=fpad, fontsize=fsize)
        plt.xticks(color='k', fontsize=12)
        plt.yticks(color='k', fontsize=12)
        plt.legend(prop={'size' : 14})
        plt.yscale(yscale)
        plt.xscale(yscale)
        #plt.xlim(1,100)
        #plt.ylim(*ylims)
        plt.savefig(outputfilename, bbox_inches='tight')
        plt.tight_layout()

    def plot_xloss_fromfile(self, outputfilename, filename, title, xlabel, ylabel, xs=[], fsize=10, fpad=10, figsize=(1, 1), fontname='Arial', yscale = 'linear'):
        Run = self.readdata(filename)
        self.plot_xloss(outputfilename, Run, title, xlabel, ylabel,
                        xs, fsize, fpad, figsize, fontname, yscale)

    def plotxy_fromfile(self, outputfilename, filename, title, xlabel, ylabel, figsize=(1, 1), fsize=10, fpad=10, xs=[], fontname='Arial', yscale='linear'):

        self.plot_xloss_fromfile(outputfilename, filename, title, xlabel, ylabel, xs=xs, figsize=figsize,
                                 fsize=fsize, fpad=fpad, fontname=fontname, yscale=yscale)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.figure()

    def plot_3_xloss(self, outputfilename, runs1, runs2, runs3, title, xlabel, ylabel, xs=[], fsize=10, fpad=10, figsize=(1, 1), fontname='Arial', yscale = 'linear'):

        cols = {'RME_sp': 'b', 'RME_sp_L': 'g', 'RME': 'r', 'ransacGaussianMean': 'y',
                'NP_sp': 'k', 'Oracle': 'tab:green', 'Top_K': 'tab:blue', 'Top_K_Filtered': 'tab:orange', 'GDAlgs':'sandybrown', 'Topk_GD':'tomato',
                'NP_sp_npre': 'gray', 'RME_sp_npre': 'skyblue', 'RME_sp_L_npre': 'springgreen', 'RME_npre': 'tomato', 'GDAlgs_npre': 'peachpuff', 'GD_nonsparse': 'plum'
                }

        markers = {'RME_sp': 'o',
                   'RME_sp_L': 'v',
                   'RME': '^',
                   'ransacGaussianMean': 'D',
                   'NP_sp': 'p',
                   'Oracle': 'x',
                   'Top_K': '.',
                   'GDAlgs':'^',
                   'Top_K_Filtered': 'o',
                   'Topk_GD':'*',
                   'NP_sp_npre': 'p',
                   'RME_sp_npre': 'o', 
                   'RME_sp_L_npre': 'v', 
                   'RME_npre': '^', 
                   'GDAlgs_npre': '^',
                   'GD_nonsparse': '*',
                   'Stage2_GD': 'o',
                  'Stage2_filter': 'p'
                   }

        labels = {'NP_sp': 'NP_sp',
                  'ransacGaussianMean': 'RANSAC',
                  'RME_sp': 'Filter_sp_LQ',
                  'RME_sp_L': 'Filter_sp_L',
                  'Oracle': 'Oracle',
                  'RME': 'Filter_nsp',
                  'Top_K': 'Stage 1',
                  'Top_K_Filtered': 'Full',
                  'GDAlgs': 'Sparse GD',
                  'Topk_GD': 'Topk_GD',
                  'NP_sp_npre': 'NP_sp_npre',
                  'RME_sp_npre': 'Filter_sp_LQ_npre', 
                  'RME_sp_L_npre': 'Filter_sp_L_npre', 
                  'RME_npre': 'Filter_nsp_npre', 
                  'GDAlgs_npre': 'Sparse GD_npre',
                  'GD_nonsparse': 'GD_nonsparse',
                  'Stage2_GD': 'Stage2_GD',
                  'Stage2_filter': 'Stage2_filter'
                  }

        fig, axs = plt.subplots(1, 3, figsize=(12, 2.5))
        runs = [runs1, runs2, runs3]
        titles = title

        for i in range(3):
            s = len(runs[i])
            #print(runs)
            str_keys = [key.__name__ for key in self.keys]
            #print(str_keys)
            #str_keys_time = [key.__name__ + '_time' for key in self.keys]
            #print(str_keys_time)

            for key in str_keys:
                #print(key)
                A = np.array([res[key] for res in runs[i]])
                #print(A)
                '''
                if explicit_xs == False:
                    xs = np.arange(*bounds)
                else:
                    xs = xs
                '''
                mins = [np.sort(x)[int(s*0.25)] for x in A.T]
                maxs = [np.sort(x)[int(s*0.75)] for x in A.T]

                axs[i].fill_between(xs, mins, maxs,alpha=0.2, color=cols[key])
                axs[i].plot(xs, np.median(A, axis=0),
                        label=labels[key], marker=markers[key], color=cols[key])
                axs[i].set_xlabel('$k$')
                axs[i].set_title(titles[i], fontsize=12)
                axs[i].set_xlim(5, 100)
                axs[i].legend(loc='lower left', fontsize=10)

        #p = copy.copy(self.params)

        rcParams['figure.figsize'] = figsize

        rc('font', family=fontname, size=fsize)
        rc('axes', labelsize='large')
        rc('legend', numpoints=1)

        # plt.title(title, pad=fpad, fontsize=fsize)
        # plt.xlabel(xlabel, fontsize=fsize, labelpad=fpad)
        # plt.ylabel(ylabel, labelpad=fpad, fontsize=fsize)
        # plt.xticks(color='k', fontsize=12)
        # plt.yticks(color='k', fontsize=12)
        fig.text(0.08, 0.5, 'success rate', va='center', rotation='vertical', fontsize=12)
        # plt.legend(prop={'size' : 14})
        plt.yscale(yscale)
        # plt.xlim(5,100)
        #plt.ylim(*ylims)
        plt.savefig(outputfilename, bbox_inches='tight')
        plt.tight_layout()

    def plot_3_xloss_fromfile(self, outputfilename, filename1, filename2, filename3, title, xlabel, ylabel, xs=[], fsize=10, fpad=10, figsize=(1, 1), fontname='Arial', yscale = 'linear'):
        Run1 = self.readdata(filename1)
        Run2 = self.readdata(filename2)
        Run3 = self.readdata(filename3)
        self.plot_3_xloss(outputfilename, Run1, Run2, Run3, title, xlabel, ylabel,
                        xs, fsize, fpad, figsize, fontname, yscale)

    def plotxy_3_fromfile(self, outputfilename, filename1, filename2, filename3, title, xlabel, ylabel, figsize=(1, 1), fsize=10, fpad=10, xs=[], fontname='Arial', yscale='linear'):

        self.plot_3_xloss_fromfile(outputfilename, filename1, filename2, filename3, title, xlabel, ylabel, xs=xs, figsize=figsize,
                                 fsize=fsize, fpad=fpad, fontname=fontname, yscale=yscale)
        plt.figure()


def topk_abs(v, k):
    u = np.argpartition(np.abs(v), -k)[-k:]
    z = np.zeros(len(v))
    z[u] = v[u]
    return z


def trim_k_abs(v, k):
    # print("v: ", v)
    u = np.argpartition(np.abs(v), -k)[-k:]
    # print("u: ", u)
    # print(2)
    return v[u]


def trim_idx_abs(v, idx):
    z = np.zeros(len(v))
    if len(idx) == 0: return z
    if len(idx) == 1: return topk_abs(v, 2)
    for i in idx:
        z[i] = v[i]

    return z