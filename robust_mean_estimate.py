import matlab
import matlab.engine
import numpy as np
import scipy
import copy
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
from pylab import rcParams
import pickle
from matplotlib import rc
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
        return tm                       #Sparse Mean
        

def err_rspca(a, b): return LA.norm(np.outer(a, a)-np.outer(b, b))


def err(a, b):
    # print("estimated: ", a)
    #if a.shape != b.shape:
        #k = a.shape[0]
        #b = trim_k_abs(b, k)
        # print(k)
        # print(trim_k_abs(b, k))
    # print("tm: ", b)
    return LA.norm(a-b)


class RunCollection(object):
    def __init__(self, func, inp):
        self.runs = []
        self.func = func
        self.inp = inp

    def run(self, trials):
        for i in range(trials):
            self.runs.append(self.func(*self.inp))


def get_error(f, params, tm):
    return LA.norm(tm - f(params))
        

"""Data generate"""


class GaussianModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, tm, var = params.m, params.d, params.tm(), params.var

        S = var * np.random.randn(m, d) + tm
        #print(S)
        print(tm)

        return S, tm


'''
class PowerlawModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, k, mu = params.m, params.d, params.k, params.mu

        tm = np.append(mu, np.zeros(d-k))   #Sparse Mean
        np.random.shuffle(tm)

        S = np.empty((m, d))
        alpha = 2.5
        xmin = 1
        mean = np.zeros((m, d))
        mean += (alpha / (alpha - 1)) * xmin
        dist = powerlaw.Power_Law(xmin = xmin, parameters = [alpha])
        for i in range(m):
            S[i, :] = dist.generate_random(d)
        S = S - mean + tm

        return S, tm
'''


class PowerlawModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, var, tm = params.m, params.d, params.var, params.tm()

        a = 0.659
        S = np.zeros((m, d))
        for i in range(m):
            for j in range(d):
                S[i][j] = var * powerlaw.rvs(a) * (2 * np.random.randint(0,2) - 1)
        
        S = S + tm

        return S, tm
        

class ParetoModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, var, tm, param = params.m, params.d, params.var, params.tm(), params.param

        #a = 2.62
        S = np.zeros((m, d))
        for i in range(m):
            for j in range(d):
                S[i][j] = var * pareto.rvs(param) * (2 * np.random.randint(0,2) - 1)
        
        S = S + tm

        return S, tm

'''
class CauchyModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, tm, var = params.m, params.d, params.tm(), params.var

        S = var * cauchy.rvs(size = (m, d)) + tm

        return S, tm
'''
    

class TModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, var, tm, param = params.m, params.d, params.var, params.tm(), params.param

        #df = 2.74
        S = var * t.rvs(param, size = (m,d)) + tm

        return S, tm
    

class FiskModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        m, d, var, tm, param = params.m, params.d, params.var, params.tm(), params.param

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
        #print(S)
        print(tm)
        S = S + tm

        return S, tm


"""Noise generate"""


class DenseNoise(object):
    def __init__(self, dist):
        self.dist = dist

    def generate(self, params, S):
        eps, m = params.eps, params.m

        G = S.copy()

        L = int(m * (1 - eps))

        G[L:] += self.dist

        indicator = np.ones(len(G))
        indicator[L:] = 0

        print(G)
        
        return G, indicator
    

class GaussianNoise(object):
    def __init__(self, noise_mean = 20, noise_var = 50):
        self.noise_mean = noise_mean
        self.noise_var = noise_var

    def generate(self, params, S):
        eps, m, d = params.eps, params.m, params.d

        Gaussian_Noise = self.noise_var * np.random.randn(m, d) + self.noise_mean

        G = S.copy()

        L = int(m * (1 - eps))

        G[L:] = Gaussian_Noise[L:]

        indicator = np.ones(len(G))     #G[i] is noise iff indicator[i] = 1
        indicator[L:] = 0
        return G, indicator


"""Data pre processing"""


def pre_processing(params, S, indicator):

    m = params.m
    eps = params.eps
    idx = np.arange(m)
    np.random.shuffle(idx)
    K = min(int(1.5 * ceil(eps * m) + 150),int(m/2))
    idx_split = np.array_split(idx, K)
    X_grouped = []
    indicator_preprocessing = np.ones(K)
    for i in range(K):
        idx_tmp = idx_split[i]
        S_tmp = [S[j] for j in idx_tmp]
        X_grouped.append(list(np.mean(S_tmp, axis = 0)))
        for h in idx_tmp:
            indicator_preprocessing[i] *= indicator[h]
    '''
    X_split = np.array_split(S, K)
    X_grouped = []
    for i in X_split:
        X_grouped.append(list(np.mean(i, axis = 0)))
    '''
    #X_grouped = np.mean(X_grouped, axis=1)
    #print(X_grouped)
    X_grouped = np.array(X_grouped)
    print('m = {m} change to K = {K}'.format(m = m, K = K))
    params.m = K
    params.eps = ceil(eps * m) / K
    return params, X_grouped, indicator_preprocessing


"""Algorithm"""


class FilterAlgs(object):

    do_plot_linear = False
    do_plot_quadratic = False

    qfilter = True
    lfilter = True

    verbose = True

    is_sparse = True
    dense_filter = False

    figure_no = 0

    fdr = 0.1

    def __init__(self, params):
        self.params = params
        pass

    """ Tail estimates """

    def drop_points(self, S, indicator, x, tail, plot=False, f=0):
        eps = self.params.eps
        m = self.params.m
        d = self.params.d
        fdr = self.fdr

        l = len(S)
        p_x = tail(x)
        p_x[p_x > 1] = 1

        sorted_idx = np.argsort(p_x)
        sorted_p = p_x[sorted_idx]

        T = l - np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1])
        if T > 0.6*l:
            T = 0

        idx = np.nonzero((p_x >= sorted_p[T]))

        if len(S) == len(idx[0]):
            tfdr = 0
        else:
            tfdr = (sum(indicator) -
                    sum(indicator[idx[0]]))/(len(S)-len(idx[0]))

        if plot == True:
            plt.plot(np.arange(l), sorted_p)
            plt.plot(T*np.ones(100), 0.01*np.arange(100), linewidth=3)
            plt.plot(np.arange(l),
                     indicator[sorted_idx], linestyle='-.', linewidth=3)
            plt.plot([0, len(S)], [0, fdr], '--')
            plt.title("sample size {}, T = {}, True FDR = {}, tail = {}".format(
                l, T, tfdr, tail.__name__))
            plt.xlabel("Experiments")
            plt.ylabel("p-values")
            plt.figure(f)

        return idx

    def tail_m(self, T):

        eps, k, m, d, tau = self.params.eps, self.params.k, self.params.m, self.params.d, self.params.tau

        return (special.erfc(T/np.sqrt(2)) + (eps**2)/(np.log(k*np.log(m*d/tau))*T**2))

    def tail_c(self, T):

        eps, k, m, d, tau = self.params.eps, self.params.k, self.params.m, self.params.d, self.params.tau
        exponent = T/2 + 1./2 - np.sqrt(1 + 2*T)/2
        v = 2*np.exp(-exponent) + (eps**2/(2*T*(np.log(T)**2)))
        # idx = np.nonzero((T < 6))
        # v = 3*np.exp(-T/3) + (eps**2/(T*(np.log(T)**2)))
        # v[idx] = 1

        return v

    def tail_t(self, T):
        """
        True tail.
        """
        eps, k, m, d, tau = self.params.eps, self.params.k, self.params.m, self.params.d, self.params.tau

        return 8*np.exp(-T**2/2) + 8*eps/((T**2)*np.log(d*np.log(d/eps*tau)))

    def linear_filter(self, S, indicator, ev, v, u):
        eps = self.params.eps

        if ev > 1 + eps*np.sqrt(np.log(1/eps)):
            if self.verbose:
                print("Linear filter...")
            l = len(S)
            S_u = S[np.ix_(np.arange(l), u)]
            dots = S_u.dot(v)
            m2 = np.median(dots)

            if self.dense_filter == False:
                x = np.abs(dots - m2) - 3*np.sqrt(ev*eps)
                idx = self.drop_points(
                    S, indicator, x, self.tail_m,  self.do_plot_linear, self.figure_no)
            else:
                x = np.abs(dots - m2)
                idx = self.drop_points(
                    S, indicator, x, self.tail_t,  self.do_plot_linear, self.figure_no)

            if self.verbose:
                bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
                print(
                    f"Filtered out {l - len(idx[0])}/{l}, {bad_filtered} false ({bad_filtered / (l - len(idx[0])):0.2f} vs {self.fdr})")
            return idx
        else:
            return (np.arange(len(S)),)

    def quadratic_filter(self, S, indicator, M_mask):

        print("Quadratic filter...")
        l = len(indicator)
        mu_e = np.mean(S, axis=0)
        x = np.abs(p(S, mu_e, M_mask))

        idx = self.drop_points(S, indicator, x, self.tail_c,
                               self.do_plot_quadratic, self.figure_no)

        if self.verbose:
            bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
            print(
                f"Filtered out {l - len(idx[0])}/{l}, {bad_filtered} false ({bad_filtered / (l - len(idx[0])):0.2f} vs {self.fdr})")
            return idx
        else:
            return (np.arange(len(S)),)

    def update_params(self, S, indicator, idx):
        S, indicator = S[idx], indicator[idx]
        self.params.m = len(S)
        return S, indicator

    def alg(self, S, indicator):

        start_time = time.time()
        k = self.params.k
        d = self.params.d
        m = self.params.m
        eps = self.params.eps
        tau = self.params.tau

        T_naive = np.sqrt(2*np.log(m*d/tau))
        med = np.median(S, axis=0)
        idx = (np.max(np.abs(med-S), axis=1) < T_naive)
        S, indicator = self.update_params(S, indicator, idx)

        if len(idx) < self.params.m:
            print("NP pruned {self.params.m - len(idx) f} points")

        while True:
            if self.lfilter == False and self.qfilter == False:
                break

            if len(S) == 0:
                print("No points remaining.")

                return 0, time.time() - start_time

            if len(S) == 1:
                print("1 point remaining.")
                return 0, time.time() - start_time

            cov_e = np.cov(S, rowvar=0)
            M = cov_e - np.identity(d)
            (mask, u) = indicat(M, k)
            M_mask = mask*M

            pre_filter_length = self.params.m

            if self.dense_filter == False:
                if LA.norm(M_mask) < eps*(np.log(1/eps)):
                    print("Valid output")
                    break

            if self.lfilter == True:

                if self.dense_filter == False:
                    cov_u = cov_e[np.ix_(u, u)]
                    ev, v = scipy.linalg.eigh(cov_u, eigvals=(k-1, k-1))
                    v = v.reshape(len(v),)
                else:
                    ev, v = scipy.linalg.eigh(cov_e, eigvals=(d-1, d-1))
                    if ev < 1+eps*np.log(1/eps):
                        print("RME exited properly")
                        break
                    v = v.reshape(len(v),)
                    u = np.arange(d)

                x = self.params.m
                idx = self.linear_filter(S, indicator, ev, v, u)[0]
                # print(idx)
                self.figure_no += 1
                S, indicator = self.update_params(S, indicator, idx)
                if len(idx) < x:
                    continue

            if self.qfilter == True:

                x = self.params.m
                idx = self.quadratic_filter(S, indicator, M_mask)[0]
                # print(idx)
                self.figure_no += 1
                S, indicator = self.update_params(S, indicator, idx)
                print("condition", len(idx), x)
                if len(idx) < x:
                    continue

            if pre_filter_length == len(idx):
                print("Could not filter")
                break

        total_time = time.time() - start_time

        if self.is_sparse == True:
            # print(topk_abs(np.mean(S, axis=0), k))
            return topk_abs(np.mean(S, axis=0), k), total_time, 
        else:
            return np.mean(S, axis=0), total_time


class NP_sp_npre(FilterAlgs):

    lfilter, qfilter = False, False       


class NP_sp(FilterAlgs):

    lfilter, qfilter = False, False

    def alg(self, S, indicator):
        params, S, indicator = pre_processing(params = self.params, S = S, indicator = indicator)
        self.params = params

        return super().alg(S = S, indicator = indicator)


class RME_sp_npre(FilterAlgs):

    lfilter, qfilter = True, True


class RME_sp(FilterAlgs):

    lfilter, qfilter = True, True

    def alg(self, S, indicator):
        params, S, indicator = pre_processing(params = self.params, S = S, indicator = indicator)
        self.params = params

        return super().alg(S = S, indicator = indicator)


class RME_sp_L_npre(FilterAlgs):

    lfilter, qfilter = True, False


class RME_sp_L(FilterAlgs):

    lfilter, qfilter = True, False

    def alg(self, S, indicator):
        params, S, indicator = pre_processing(params = self.params, S = S, indicator = indicator)
        self.params = params

        return super().alg(S = S, indicator = indicator)


class RME_npre(FilterAlgs):

    lfilter, qfilter = True, False
    dense_filter = True
    # do_plot_linear = True


class RME(FilterAlgs):

    lfilter, qfilter = True, False
    dense_filter = True
    # do_plot_linear = True

    def alg(self, S, indicator):
        params, S, indicator = pre_processing(params = self.params, S = S, indicator = indicator)
        self.params = params

        return super().alg(S = S, indicator = indicator)


class Stage2_filter(FilterAlgs):

    lfilter, qfilter = True, True

    def __init__(self, params):
        self.params = params

    def top_k_extract(self, arr, k):
        """Output the top k indices of arr."""
        return np.argpartition(np.abs(arr), -k)[-k:]

    def trim_data(self, S, top_indices):
        """Set the non-top-k coordinates to 0 for every data point."""
        k = self.params.k
        S_new = S[:, top_indices]
        self.params.d = k
        return S_new

    def alg(self, S, indicator):
        """Main algorithm."""
        params, S, indicator = pre_processing(self.params, S, indicator)
        self.params = params
        start_time = time.time()
        k = self.params.k
        d = self.params.d
        #stage1_mean, pred_k = Top_K.GD(self, S, 200)
        S_trimmed = S[:,0:k]
        self.params.d = k
        time_stage_1 = time.time() - start_time
        #self.params.d = pred_k
        mean, time_stage_2  =  super().alg(S_trimmed, indicator)
        #print('ATTENTION!')
        #print(mean)
        if type(mean) == int:
            mean = np.zeros(d)
        #print(top_indices)
        #print(stage1_mean)
        final_mean = np.zeros(d)
        #print(final_mean)
        #for i in range(len(top_indices)):

            #final_mean[top_indices[i]] = mean[i]

        for i in range(k):
            final_mean[i] = mean[i]

        #print(final_mean)
        return final_mean, time_stage_1 + time_stage_2

class Top_K_Filtered(FilterAlgs):

    lfilter, qfilter = True, True

    def __init__(self, params):
        self.params = params

    def top_k_extract(self, arr, k):
        """Output the top k indices of arr."""
        return np.argpartition(np.abs(arr), -k)[-k:]

    def trim_data(self, S, top_indices):
        """Set the non-top-k coordinates to 0 for every data point."""
        k = self.params.k
        S_new = S[:, top_indices]
        self.params.d = k
        return S_new

    def alg(self, S, indicator):
        """Main algorithm."""
        params, S, indicator = pre_processing(self.params, S, indicator)
        self.params = params
        start_time = time.time()
        k = self.params.k

        stage1_mean, pred_k = Top_K.GD(self, S, 200)
        S_trimmed = self.trim_data(S, pred_k)
        time_stage_1 = time.time() - start_time
        #self.params.d = pred_k
        mean, time_stage_2  =  super().alg(S_trimmed, indicator)
        #print('ATTENTION!')
        #print(mean)
        if type(mean) == int:
            mean = np.zeros(len(stage1_mean))
        #print(top_indices)
        #print(stage1_mean)
        final_mean = np.zeros(len(stage1_mean))
        #print(final_mean)
        #for i in range(len(top_indices)):

            #final_mean[top_indices[i]] = mean[i]

        j = 0
        for i in pred_k:
            final_mean[i] = mean[j]
            j = j + 1

        #print(final_mean)
        return final_mean, time_stage_1 + time_stage_2



class GD_nonsparse(object):

    def __init__(self, params):
        self.params = params

    def alg(self, S, indicator):
        start_time = time.time()
        S_tmp = matlab.double(S.tolist())
        tmp = [self.params.eps]
        tmp = matlab.double(tmp)
        estimated_mean = eng.robust_mean_pgd(S_tmp, tmp[0][0], 100)

        estimated_mean_total = np.zeros(len(estimated_mean))
        j = 0
        for i in range(len(estimated_mean)):
            estimated_mean_total[i] = estimated_mean[j][0]
            j = j + 1
        total_time = time.time() - start_time

        return estimated_mean_total, total_time


class Stage2_GD(object):

    def __init__(self, params):
        self.params = params
    
    def top_k_extract(self, arr, k):
        """Output the top k indices of arr."""
        return np.argpartition(np.abs(arr), -k)[-k:]

    def trim_data(self, S, top_indices):
        """Set the non-top-k coordinates to 0 for every data point."""
        k = self.params.k
        S_new = S[:, top_indices]
        self.params.d = k
        return S_new

    def alg(self, S, indicator):
        params, S, indicator = pre_processing(self.params, S, indicator)
        self.params = params
        start_time = time.time()
        k = self.params.k
        d = self.params.d

        #stage1_mean, pred_k = Top_K.GD(self, S, 200)
        #top_indices = self.top_k_extract(stage1_mean, pred_k)
        #S_trimmed = self.trim_data(S, top_indices)
        S_trimmed = S[:,0:k]


        S_trimmed = matlab.double(S_trimmed.tolist())
        tmp = [self.params.eps]
        tmp = matlab.double(tmp)
        estimated_mean = eng.robust_mean_pgd(S_trimmed, tmp[0][0], 100)

        estimated_mean_total = np.zeros(d)
        for i in range(k):
            estimated_mean_total[i] = estimated_mean[i][0]
        total_time = time.time() - start_time

        return estimated_mean_total, total_time


class Topk_GD(object):

    def __init__(self, params):
        self.params = params
    
    def top_k_extract(self, arr, k):
        """Output the top k indices of arr."""
        return np.argpartition(np.abs(arr), -k)[-k:]

    def trim_data(self, S, top_indices):
        """Set the non-top-k coordinates to 0 for every data point."""
        k = self.params.k
        S_new = S[:, top_indices]
        self.params.d = k
        return S_new

    def alg(self, S, indicator):
        params, S, indicator = pre_processing(self.params, S, indicator)
        self.params = params
        start_time = time.time()
        k = self.params.k

        stage1_mean, pred_k = Top_K.GD(self, S, 200)
        #top_indices = self.top_k_extract(stage1_mean, pred_k)
        #S_trimmed = self.trim_data(S, top_indices)
        S_trimmed = self.trim_data(S, pred_k)


        S_trimmed = matlab.double(S_trimmed.tolist())
        tmp = [self.params.eps]
        tmp = matlab.double(tmp)
        estimated_mean = eng.robust_mean_pgd(S_trimmed, tmp[0][0], 100)

        estimated_mean_total = np.zeros(len(stage1_mean))
        j = 0
        print(stage1_mean)
        print(estimated_mean)
        for i in pred_k:
            estimated_mean_total[i] = estimated_mean[j][0]
            j = j + 1
        print('ATTENTION')
        print(estimated_mean)
        print(pred_k)
        print(stage1_mean)
        total_time = time.time() - start_time

        return estimated_mean_total, total_time


class GDAlgs_npre(object):

    def __init__(self, params):
        self.params = params
        self.sparse = True
        pass

    def project_onto_capped_simplex_simple(self, w, cap):
        tL = np.min(w) - 1
        tR = np.max(w)

        for b_search in range(1, 50):
            t = (tL + tR)/2
            if np.sum(np.minimum(np.maximum(w-t, 0), cap)) < 1:
                tR = t
            else:
                tL = t
        v = np.minimum(np.maximum(w-t, 0), cap)
        return v

    def alg(self, S, indicator):

        self.sparse = True
        k = self.params.k
        d = self.params.d
        m = self.params.m
        eps = self.params.eps
        nItrs = 200

        step_size = 1/m
        tol = 0.01
        w = np.ones(m) / m
        X = S
        eps_m = round(eps * m)
        nabla_f_w_total_time = 0
        start_time = time.time()
        previous_obj = -1
        previous_w = np.ones(m) / m
        for i in range(nItrs):
            if self.sparse:
                Xw = X.T @ w
                #Sigma_w = np.cov(X, rowvar = 0)
                #Sigma_w_minus_I = Sigma_w - np.eye(d)
                #print('here1 ', Sigma_w_minus_I.diagonal())
                Sigma_w = (X.T @ spdiags(w, 0, m, m) @ X) - np.outer(Xw, Xw)
                Sigma_w_minus_I = Sigma_w - np.eye(d)
                #print('here2', Sigma_w_minus_I.diagonal())
                #find indices of largest k entries of each row of Sigma_w_minus_I
                largest_k_each_row_index_array = np.argpartition(Sigma_w_minus_I, kth=-k, axis=-1)[:, -k:]
                #find corresponding entries
                largest_k_each_row = np.take_along_axis(Sigma_w_minus_I, largest_k_each_row_index_array, axis=-1)
                #find squared F norm of each of these rows
                squared_F_norm_of_each_row = np.sum(largest_k_each_row * largest_k_each_row, axis=-1)
                #find indices of largest k rows
                largest_rows_index_array = np.argpartition(squared_F_norm_of_each_row, kth=-k)[-k:]
                cur_obj = np.sum(squared_F_norm_of_each_row[largest_rows_index_array])
                
                #we are done
                #print(cur_obj)
                if previous_obj != -1 and cur_obj < previous_obj and cur_obj > previous_obj - tol * previous_obj:
                    break     
                if previous_obj == -1 or cur_obj <= previous_obj:
                    step_size *= 2
                else:
                    w = previous_w
                    step_size /= 4
                    continue
                previous_obj = cur_obj
                previous_w = w
                psi_w = np.zeros((d, d), dtype=int)
                largest_k_each_row_index_array = largest_k_each_row_index_array[largest_rows_index_array]
                #psi is indicator matrix with 1s corresponding to entries included in F,k,k norm, and 0 elsewhere
                psi_w[largest_rows_index_array, largest_k_each_row_index_array.T] = 1

                psi_w = coo_matrix(psi_w)

                Z_w = psi_w.multiply(Sigma_w_minus_I)

                X_T_Z_w_X_diag = np.sum(Z_w.data * (X.T[psi_w.row, :] * X.T[psi_w.col, :]).T, axis=-1)
                nabla_f_w = (X_T_Z_w_X_diag - (X @ (Z_w @ (X.T @ w))) - (X @ (Z_w.T @ (X.T @ w)))) / sparse_norm(Z_w)
            else:
                Xw = X.T @ w
                #Sigma_w = np.cov(X, rowvar = 0)
                #Sigma_w_minus_I = Sigma_w - np.eye(d)
                #print('here1 ', Sigma_w_minus_I.diagonal())
                Sigma_w = (X.T @ spdiags(w, 0, m, m) @ X) - np.outer(Xw, Xw)
                print(np.allclose(Sigma_w, Sigma_w.T, rtol=1e-05, atol=1e-08))
                Sigma_w_minus_I = Sigma_w - np.eye(d)
                #print('here2', Sigma_w_minus_I.diagonal())
                #find indices of largest k entries of each row of Sigma_w_minus_I
                largest_k_each_row_index_array = np.argpartition(Sigma_w_minus_I, kth=-k, axis=-1)[:, -k:]
                #find corresponding entries
                largest_k_each_row = np.take_along_axis(Sigma_w_minus_I, largest_k_each_row_index_array, axis=-1)
                #find squared F norm of each of these rows
                squared_F_norm_of_each_row = np.sum(largest_k_each_row * largest_k_each_row, axis=-1)
                #find indices of largest k rows
                largest_rows_index_array = np.argpartition(squared_F_norm_of_each_row, kth=-k)[-k:]
                cur_obj = np.sum(squared_F_norm_of_each_row[largest_rows_index_array])
                print(cur_obj)

                Xw = np.matmul(X.T, w)
                #Sigma_w_fun = lambda v: np.matmul(X.T, w * np.matmul(X, v)) - Xw *np.matmul(Xw.T, v)
                #Sigma_w_linear_operator = LinearOperator((d, d), matvec=Sigma_w_fun)
                u_val, u = LA.eigh(Sigma_w)
                #print(u, u_val)
                u_val = u_val[0]
                u = u[:, 0]
                Xu = X @ u
                #print(w.T, Xu)
                print(X.shape, w.shape, Xu.shape, u.shape, (X.T @ w).shape,np.inner(u, (X.T @ w)))
                nabla_f_w = Xu * Xu - (2 * np.inner(u, (X.T @ w))) @ Xu
            w = w - step_size * nabla_f_w/ LA.norm(nabla_f_w)
            #print(w)
            w = self.project_onto_capped_simplex_simple(w, (1/(m - eps_m)))
            #print(w.shape)
        total_time = time.time() - start_time
        print('Time to run GD ', total_time)
        mu_gd = topk_abs(np.sum(w * X.T, axis=1), k)
        return mu_gd, total_time


class GDAlgs(GDAlgs_npre):

    def alg(self, S, indicator):
        params, S, indicator = pre_processing(params = self.params, S = S, indicator = indicator)
        self.params = params

        return super().alg(S = S, indicator = indicator)


class Top_K(object):

    def __init__(self, params):
        self.params = params

    def GD(self, S, iter_num):
        """Stage 1 algorithm."""
        d = self.params.d
        m = self.params.m
        eps = self.params.eps
        '''
        #group_size = self.params.group_size
        #K = m // group_size  # number of subgroups
        K = 2 * ceil(eps * m)
        #self.params.m = K
        # k = self.params.k
        X_split = np.array_split(S, K)
        X_grouped = []
        for i in X_split:
            X_grouped.append(list(np.mean(i, axis = 0)))
        #X_grouped = np.mean(X_grouped, axis=1)
        #print(X_grouped)
        X_grouped = np.array(X_grouped)
        # gradient descent
        '''
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
        if len(top_k_indices) < 2:
            top_k_indices = np.argpartition(np.abs(estimated_mean), -2)[-2:]
        self.params.k = len(top_k_indices)
        return trim_idx_abs(estimated_mean, top_k_indices), top_k_indices
        # print("estimated: ", estimated_mean)
        # top_k_indices = self.top_k_extract(estimated_mean, k)
        # return top_k_indices # output a list of k indices
        # print("topk_est: ", topk_abs(estimated_mean, k))
        # return topk_abs(estimated_mean, k)

    def alg(self, S, indicator):
        
        """Main algorithm."""
        # top_indices = self.GD(S)
        # S_new = self.trim_data(S, top_indices)
        # return S_new
        # print("GD: ", self.GD(S))
        #k = self.params.k
        params, S, indicator = pre_processing(self.params, S, indicator)
        self.params = params
        estimated_mean, pred_k = self.GD(S, 200)
        start_time = time.time()
        estimated_mean, _ = self.GD(S,600)
        total_time = time.time() - start_time

        return trim_idx_abs(estimated_mean, pred_k), total_time


class Oracle(object):

    def __init__(self, params):
        self.params = params

    def alg(self, S, indicator):
        start_time = time.time()
        MOM = [0,0,0,0,0,0]
        tm = self.params.tm()
        S_1 = np.array([S[i] for i in range(len(indicator)) if indicator[i]!=0])
        MOM[0] = topk_abs(np.mean(S_1, axis = 0), self.params.k)
        S_2 = np.array_split(S_1, 10)
        mean_2 = []
        for i in range(len(S_2)):
            mean_2.append(np.mean(S_2[i], axis = 0))
        MOM[1] = topk_abs(np.median(mean_2, axis = 0), self.params.k)
        S_4 = np.array_split(S_1, 50)
        mean_4 = []
        for i in range(len(S_4)):
            mean_4.append(np.mean(S_4[i], axis = 0))
        MOM[2] = topk_abs(np.median(mean_4, axis = 0), self.params.k)
        S_5 = np.array_split(S_1, 100)
        mean_5 = []
        for i in range(len(S_5)):
            mean_5.append(np.mean(S_5[i], axis = 0))
        MOM[3] = topk_abs(np.median(mean_5, axis = 0), self.params.k)
        S_10 = np.array_split(S_1, 150)
        mean_10 = []
        for i in range(len(S_10)):
            mean_10.append(np.mean(S_10[i], axis = 0))
        MOM[4] = topk_abs(np.median(mean_10, axis = 0), self.params.k)
        S_20 = np.array_split(S_1, 200)
        mean_20 = []
        for i in range(len(S_20)):
            mean_20.append(np.mean(S_20[i], axis = 0))
        MOM[5] = topk_abs(np.median(mean_20, axis = 0), self.params.k)
        MOM_loss = [0,0,0,0,0,0]
        for i in range(6):
            MOM_loss[i] = LA.norm(MOM[i]-tm)
        index = np.argmin(MOM_loss)
        # print(S_true)
        # print(indicator)
        return MOM[index], time.time() - start_time
        # return np.mean(S_true, axis=0)


class ransacGaussianMean(object):
    def __init__(self, params):
        self.params = params
        pass

    def alg(self, S, indicator):
        start_time = time.time()
        k = self.params.k
        d = self.params.d
        m = self.params.m
        eps = self.params.eps
        tau = self.params.tau

        T_naive = np.sqrt(2*np.log(m*d/tau))
       
        med = np.median(S, axis=0)
        S = S[np.max(np.abs(med-S), axis=1) < T_naive]

        empmean = np.mean(S, axis=0)

        ransacN = S.shape[0]//2
        print("ransacN", ransacN)
        
        if ransacN > m: 
            print("Ransac, Here")
            return topk_abs(empmean, k), time.time() - start_time
        
        numIters = 200
        # thresh = d*np.log(d) + 2*(np.sqrt(d* np.log(d) * np.log(m/tau)) + np.log(m/tau)) + (eps**2)*(np.log(1/eps))**2
        thresh = d + np.sqrt(d)
        print("thresh", thresh)
        bestMean = empmean

        # bestInliers = (S[LA.norm(S-empmean) < np.sqrt(thresh)]).shape[0]
        bestInliers = sum([(LA.norm(x-empmean) < np.sqrt(thresh)) for x in S])
        # print("bestInliers", bestInliers)
        # bestMedian = np.median(np.array([LA.norm(x - empmean) for x in S]))
        # print(len(S))
        # print("Mean inlier val:", np.mean(sum([(LA.norm(S[i]-empmean) < np.sqrt(thresh)) for i in np.arange(len(S)) if indicator[i]!=0])))
        # print("Mean outlier val:", np.mean(sum([(LA.norm(S[i]-empmean) < np.sqrt(thresh)) for i in np.arange(len(S)) if indicator[i]==0])))

        
        for i in np.arange(1, numIters, 1):
            ransacS = S[np.random.choice(S.shape[0], ransacN, replace=False)]
            ransacMean = np.mean(ransacS, axis=0)
            curInliers = sum([(LA.norm(x-ransacMean) < np.sqrt(thresh)) for x in S])
            # curInliers = (S[LA.norm(S-ransacMean) < np.sqrt(thresh)]).shape[0]
            # print(curInliers)
            print(curInliers, bestInliers)
            print(len(S))
            if curInliers > bestInliers:
                bestMean = ransacMean
                bestInliers = curInliers
            # X = S - ransacMean
            # curMedian = np.median(np.array([LA.norm(x - ransacMean) for x in S]))
            # if curMedian < bestMedian:
            #     bestMean = ransacMean
            #     bestMedian = curMedian


        # print(bestMean)
        total_time = time.time() - start_time
        return topk_abs(bestMean, k), total_time
    

class load_data(RunCollection):

    def __init__(self, model, noise_model, params, loss, keys=[]):

        self.params = params
        self.keys = keys
        self.model = model
        self.noise_model = noise_model
        self.loss = loss
        self.inp = 0
        self.Run = 0
        #self.rspca = False
        #self.unknown_norm = False

    '''
    def testcount(self, f, maxcount, samp, medianerr=False):
        count = 0
        vs = []
        running_time = []
        for i in range(maxcount):
            self.params.m = samp
            #inp, S, indicator, tm = self.model.generate(self.params)
            S, tm = self.model.generate(self.params)
            S, indicator = self.noise_model.geneater(self.params, S)
            inp = self.params
            print("samp: ", samp, "i: ", i)
            func = f(inp)
            estimated_mean, running_time_tmp = func.alg(S, indicator)
            vnew = self.loss(estimated_mean, tm)

            if self.rspca:
                vbound = 0.15
            else:
                vbound = 1.2

            if medianerr == False:
                if vnew < vbound:
                    count += 1
            else:
                vs.append(vnew)
            
            running_time.append(running_time_tmp)

        if medianerr == True:
            ans = np.median(vs)
        else:
            ans = count
        ans_time = np.mean(running_time)

        return ans, ans_time

    def get_maxm(self, f, minsamp=10, sampstep=100, medianerr=False):
        samp = minsamp

        if medianerr == True:

            bound = 0.3
        else:
            bound = 7

        while True:

            count, _ = self.testcount(f, 10, samp, medianerr)
            print("Maxm count: ", count)
            if (medianerr and count < bound) or (not (medianerr) and count > bound):
                break
            samp += sampstep

        return samp

    def search_m(self, f, bounds, medianerr=False):
        minm, maxm = bounds
        samp = (minm + maxm)//2
        print(samp)

        if medianerr == True:
            bound = 0.3
        else:
            bound = 7

        for i in range(25):
            count, _ = self.testcount(f, 10, samp, medianerr)
            print("Search count: ", count)
            if (medianerr and count < bound) or (not (medianerr) and count > bound):
                maxm = samp
                samp = (minm + maxm)//2
            else:
                minm = samp
                samp = (minm + maxm)//2

        return samp, _
    '''

    def get_dataxy(self, xvar_name, xs=[]):

        results = {}
        '''
        if y_is_m == True:
            l, s = mrange
            samp = l
        
        if explicit_xs == False:
            xs = np.arange(*bounds)
        else:
            xs = xs
        '''

        for xvar in xs:
            if xvar_name == 'm':
                self.params.m = xvar
            elif xvar_name == 'k':
                self.params.k = xvar
                self.params.mu = np.ones(xvar) * 2
                '''
                self.params.tv = np.append(np.ones(xvar), np.zeros(
                    self.params.d-xvar))/np.sqrt(xvar)
                self.params.fv = np.append(
                    np.zeros(self.params.d-xvar), np.ones(xvar))/np.sqrt(xvar)
                '''

            elif xvar_name == 'd':
                self.params.d = xvar
            elif xvar_name == 'eps':
                self.params.eps = xvar
                '''J = self.params.J
                self.params.m = int(J/xvar)'''

            elif xvar_name == 'param':
                self.params.param = xvar

            elif xvar_name == 'var':
                self.params.var = xvar

            elif xvar_name == 'group_size':
                self.params.group_size = xvar

            elif xvar_name == 'm_k':
                self.params.k = xvar
                self.params.m = xvar * 80
                self.params.mu = np.ones(xvar) * 2

            elif xvar_name == 'test':
                self.params.k = xvar
                self.params.mu = np.ones(xvar) * 2
                if xvar_name == 5:
                    self.params.m = 400
                if xvar_name == 25:
                    self.params.m = 10000


            #if y_is_m == False:
                #inp, S, indicator, tm = self.model.generate(self.params)
            S, tm = self.model.generate(self.params)
            S, indicator = self.noise_model.generate(self.params, S)

            for f in self.keys:
                inp_copy = copy.copy(self.params)
                S_copy = copy.deepcopy(S)
                indicator_copy = copy.deepcopy(indicator)

                func = f(inp_copy)
                #O = Oracle(inp_copy)

                '''
                if self.unknown_norm == True:
                    f.unknown_norm = True
                '''

                if xvar_name == 'biter':
                    f.biter = xvar+1

                '''
                if relative == True:
                    estimated_mean, running_time = func.alg(S_copy, indicator_copy)
                    estimated_mean_oracle, running_time_oracle = func.alg(S_copy, indicator_copy)
                    results.setdefault(f.__name__, []).append(self.loss(estimated_mean, tm)/self.loss(estimated_mean_oracle, tm))
                    results.setdefault(f.__name__ + '_time', []).append(running_time - running_time_oracle)
                else:
                '''
                estimated_mean, running_time = func.alg(S_copy, indicator_copy)
                results.setdefault(f.__name__, []).append(
                    self.loss(estimated_mean, tm))
                results.setdefault(f.__name__ + '_time', []).append(running_time)

            '''
            else:
                for f in self.keys:
                    minsamp, sampstep = 2, 100
                    print("xvar: ", xvar)
                    maxm = self.get_maxm(f, minsamp, sampstep, medianerr)
                    samp, _ = self.search_m(f, (minsamp, maxm), medianerr)
                    results.setdefault(f.__name__, []).append(samp)
                    results.setdefault(f.__name__ + '_time', []).append(_)
            '''
        return results

    def setdata_tofile(self, filename, xvar_name, trials, xs=[]):
        start_time = time.perf_counter()
        self.setdata(xvar_name, trials, xs)
        with open(filename, 'wb') as g:
            pickle.dump(self.Run.runs, g, -1)
        
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print("runtime:", runtime)

    def setdata(self, xvar_name, trials, xs=[]):

        Runs_l_samples = RunCollection(
            self.get_dataxy, (xvar_name, xs))
        Runs_l_samples.run(trials)
        self.Run = Runs_l_samples

    def get_dataxy_heatmap(self, xs, ys):

        results = {}
        f = self.keys

        for yvar in ys[::-1]:
            heat = []
            for xvar in xs:
                self.params.m = xvar
                self.params.k = yvar
                self.params.mu = np.ones(yvar) * 2

                S, tm = self.model.generate(self.params)
                S, indicator = self.noise_model.generate(self.params, S)

                inp_copy = copy.copy(self.params)
                S_copy = S.copy()
                indicator_copy = indicator.copy()

                func = f(inp_copy)

                estimated_mean, running_time = func.alg(S_copy, indicator_copy)
                heat.append(self.loss(estimated_mean, tm))
            results.setdefault(f.__name__, []).append(heat)
        # print("heatmap results:", results)
        return results
                

    def setdata_tofile_heatmap(self, filename, trials, xs = [], ys = []):
        start_time = time.perf_counter()
        self.setdata_heatmap(trials, xs, ys)
        with open(filename, 'wb') as g:
            pickle.dump(self.Run.runs, g, -1)

        end_time = time.perf_counter()
        runtime = end_time - start_time
        print("runtime:", runtime)

    def setdata_heatmap(self, trials, xs=[], ys=[]):

        Runs_l_samples = RunCollection(
            self.get_dataxy_heatmap, (xs, ys)
        )
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
        #self.rspca = False

    def readdata(self, filename):
        with open(filename, 'rb') as g:
            ans = pickle.load(g)
        return ans

    def plot_xloss(self, outputfilename, runs, title, xlabel, ylabel, xs=[], fsize=10, fpad=10, figsize=(1, 1), fontname='Arial', yscale = 'linear'):

        cols = {'RME_sp': 'b', 'RME_sp_L': 'g', 'RME': 'r', 'ransacGaussianMean': 'y',
                'NP_sp': 'k', 'Oracle': 'b', 'Top_K': 'g', 'Top_K_Filtered': 'palevioletred', 'GDAlgs':'sandybrown', 'Topk_GD':'tomato',
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
                  'Top_K_Filtered': 'Top_K + Filter_sp_LQ',
                  'GDAlgs': 'Sparse GD',
                  'Topk_GD': 'Full',
                  'NP_sp_npre': 'NP_sp_npre',
                  'RME_sp_npre': 'Filter_sp_LQ_npre', 
                  'RME_sp_L_npre': 'Filter_sp_L_npre', 
                  'RME_npre': 'Filter_nsp_npre', 
                  'GDAlgs_npre': 'Sparse GD_npre',
                  'GD_nonsparse': 'GD_nonsparse',
                  'Stage2_GD': 'Stage2_GD',
                  'Stage2_filter': 'Stage2_filter'
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
        plt.xlim(5,100)
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
        plt.figure()

    def plot_xtime(self, outputfilename, runs, title, xlabel, ylabel, xs=[], fsize=10, fpad=10, figsize=(1, 1), fontname='Arial', yscale = 'linear'):

        cols = {'RME_sp_time': 'b', 'RME_sp_L_time': 'g', 'RME_time': 'r', 'ransacGaussianMean_time': 'y',
                'NP_sp_time': 'k', 'Oracle_time': 'c', 'Top_K_time': 'darkseagreen', 'Top_K_Filtered_time': 'palevioletred', 'GDAlgs_time':'sandybrown', 'Topk_GD_time':'m',
                'NP_sp_npre_time': 'gray', 'RME_sp_npre_time': 'skyblue', 'RME_sp_L_npre_time': 'springgreen', 'RME_npre_time': 'tomato', 'GDAlgs_npre_time': 'peachpuff', 'GD_nonsparse_time': 'plum'
                }

        markers = {'RME_sp_time': 'o',
                   'RME_sp_L_time': 'v',
                   'RME_time': '^',
                   'ransacGaussianMean_time': 'D',
                   'NP_sp_time': 'p',
                   'Oracle_time': 'x',
                   'Top_K_time': '.',
                   'GDAlgs_time':'^',
                   'Top_K_Filtered_time': 'o',
                   'Topk_GD_time':'*',
                   'NP_sp_npre_time': 'p',
                   'RME_sp_npre_time': 'o', 
                   'RME_sp_L_npre_time': 'v', 
                   'RME_npre_time': '^', 
                   'GDAlgs_npre_time': '^',
                   'GD_nonsparse_time': '*'
                   }

        labels = {'NP_sp_time': 'NP_sp',
                  'ransacGaussianMean_time': 'RANSAC',
                  'RME_sp_time': 'Filter_sp_LQ',
                  'RME_sp_L_time': 'Filter_sp_L',
                  'Oracle_time': 'Oracle',
                  'RME_time': 'Filter_nsp',
                  'Top_K_time': 'Top_K',
                  'Top_K_Filtered_time': 'Top_K + Filter_sp_LQ',
                  'GDAlgs_time': 'Sparse GD',
                  'Topk_GD_time': 'Top_K + Sparse GD',
                  'NP_sp_npre_time': 'NP_sp_npre',
                  'RME_sp_npre_time': 'Filter_sp_LQ_npre', 
                  'RME_sp_L_npre_time': 'Filter_sp_L_npre', 
                  'RME_npre_time': 'Filter_nsp_npre', 
                  'GDAlgs_npre_time': 'Sparse GD_npre',
                  'GD_nonsparse_time': 'GD_nonsparse'
                  }

        s = len(runs)
        #print(runs)
        #str_keys = [key.__name__ for key in self.keys]
        #print(str_keys)
        str_keys_time = [key.__name__ + '_time' for key in self.keys]
        #print(str_keys_time)

        for key in str_keys_time:
            print(key)
            A = np.array([res[key] for res in runs])
            print(A)
            '''
            if explicit_xs == False:
                xs = np.arange(*bounds)
            else:
                xs = xs
            '''
            mins = [np.sort(x)[int(s*0.25)] for x in A.T]
            maxs = [np.sort(x)[int(s*0.75)] for x in A.T]

            plt.fill_between(xs, mins, maxs, color=cols[key], alpha=0.2)
            plt.plot(xs, np.median(A, axis=0),
                     label=labels[key], color=cols[key], marker=markers[key])

        #p = copy.copy(self.params)

        rcParams['figure.figsize'] = figsize

        rc('font', size=fsize)
        rc('axes', labelsize='large')
        rc('legend', numpoints=1)

        plt.title(title, pad=fpad, fontsize=fsize)
        plt.xlabel(xlabel, fontsize=fsize, labelpad=fpad)
        plt.ylabel(ylabel, labelpad=fpad, fontsize=fsize)
        plt.legend()
        plt.yscale(yscale)
        #plt.ylim(*ylims)
        plt.savefig(outputfilename, bbox_inches='tight')
        plt.tight_layout()

    def plot_xtime_fromfile(self, outputfilename, filename, title, xlabel, ylabel, xs=[], fsize=10, fpad=10, figsize=(1, 1), fontname='Arial', yscale='linear'):
        Run = self.readdata(filename)
        self.plot_xtime(outputfilename, Run, title, xlabel, ylabel,
                        xs, fsize, fpad, figsize, fontname, yscale)

    def plotxy_fromfile_time(self, outputfilename, filename, title, xlabel, ylabel, figsize=(1, 1), fsize=10, fpad=10, xs=[], fontname='Arial', yscale='linear'):

        self.plot_xtime_fromfile(outputfilename, filename, title, xlabel, ylabel, figsize=figsize,
                                 fsize=fsize, fpad=fpad, xs=xs, fontname=fontname, yscale=yscale)
        plt.figure()

    def plot_heatmap(self, outputfilename, runs, title, xlabel, ylabel, xs=[], ys = [],fsize=10, fpad=10, figsize=(1,1), fontname='Arial', yscale = 'linear'):

        f = self.keys
        key = f.__name__

        A = np.array([res[key] for res in runs])
        A = np.median(A, axis=0)
        print("heatmap input:", A)

        fig, ax = plt.subplots()
        im = ax.imshow(A, cmap=mpl.colormaps['YlGn'])

        ax.set_xticks(np.arange(len(xs)), labels=xs)
        ax.set_yticks(np.arange(len(ys)), labels=ys[::-1])
        ax.set_title(title)
        for i in range(len(xs)):
            for j in range(len(ys)):
                text = ax.text(i, j, round(A[j, i],2),
                       ha="center", va="center", color="black")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("loss", rotation=-90, va="bottom")

        plt.xlabel(xlabel, fontsize=fsize, labelpad=fpad)
        plt.ylabel(ylabel, labelpad=fpad, fontsize=fsize)
        plt.savefig(outputfilename, bbox_inches='tight')


    def plot_heatmap_fromfile(self, outputfilename, filename, title, xlabel, ylabel, xs=[], ys=[], fsize=10, fpad=10, figsize=(1, 1), fontname='Arial', yscale = 'linear'):
        Run = self.readdata(filename)
        self.plot_heatmap(outputfilename, Run, title, xlabel, ylabel,
                        xs, ys,fsize, fpad, figsize, fontname, yscale)

    def plotheatmap_fromfile(self, outputfilename, filename, title, xlabel, ylabel, figsize=(1, 1), fsize=10, fpad=10, xs=[], ys=[], fontname='Arial', yscale='linear'):

        self.plot_heatmap_fromfile(outputfilename, filename, title, xlabel, ylabel, xs=xs, ys=ys, figsize=figsize,
                                 fsize=fsize, fpad=fpad, fontname=fontname, yscale=yscale)
        plt.show()

""" P(x) for quadratic filter """


def p(X, mu, M):

    F = LA.norm(M)
    D = X - mu
    vec = (D.dot(M) * D).sum(axis=1)
    return (vec - np.trace(M))/F


""" Thing that thresholds to the largest k entries indicaors """


def indicat(M, k): 
    
    """
    creates an indicator matrix for 
    the largest k diagonal entries and 
    largest k**2 - k off-diagonal entries
    """
    
    ans = np.zeros(M.shape)

    u = np.argpartition(M.diagonal(), -k)[-k:] # Finds largest k indices of the diagonal 
    ans[(u,u)] = 1

    idx = np.where(~np.eye(M.shape[0],dtype=bool)) # Change this too
    val = np.partition(M[idx].flatten(), -k**2+k)[-k**2+k] # (k**2 - k)th largest off-diagonl element
    idx2 = np.where(M > val)
    
    ans[idx2] = 1
    
    return (ans, u)


""" Threshold to top-k in absolute value """


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