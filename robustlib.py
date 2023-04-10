import numpy as np
import scipy
import copy
from scipy import special
from numpy import linalg as LA
import matplotlib.pyplot as plt
from pylab import rcParams
import pickle
from matplotlib import rc
import ast
import mpld3
mpld3.enable_notebook()

def err_rspca(a,b): return LA.norm(np.outer(a,a)-np.outer(b, b))

def err(a,b): return LA.norm(a-b)

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

""" Parameter object for noise models """

class Params(object):
    def __init__(self, d = 0, m = 0, eps = 0, k = 0, tau = 0.2, mass = 0, tv = 0, fv = 0):
        self.d = d
        self.m = m
        self.eps = eps
        self.k = k
        self.tau = tau
        self.mass = mass
        self.tv = tv
        self.fv = fv


""" Noise models """

class DenseNoiseModel(object):
    def __init__(self, dist):
        self.dist = dist

    def generate(self, params):
        d, k, eps, m, tau = params.d, params.k, params.eps, params.m, params.tau

        tm = np.append(np.ones(k), np.zeros(d-k))

        G = np.random.randn(m, d) + tm

        S = G.copy()

        L = int(m*(1-eps))
        S[L:] += self.dist

        indicator = np.ones(len(S))
        indicator[L:] = 0
        params = Params(d,m,eps,k,tau)
        return params, S, indicator, tm


class MixtureMean(object):
    def __init__(self):
        pass

    def generate(self, params):
        d, k, eps, m, tau = params.d, params.k, params.eps, params.m, params.tau

        # tm = np.append(np.ones(k), np.zeros(d-k))
        tm = np.zeros(d)
        fm = np.append(np.zeros(d-k), np.ones(k))

        cov = 2*np.identity(d) - np.diag(fm)

        G = np.random.randn(m, d) + tm
        GF = (1/2)*np.random.randn(m, d) + (1/np.sqrt(2))

        S = G.copy()

        L = int(m*(1-eps))

        S[L:] = GF[L:]

        indicator = np.ones(len(S))
        indicator[L:] = 0
        params = Params(d,m,eps,k,tau)
        return params, S, indicator, tm


class BimodalModel(object):
    def __init__(self):
        pass
    
    def generate(self, params):
        d, k, eps, m, tau = params.d, params.k, params.eps, params.m, params.tau

        tm = np.zeros(d)
        fm = np.append(np.zeros(d-k), np.ones(k))

        cov = 2*np.identity(d) - np.diag(fm)

        G = np.random.randn(m, d) + tm
        G2 = np.random.multivariate_normal(np.zeros(d), cov, (m,))
        G1 = np.random.randn(m, d) + fm

        S = G.copy()

        L = int(m*(1-eps))
        M = int((L + m)/2)

        S[L:M] = G1[L:M]
        S[M:] = G2[M:]

        indicator = np.ones(len(S))
        indicator[L:] = 0
        params = Params(d,m,eps,k,tau)
        return params, S, indicator, tm

class TailFlipModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        eps, m, d, k = params.eps, params.m, params.d, params.k

        tm = 0
        G = np.random.randn(m, d)
        S = G.copy()

        v = np.random.randn(k)
        v = np.append(v, np.zeros(d-k))
        np.random.shuffle(v)

        Sdots = S.dot(v)
        idx = np.argsort(Sdots)
        S = S[idx]

        S[:int(eps*m)] = -S[:int(eps*m)]
        
        indicator = np.ones(m)
        indicator[:int(eps*m)] = 0

        return params, S, indicator, tm

class RSPCA_MixtureModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        d, eps, m, tau, mass, k = params.d, params.eps, params.m, params.tau, params.mass, params.k
        tv, fv = params.tv, params.fv
        # tm = np.append(np.ones(k), np.zeros(d-k))

        tcov = np.identity(d) + np.outer(tv, tv)
        fcov = np.identity(d) + mass*np.outer(fv, fv)

        G1 = np.random.multivariate_normal(np.zeros(d), tcov, (m, ))
        G2 = np.random.multivariate_normal(np.zeros(d), fcov, (m, ))
        S = G1.copy()

        L = int(m*(1-eps))
        S[L:] = G2[L:]

        indicator = np.ones(len(S))
        indicator[L:] = 0
        params = Params(d,m,eps,k,tau, mass = mass)
        return params, S, indicator, tv

        
""" Algorithms """

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
    
    def drop_points(self, S, indicator, x, tail, plot = False, f = 0):
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
        if T > 0.6*l : T = 0

        idx = np.nonzero((p_x >= sorted_p[T]))

        if len(S)==len(idx[0]):
            tfdr = 0
        else:
            tfdr = (sum(indicator) - sum(indicator[idx[0]]))/(len(S)-len(idx[0]))

        if plot==True:
            plt.plot(np.arange(l), sorted_p)
            plt.plot(T*np.ones(100), 0.01*np.arange(100), linewidth=3)
            plt.plot(np.arange(l), indicator[sorted_idx], linestyle='-.', linewidth=3)
            plt.plot([0,len(S)],[0,fdr], '--')
            plt.title("sample size {}, T = {}, True FDR = {}, tail = {}".format(l, T, tfdr, tail.__name__))
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
            S_u = S[np.ix_(np.arange(l),u)]
            dots = S_u.dot(v)
            m2 = np.median(dots)

            if self.dense_filter == False:
                x = np.abs(dots - m2) - 3*np.sqrt(ev*eps)
                idx = self.drop_points(S, indicator, x, self.tail_m,  self.do_plot_linear, self.figure_no)  
            else:
                x = np.abs(dots - m2)
                idx = self.drop_points(S, indicator, x, self.tail_t,  self.do_plot_linear, self.figure_no)  
            
            if self.verbose:
                bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
                print(f"Filtered out {l - len(idx[0])}/{l}, {bad_filtered} false ({bad_filtered / (l - len(idx[0])):0.2f} vs {self.fdr})")
            return idx
        else:
            return (np.arange(len(S)),)
    
    def quadratic_filter(self, S, indicator, M_mask):

        print("Quadratic filter...")
        l = len(indicator)
        mu_e = np.mean(S, axis = 0)
        x = np.abs(p(S, mu_e, M_mask))

        idx = self.drop_points(S, indicator, x, self.tail_c, self.do_plot_quadratic, self.figure_no)
        
        if self.verbose:
            bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
            print(f"Filtered out {l - len(idx[0])}/{l}, {bad_filtered} false ({bad_filtered / (l - len(idx[0])):0.2f} vs {self.fdr})")
            return idx
        else:
            return (np.arange(len(S)),)        
    
    def update_params(self, S, indicator, idx):
        S, indicator = S[idx], indicator[idx]
        self.params.m = len(S)
        return S, indicator
    
    def alg(self, S, indicator):
        
        k = self.params.k
        d = self.params.d
        m = self.params.m
        eps = self.params.eps
        tau = self.params.tau
        
        T_naive = np.sqrt(2*np.log(m*d/tau))
        med = np.median(S, axis=0)
        idx = (np.max(np.abs(med-S), axis=1) < T_naive)
        S, indicator =  self.update_params(S, indicator, idx)

        if len(idx) < self.params.m: print("NP pruned {self.params.m - len(idx) f} points")
        
        while True:
            if self.lfilter == False and self.qfilter == False:
                break        

            if len(S)==0: 
                print("No points remaining.")
                return 0

            if len(S)==1: 
                print("1 point remaining.")
                return 0

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
                    cov_u = cov_e[np.ix_(u,u)]
                    ev, v = scipy.linalg.eigh(cov_u, eigvals=(k-1,k-1))
                    v = v.reshape(len(v),)
                else:
                    ev, v = scipy.linalg.eigh(cov_e, eigvals=(d-1,d-1))
                    if ev <  1+eps*np.log(1/eps):
                        print("RME exited properly")
                        break
                    v = v.reshape(len(v),)
                    u = np.arange(d)

                x = self.params.m
                idx = self.linear_filter(S, indicator, ev, v, u)[0]
                # print(idx)
                self.figure_no += 1
                S, indicator =  self.update_params(S, indicator, idx)
                if len(idx) < x: continue

            if self.qfilter == True:

                x = self.params.m
                idx = self.quadratic_filter(S, indicator, M_mask)[0]
                # print(idx)
                self.figure_no += 1
                S, indicator =  self.update_params(S, indicator, idx)
                print("condition", len(idx), x)
                if len(idx) < x: continue

            if pre_filter_length == len(idx): 
                print("Could not filter")
                break
                    
        if self.is_sparse == True:
            # print(topk_abs(np.mean(S, axis=0), k))
            return topk_abs(np.mean(S, axis=0), k)
        else:
            return np.mean(S, axis=0)
            
class NP_sp(FilterAlgs):
    lfilter, qfilter = False, False

class RME_sp(FilterAlgs):
    lfilter, qfilter = True, True

class RME_sp_L(FilterAlgs):
    lfilter, qfilter = True, False

class RME(FilterAlgs):
    
    lfilter, qfilter = True, False
    dense_filter = True
    # do_plot_linear = True

class RSPCAb(FilterAlgs):

    do_plot_rspca = False
    biter = 2
    unknown_norm = False

    def __init__(self, params):
        self.params = params

    def tail_rspca(self, T):
        eps = self.params.eps
        # T-=3
        idx = np.nonzero((T < 10*np.log(1/eps)))
        v = (10/((T*np.log(T))**2))
        v = (eps)*v
        idx = np.isnan(v)
        v[idx] = 1
        v[idx] = 1
        return v

    def flatcov(self, S):
        d, m = self.params.d, self.params.m

        Id = np.identity(d)
        S_cov = S[:,:,np.newaxis]  * S[:,np.newaxis,:] - Id
        S_cov = S_cov.reshape((m, d**2))
        return S_cov

    def get_largest_indices(self, S_cov):
        d, k = self.params.d, self.params.k

        mu = np.mean(S_cov, axis=0)
        u = np.argpartition(mu, -k**2)[-k**2:]

        Mask = np.zeros(d**2)
        Mask[u] = 1
        Mask = Mask.reshape(d,d)
        indices = np.nonzero(Mask)

        return u, indices 

    def guess_restricted_cov(self, S_cov, u, indices, v_prev):
        d, k = self.params.d, self.params.k

        Q = np.dstack(indices)[0]
        Q2 = np.array([np.tile(Q, (len(Q),1)), np.repeat(Q, len(Q), 0)]).transpose([1,0,2])
        
        Cov = np.outer(v_prev, v_prev)

        T1 = (Cov[Q2[:,0,0], Q2[:,1,1]] * Cov[Q2[:,0,1], Q2[:,1,0]]).reshape(k**2, k**2)
        vecCov = (np.identity(d) + np.outer(v_prev,v_prev)).reshape((d**2,))[u]
        T2 = np.outer(vecCov, vecCov)

        return T1 + T2

    def vec2mat_restricted_eigv(self, vec, u):
        d, k = self.params.d, self.params.k

        z = np.zeros(d**2)
        z[u] = vec

        z = z.reshape(d,d)

        # M = np.zeros((d,d))
        # M[np.nonzero(z)] = 1
        # print(M)

        ev, ans = scipy.linalg.eigh(z, eigvals=(d-1,d-1))

        if self.unknown_norm:
            ans = ev*ans.reshape(d,)
        else:
            ans = ans.reshape(d,)
        # ans = topk_abs(ans, k)

        return ans

    def boot_iteration(self, S, indicator, v_prev, thresh):
        k = self.params.k
        d = self.params.d
        eps = self.params.eps
        tau = self.params.tau

        pre_filter_length = self.params.m
        T_naive = np.sqrt(2*np.log(self.params.m*d/tau))
        idx = (np.max(np.abs(S), axis=1) < T_naive)
        # if self.verbose:
        #         bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
        #         print(f"Filtered out {l - len(idx)}/{l}, {bad_filtered} false ({bad_filtered / (l - len(idx)):0.2f} vs {self.fdr})")
        S, indicator =  self.update_params(S, indicator, idx)

        if pre_filter_length > len(S) and self.verbose == True: 
            print("NP filtered something!..")

        while True: 
            # print("S shape..", S.shape)
            S_cov = self.flatcov(S)
            u, indices = self.get_largest_indices(S_cov)
            
            S_cov_r = S_cov[np.ix_(np.arange(self.params.m), u)]
            guess_cov_r = self.guess_restricted_cov(S_cov, u, indices, v_prev)
            # print("S_cov_r", S_cov_r)
            cov_r = np.cov(S_cov_r, rowvar=0)    
            M = cov_r - guess_cov_r
            # print("cov_r", cov_r)
            # print("guess_cov_r", guess_cov_r)
            # print("M", M)

            ev, v = scipy.linalg.eigh(M, eigvals=(k**2-1,k**2-1))
            v = v.reshape(len(v),)

            if ev < thresh:
                if self.verbose == True:
                    print("Valid exit.")
                break
            else:
                if self.verbose == True:
                    print("RSPCAb filtering...")

                l = pre_filter_length = self.params.m
                
                dots = S_cov_r.dot(v)
                med = np.median(dots)
                x = np.abs(dots - med)

                idx = self.drop_points(S, indicator, x, self.tail_rspca, self.do_plot_rspca, self.figure_no)[0]
                self.figure_no += 1

                if self.verbose:
                    bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
                    print(f"Filtered out {l - len(idx)}/{l}, {bad_filtered} false ({bad_filtered / (l - len(idx)):0.2f} vs {self.fdr})")

                S, indicator = self.update_params(S, indicator, idx)

                if pre_filter_length > len(idx):
                    continue
                else:
                    break
        return self.vec2mat_restricted_eigv(np.mean(S_cov_r, axis=0), u)

    def alg(self, S, indicator, tv = 0):
        biter, eps = self.biter, self.params.eps
        S_copy, indicator_copy = S.copy(), indicator.copy()
        pcopy = copy.copy(self.params)

        eps_prev = 10
        v_prev = np.zeros(self.params.d)

        for i in range(biter):
            # print("Shape in biter", S.shape)
            self.params = pcopy
            S = S_copy
            indicator = indicator_copy
            v_prev = self.boot_iteration(S, indicator, v_prev, eps_prev)
            eps_prev = (eps_prev*eps)**(1/2) + eps*np.log(1/eps)
            if self.verbose:
                print(f"loss after {i+1} iteration!", LA.norm(np.outer(v_prev, v_prev) - np.outer(tv,tv)))
        return v_prev


class RSPCAdense(FilterAlgs):
    is_sparse = True
    do_plot_rspca = False

    def __init__(self, params):
        self.params = params

    def tail_rspcaDense(self, T): 
        eps, k, m, d, tau = self.params.eps, self.params.k, self.params.m, self.params.d, self.params.tau 
        
        return (special.erfc(T)) #+ (eps**2)/(np.log(d*np.log(m*d/tau))*T**2))

    def alg(self, S, indicator):
        k = self.params.k
        d = self.params.d
        eps = self.params.eps
        tau = self.params.tau

        pre_filter_length = self.params.m
        T_naive = np.sqrt(2*np.log(self.params.m*d/tau))
        idx = (np.max(np.abs(S), axis=1) < T_naive)
        S, indicator =  self.update_params(S, indicator, idx)

        if pre_filter_length > len(S) and self.verbose == True: 
            print("NP filtered something!..")

        while True:
            if len(S) <=1:
                break
            Cov = np.cov(S, rowvar = 0)
            print(Cov.shape)
            ev, v = scipy.linalg.eigh(Cov, eigvals=(d-1,d-1))
            v = v.reshape(len(v),)
            dots = S.dot(v)
            dsort = np.sort(dots)
            if ev > dsort[int(0.25*len(dots))] and ev < dsort[int(0.75*len(dots))]:
                break
            else:
                self.stddev =  (1/1.25)*np.abs(dsort[int(0.75*len(dots))]-dsort[int(0.25*len(dots))])
                print("Stddev", self.stddev)

                l = pre_filter_length = self.params.m
                
                dots = dots.reshape(len(dots),)
                med = np.median(dots)
                x = np.abs(dots - med)/self.stddev

                # print(x)

                idx = self.drop_points(S, indicator, x, self.tail_rspcaDense, self.do_plot_rspca, self.figure_no)[0]
                self.figure_no += 1

                if self.verbose:
                    bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
                    print(f"Filtered out {l - len(idx)}/{l}, {bad_filtered} false ({bad_filtered / (l - len(idx)):0.2f} vs {self.fdr})")

                S, indicator = self.update_params(S, indicator, idx)

                if pre_filter_length > len(idx):
                    continue
                else:
                    break

        Cov = np.cov(S, rowvar=0)
        _, v = scipy.linalg.eigh(Cov, eigvals=(d-1,d-1))
        v = v.reshape(d,)

        if self.is_sparse:
            return topk_abs(v, k)
        else:
            return v



class Oracle(object):

    def __init__(self, params):
        self.params = params

    def alg(self, S, indicator):
        S_true = np.array([S[i] for i in range(len(indicator)) if indicator[i]!=0])
        # print(S_true)
        # print(indicator)
        return topk_abs(np.mean(S_true, axis=0), self.params.k)
        # return np.mean(S_true, axis=0)



class ransacGaussianMean(object):
    def __init__(self, params):
        self.params = params
        pass

    def alg(self, S, indicator):
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
            return topk_abs(empmean, k)
        
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
        return topk_abs(bestMean, k)


class load_data(RunCollection):
        
    def __init__(self, model, params, loss, keys = []):

        self.params = params
        self.keys = keys
        self.model = model
        self.loss = loss
        self.inp = 0
        self.Run = 0
        self.rspca = False
        self.unknown_norm = False

    def testcount(self, f, maxcount, samp, medianerr = False):
        count = 0
        vs = []
        for i in range(maxcount):
            self.params.m = samp
            inp, S, indicator, tm = self.model.generate(self.params)
            print("samp: ", samp, "i: ", i)
            func = f(inp)
            vnew = self.loss(func.alg(S, indicator), tm)

            if self.rspca: 
                vbound = 0.15
            else:
                vbound = 1.2

            if medianerr == False:
                if vnew < vbound:
                    count += 1
            else:
                vs.append(vnew)

        if medianerr == True:
            ans = np.median(vs)
        else:
            ans = count
        return ans

    def get_maxm(self, f, minsamp=2, sampstep=100, medianerr = False):
        samp = minsamp

        if medianerr == True:
            
            bound = 0.3
        else:
            bound = 7

        while True:
            
            count = self.testcount(f, 10, samp, medianerr)
            print("Maxm count: ", count)
            if (medianerr and count < bound) or (not(medianerr) and count > bound):
                break
            samp+=sampstep 

        return samp

    def search_m(self,f, bounds, medianerr = False):
        minm, maxm = bounds
        samp = (minm + maxm)//2
        print(samp)

        if medianerr == True: 
            bound = 0.3
        else:
            bound = 7

        for i in range(25):
            count = self.testcount(f, 10, samp, medianerr)
            print("Search count: ", count)
            if (medianerr and count < bound) or (not(medianerr) and count > bound): 
                maxm = samp
                samp = (minm + maxm)//2
            else: 
                minm = samp
                samp = (minm + maxm)//2

        return samp


    def get_dataxy(self, xvar_name, bounds, y_is_m = False, mrange = 0, relative = False,  explicit_xs = False, xs = [], medianerr = False):

        results = {}
        if y_is_m == True:
            l, s = mrange
            samp = l

        if explicit_xs == False:
            xs = np.arange(*bounds)
        else:
            xs = xs

        for xvar in xs:
            if xvar_name == 'm':
                self.params.m = xvar
            elif xvar_name == 'k':
                self.params.k = xvar
                self.params.tv = np.append(np.ones(xvar), np.zeros(self.params.d-xvar))/np.sqrt(xvar)
                self.params.fv = np.append(np.zeros(self.params.d-xvar), np.ones(xvar))/np.sqrt(xvar)

            elif xvar_name == 'd':
                self.params.d = xvar
            elif xvar_name == 'eps':
                self.params.eps = xvar

            if y_is_m == False:
                inp, S, indicator, tm = self.model.generate(self.params)                
                for f in self.keys:
                    inp_copy = copy.copy(inp)
                    S_copy = S.copy()
                    indicator_copy = indicator.copy()

                    func = f(inp_copy)
                    O = Oracle(inp_copy)

                    if self.unknown_norm == True:
                        f.unknown_norm = True

                    if xvar_name == 'biter':
                        f.biter = xvar+1

                    if relative == True:
                        results.setdefault(f.__name__, []).append(self.loss(func.alg(S_copy, indicator_copy), tm)/self.loss(O.alg(S_copy, indicator_copy), tm))
                    else:
                        results.setdefault(f.__name__, []).append(self.loss(func.alg(S_copy, indicator_copy), tm))

            else:
                for f in self.keys:
                    minsamp, sampstep = 2,100
                    print("xvar: ",xvar)
                    maxm = self.get_maxm(f, minsamp, sampstep, medianerr)
                    samp = self.search_m(f, (minsamp,maxm), medianerr)
                    results.setdefault(f.__name__, []).append(samp)               
        return results


    def setdata_tofile(self, filename, xvar_name, bounds, trials, ylims, y_is_m = False, mrange = [], relative=False,  explicit_xs = False, xs = [], medianerr = False):
        self.setdata(xvar_name, bounds, trials, ylims, y_is_m, mrange, relative,  explicit_xs, xs, medianerr)
        with open(filename, 'wb') as g:
            pickle.dump(self.Run.runs, g, -1)
            
    def setdata(self, xvar_name, bounds, trials, ylims, y_is_m = False, mrange = [], relative=False,  explicit_xs = False, xs = [], medianerr = False):

        Runs_l_samples = RunCollection(self.get_dataxy, (xvar_name, bounds, y_is_m, mrange, relative,  explicit_xs, xs, medianerr))
        Runs_l_samples.run(trials)
        self.Run = Runs_l_samples



class plot_data(RunCollection):

    def __init__(self, model, params, loss, keys = []):

        self.params = params
        self.keys = keys
        self.model = model
        self.loss = loss
        self.inp = 0
        self.Run = 0
        self.rspca = False
        self.unknown_norm = False


    def readdata(self, filename):
        with open(filename, 'rb') as g:
            ans = pickle.load(g)
        return ans 

    def plot_xloss(self, outputfilename, runs, xvar_name, bounds, title, xlabel, ylabel, ylims, y_is_m = False, relative = False, explicit_xs = False, xs = [], fsize = 10, fpad = 10, figsize = (1,1), fontname = 'Arial'):

        cols = {'RSPCAdense':'k', 'RSPCAb':'b', 'RME_sp':'b', 'RME_sp_L':'g', 'RME':'r','ransacGaussianMean':'y' , 'NP_sp':'k', 'Oracle':'c'}
       
        markers = {'RSPCAdense':'.', 
                    'RSPCAb':'o', 
                    'RME_sp':'o', 
                    'RME_sp_L':'v', 
                    'RME':'^',
                    'ransacGaussianMean':'D' , 
                    'NP_sp':'p', 
                    'Oracle':'x'}
        
        labels = {'RSPCAdense':"Robust dense PCA", 
                'RSPCAb':"Robust sparse PCA", 
                'NP_sp':'NP', 
                'ransacGaussianMean':'RANSAC',
                'RME_sp':'RME_sp',
                'RME_sp_L':'RME_sp_L',
                'Oracle':'oracle',
                'RME':'RME'
                }

        s = len(runs)
        str_keys = [key.__name__ for key in self.keys]

        for key in str_keys:
            A = np.array([res[key] for res in runs])
            if explicit_xs == False:
                xs = np.arange(*bounds)
            else:
                xs = xs
            mins = [np.sort(x)[int(s*0.25)] for x in A.T]
            maxs = [np.sort(x)[int(s*0.75)] for x in A.T]

            plt.fill_between(xs, mins, maxs, color = cols[key], alpha=0.2)
            plt.plot(xs, np.median(A,axis = 0), label=labels[key], color = cols[key], marker =markers[key])

        p = copy.copy(self.params)

        rcParams['figure.figsize'] = figsize

        rc('font', family=fontname, size=fsize)
        rc('axes', labelsize='large')
        rc('legend', numpoints=1)

        plt.title(title, pad = fpad, fontsize = fsize)
        plt.xlabel(xlabel, fontsize = fsize, labelpad = fpad)
        plt.ylabel(ylabel, labelpad = fpad, fontsize = fsize)
        plt.legend()
        plt.ylim(*ylims)
        plt.savefig(outputfilename, bbox_inches = 'tight')
        plt.tight_layout()

    def plot_xloss_fromfile(self, outputfilename, filename, xvar_name, bounds, title, xlabel, ylabel, ylims, y_is_m = False, relative = False, explicit_xs = False, xs = [], fsize = 10, fpad = 10, figsize = (1,1), fontname = 'Arial'):
        Run = self.readdata(filename)
        self.plot_xloss(outputfilename, Run, xvar_name, bounds, title, xlabel, ylabel, ylims, y_is_m, relative, explicit_xs, xs, fsize, fpad, figsize, fontname)

    def plotxy_fromfile(self, outputfilename, filename, xvar_name, bounds, ylims, title, xlabel, ylabel, figsize = (1,1), fsize = 10, fpad = 10, relative = False,  explicit_xs = False, xs = [], fontname = 'Arial'):

        self.plot_xloss_fromfile(outputfilename, filename, xvar_name, bounds, title, xlabel, ylabel, ylims, figsize = figsize, fsize =fsize , fpad = fpad, relative = relative, explicit_xs = explicit_xs, xs = xs, fontname = fontname)
        plt.figure()


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
        