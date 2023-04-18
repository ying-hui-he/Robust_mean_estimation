import matlab
import matlab.engine
import numpy as np
import powerlaw
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150

eng = matlab.engine.start_matlab()

class Params(object):
    def __init__(self, n, d = 500, k = 4, p = 0.1):
        self.d = d
        self.k = k
        self.p = p
        self.n = n

"""Noise type"""
def robust_mean_estimate_full(n=1200):
# establish the dataset
    d = 500 # dimension
    k = 4 # sparsity
    #n = 12000 # sample size
    K = int(n/4) # number of subgroups
    # TODO plot a graph with K as x axis and y as performance
    p = 0.1 # corruption probability
    mu = np.zeros(d)
    mu_sparse = [20, -10, -5, 2]
    mu[:k] = mu_sparse

    setting = 'powerlaw'
    distribution = 'normal'
    X = np.empty((n, d))

    if (distribution == 'normal'):
        # normal distribution
        X = np.random.normal(0, 1, size=(n, d))
    else: 
        # power law distribution
        alpha = 2.5
        xmin = 1
        mean = np.zeros((n, d))
        mean += (alpha / (alpha - 1)) * xmin
        dist = powerlaw.Power_Law(xmin=xmin, parameters=[alpha])
        for i in range(n):
            X[i, :] = dist.generate_random(d)
        X = X - mean
        #print(X[0])

    for i in range(k):
        X[:, i] += mu_sparse[i]

    # print(np.mean(X, axis=0))



    # Add corrupted data 
    for j in range(np.floor(p * n).astype(int)):
        X[j, :] = 20 + 50 * np.random.standard_cauchy(size=(1, d)) # non-standard Cauchy, mean 20, scale 50
    # print(np.mean(X, axis=0))

    # Grouping preprocessing
    X_grouped = np.split(X, K)
    X_grouped = np.mean(X_grouped, axis=1)

    # print(len(X_grouped))

    # initialize the parameters
    alpha = 1e-3
    u = alpha * np.ones(d)
    v = alpha * np.ones(d)


    eta = 0.05
    rho = 1
    max_iter = 1000
    #dists = np.zeros(max_iter)
    #error = np.zeros(max_iter, d)
    #pred = np.zeros(max_iter, d)

    for t in range(max_iter):
        grad_u = np.zeros(d)
        grad_v = np.zeros(d)
        for i in range(K):
            grad_u += -np.sign(X_grouped[i, :].reshape(d) - u * u + v * v) * u
            grad_v += np.sign(X_grouped[i, :].reshape(d) - u * u + v * v) * v
        u -= eta * grad_u / K
        v -= eta * grad_v / K
        eta *= rho
        #dists[t] = np.linalg.norm(u * u - v * v - mu, ord=1)
        #error[t] = abs(u * u - v * v - mu)
        #pred[t] = abs(u * u - v * v)
    #print(u)
    estimated_mean_1 = u * u - v * v
    indices_top_k = np.argpartition(np.abs(estimated_mean_1), -k)[-k:]
    estimated_mean_stage_1 = np.zeros(len(estimated_mean_1))
    new_X = X[:,indices_top_k]
    for i in indices_top_k:
        estimated_mean_stage_1[i] = estimated_mean_1[i]

    new_X = matlab.double(new_X.tolist())
    estimated_mean_2 = eng.robust_mean_pgd(new_X,p,5000)

    estimated_mean_total = np.zeros(len(estimated_mean_1))
    j = 0
    for i in indices_top_k:
        estimated_mean_total[i] = estimated_mean_2[j][0]
        j = j + 1

    error = np.linalg.norm(estimated_mean_total - mu, ord = 1)
    error_1 = np.linalg.norm(estimated_mean_stage_1 - mu, ord = 1)
    
    return error, error_1 ,estimated_mean_total
#print("estimated_mean_total:")
#print(estimated_mean_total)
#print("shape: ")
#print(estimated_mean_total.shape)

x_axis = []
y_axis_1 = []
y_axis_2 = []

for i in range(4,1004,4):
    x_axis.append(i)
    error, error_1, mean = robust_mean_estimate_full(i)
    y_axis_1.append(error)
    y_axis_2.append(error_1)

plt.plot(x_axis,y_axis_1)
plt.ylabel('Loss')
plt.xlabel('Samples')
plt.savefig(f'tem_3.png')
plt.show()

plt.plot(x_axis,y_axis_1,'b',x_axis,y_axis_2,'g')
plt.ylabel('Loss')
plt.xlabel('Samples')
plt.savefig(f'tem_4.png')
plt.show()



#t_min = np.argmin(dists)
#print(error[t_min])
#print(pred[t_min])

#error = np.transpose(error, (1, 0, 2))
#pred = np.transpose(pred, (1, 0, 2))
'''
# prediction plot
point_num = 10
for i in range(point_num):
    plt.plot(pred[i])
# plt.yscale('log')
plt.ylabel('Predicted value')
plt.xlabel('Time')
plt.title(f'The predicted values of the first {point_num} coordinates varied with time.')

# plt.savefig(f'output/{setting}_pred.png')
plt.show()

# error plot
for i in range(point_num):
    plt.plot(error[i])
plt.yscale('log')
plt.ylabel('Error')
plt.xlabel('Time')
plt.title(f'The prediction error of the first {point_num} coordinates varied with time.')

# plt.savefig(f'output/{setting}_error.png')
plt.show()

plt.plot(dists)
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Time')
plt.title(f'The loss varied with time.')
# plt.savefig(f'output/{setting}_loss.png')
np.min(dists)


def topk_abs(v, k):
    # print("v: ", v)
    u = np.argpartition(np.abs(v), -k)[-k:]
    # print("u: ", u)
    # print(2)
    z = np.zeros(len(v))
    z[u] = v[u]
    return z
'''