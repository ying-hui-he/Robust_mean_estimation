# Robust Sparse Mean Estimation

This project implement the subgradient method for robust mean estimation and compare it with other algorithms under the heavy-tailed setting. We make use of some of the code in the following papers for our experiments: "Outlier-Robust High-Dimensional Sparse Estimation via Iterative Filtering" (https://arxiv.org/abs/1911.08085), "Outlier-Robust Sparse Estimation via Non-Convex Optimization" (https://arxiv.org/abs/2109.11515) and Robust Learning of Fixed-Structure Bayesian Networks in Nearly-Linear Time (https://arxiv.org/abs/2105.05555).

Prerequisities
==
* Linux or macOS
* python3+
* matlab

The main libraries used in this project are listed below:
* `numpy`
* `scipy`
* `matplotlib`

For more information, refer to `requirements.txt`.

Explanation of Files
==
* `robust_mean_estimate.py`: Library containing code for robust mean estimation algorithm and plot function which compare the performance of these algorithms across various heavy-tailed distribution. 

* `robust_mean_estimate_acc.py`: Library containing code for checking the accuracy of stage 1 algorithm.

* `acc.ipynb`: Jupyter notebook to test the success rate of stage 1 algorithm across varying corruption ratios $\epsilon$. 

* `err-k.ipynb`: Jupyter notebook to compare the performance for these algorithms across varying sparsity levels $k$.

* `inf.ipynb`: Jupyter notebook to compare the performance for these algorithms in the infinite variance regime with respect to the heaviness of the tail distribution.

* `sensitivity.ipynb`: Jupyter notebook to compare the performance for these algorithms across varying prior knowledge of upper bound of sparsity.

* `err-eps.ipynb`: Jupyter notebook to compare the performance for these algorithms across varying corruption ratios $\epsilon$.

* `time.ipynb`: Jupyter notebook to compare the running time for these algorithm across varying dimention $d$.