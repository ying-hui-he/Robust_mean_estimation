% This script simply calls other scripts in the directory to generate the figures in the paper
% If the QuickDemo.m code runs, the code is working properly. 

% Just a quick demo with timing
disp('*** First produce a simple variance versu cardnality plot... ***')
figure;QuickDemo

% Generate the variance vs cardinality plots on biological data
disp('*** Same thing on biological data ***')
figure;BioAndRandomPlots % Edit the script itself to switch data sets

% === What follows is only focused on comparisons with other algorithms ====

% Plot upper and lower bounds for various methods on 
% restricted eigenvalues of random matrices.
disp('The next experiments require DSPCA and SEDUMI to be installed')
disp('Some mex binaries will hoepfully be compiled...')
testv=input('Continue [y/n]? ','s');
if testv=='y'
% Compile ind2subv.c first
mex ind2subv.c
figure;TestSDPgaussian

% Generate the optimality vs noise figure and the comparison with DSPCA
% This requires the DSPCA code. Source and binaries available at: http://www.princeton.edu/~aspremon/DSPCA.htm
% This also requires SEDUMI to be installed and in your path.
figure;OptimalityVsNoise
figure;DSPCAinGaps

% Generate plots on greedy vs LASSO optimality. The LASSO plots were generated using LARS (not included).
% The function TestOptimalitySubsetSelection.m can be used to test the optimality of a pattern. 
figure; figure_subset_selection_opt_1.m
figure; figure_subset_selection_opt_2.m
figure; figure_subset_selection_mses_1
figure; figure_subset_selection_mses_2
end