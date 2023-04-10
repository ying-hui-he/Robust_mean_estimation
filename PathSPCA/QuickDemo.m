% Compute Full Path for sparse PCA problems given covariance or data info

if 1
% Random example
n=50;               % Dimension
ratio=10;         % "Signal to noise" ratio
rand('state',25);   % Fix random seed
% Form test matrix as: rank one sparse + noise
testvec=1./(1:n);
testvec=testvec/(norm(testvec));
S=rand(n,n);
S=S'*S/n+ratio*testvec'*testvec;
end

if 0
% Gene expression data
load SampleCov.mat
S=cov3;n=size(cov3,1);
A=chol(S);
end

disp(' ');
% Test path given covariance
[d,ix]=sort(diag(S),'descend');S=S(ix,ix);% Sort covariance w.r.t. diag
tic;disp('Computing full regularization path from covariance matrix...')
[vars1,rhos1,res1]=FullPathCov(S);
toc;

% Test path given data matrix only (Cholesky of covariance here)
A=chol(S);
disp(' ');tic;disp('Computing full regularization path from data matrix...')
[vars2,rhos2,res2]=FullPathData(A,10);
toc;

% Test optimality of certain patterns
disp(' ');disp('Testing optimality of some points...')
pattern1=res1(1:6,6); % Cardinality 6
[opt1, rho1]=TestOptimalityDual(A,S,pattern1);
disp(['Pattern 1 optimality: ',num2str(opt1)]);
pattern2=res1(1:25,25); % Cardinality 25
[opt2, rho2]=TestOptimalityDual(A,S,pattern2);
disp(['Pattern 2 optimality: ',num2str(opt2)]);

% Plot variance versus cardinality
plot(1:length(vars1),vars1,'-','LineWidth',2)
xlabel('Cardinality');ylabel('Variance');