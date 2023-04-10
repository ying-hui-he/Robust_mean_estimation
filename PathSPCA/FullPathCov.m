function [vars,rhobreaks,res]=FullPathCov(S)
% Given covariance matrix, compute full sparse PCA path
% Input: 
%   S: covariance matrix with decreasing diagonal
% Output: 
%   vars: vector of variances for each target cardinality
%   rhobreaks: the coresponding rho penalties
%   res: each column contains the corresponding subset of variables

ds=diag(S);
if any(ds(1:end-1)-ds(2:end)<0)
    disp('Error in FullPathCov input: diagonal of input matrix should be decreasing');
    isopt=0;rho=NaN;return;
end

n=size(S,1);A=chol(S);
subset=[1];subres=[subset';zeros(n-length(subset),1)];
res=[];rhobreaks=[sum(A(:,1).^2)];sol=[];vars=[];

% Loop through variables
for i=1:n
    % Compute solution at current subset
    [v,mv]=maxeig(S(subset,subset));
    vsol=zeros(n,1);vsol(subset)=v;
    sol=[sol,vsol];vars=[vars,mv];
    % Compute x at current subset
    x=A(:,subset)*v;x=x/norm(x);
    res=[res,[subset';zeros(n-length(subset),1)]];
    % Compute next rho breakpoint
    set=1:n;set(subset)=[];
    vals=(x'*A(:,set)).^2;
    [rhomax,vpos]=max(vals);
    rhobreaks=[rhobreaks;rhomax];
    subset=[subset,set(vpos)];
end