function [vars,rhobreaks,res]=FullPathData(A,k)
% Given data matrix A, compute full sparse PCA path
% Input: 
%   A: data matrix (n samples times m variables)
%   k: max cardinality to check
% Output: 
%   vars: vector of variances for each target cardinality
%   rhobreaks: the coresponding rho penalties
%   res: each column contains the corresponding subset of variables

if size(A,2)<size(A,1)
    disp('FullPathData: more observations than variables, use FullPathCov instead');
    vars=[];rhobreaks=[];res=[];
end

n=size(A,2);
vars=sum(A.^2);[vmax,vp]=max(vars);
subset=[vp];subres=[subset';zeros(n-length(subset),1)];
res=[];rhobreaks=[sum(A(:,vp).^2)];sol=[];vars=[];
Stemp=rhobreaks;
% Loop through variables
for i=1:k
    % Compute solution at current subset
    [v,mv]=maxeig(Stemp);
    vsol=zeros(n,1);vsol(subset)=v;
    sol=[sol,vsol];vars=[vars,mv];
    % Compute x at current subset
    x=A(:,subset)*v;x=x/norm(x);
    res=[res,x];
    % Compute next rho breakpoint
    set=1:n;set(subset)=[];
    vals=(x'*A(:,set)).^2;
    [rhomax,vpos]=max(vals);
    rhobreaks=[rhobreaks;rhomax];
    % Update temp covariance matrix and subset
    Stemp=[Stemp,zeros(i,1);zeros(1,i),0]+[zeros(i,i),A(:,subset)'*A(:,set(vpos));A(:,set(vpos))'*A(:,subset),A(:,set(vpos))'*A(:,set(vpos))];
    subset=[subset,set(vpos)];
end