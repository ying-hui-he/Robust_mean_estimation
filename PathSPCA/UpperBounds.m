function [bnds,rhov,dualvals]=UpperBounds(A,S,res)
% Compute upper bounds on the sparse maximum eigenvalue problem
% Input: 
%   S: a covariance matrix
%   A: its factorization with S=A'*A
% Output: 
%   bnds: upper bounds from the dual of the card. contrained max eig problem
%   rhov: the corresponding penalties
%   dualvals: the dual objective values of the card. penalized max eig problem

disp('Compute upper bounds on variance...');

ds=diag(S);
if any(ds(1:end-1)-ds(2:end)<0)
    disp('Error in upper bounds: diagonal of input matrix should be decreasing');
    bnds=[];rhov=[];return;
end
if norm(A'*A-S)>1e-12;
    disp('Error in upper bounds: A not factor of S');
    bnds=[];rhov=[];return;    
end

dualvals=[];rhov=[];optrep=[];
% Compute best dual values for each variable subset
kp=size(res,2);
for i=1:kp
    csubset=res(find(res(:,i)),i);
    [isopt,rho]=TestOptimalityDual(A,S,csubset);
    dv=dualvalue(A,S,rho,csubset);
    optrep=[optrep;isopt];dualvals=[dualvals,dv];rhov=[rhov;rho];
end
% Also add full and empty set
dualvals=[dualvals,max(eig(S))];rhov=[rhov;0];
dualvals=[dualvals,0];rhov=[rhov;max(diag(S))];
% Compute bounds
cards=sum(res>0);
primalbnds=dualvals'*ones(1,kp)+rhov*cards;
bnds=min(primalbnds);