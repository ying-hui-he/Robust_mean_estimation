% Test code on various examples
tic;clear all;

if 0 % Random matrices
p=75;
m=100;
k=m;
kp=10;
randn('state',33);
% Generate random matrix
F=randn(p,m)/sqrt(p); % Gaussian
%F=sign(randn(p,m))/sqrt(p); % Bernoulli
S=F'*F;
S=S+eye(size(S))*1e-14;
[d,ix]=sort(diag(S),'descend');S=S(ix,ix);F=F(:,ix);
end

if 1 % Bio data eisen
load EisenData
S=eisen2+eye(size(eisen2))*1e-14;
[d,ix]=sort(diag(S),'descend');S=S(ix,ix);
F=chol(S);
k=size(S,1);
kp=k;
end

if 0 % Bio data colon
load CovColon
S=covcolon+eye(size(covcolon))*1e-12;
[d,ix]=sort(diag(S),'descend');S=S(ix,ix);
F=chol(S);
k=size(S,1);
kp=k;
end

if 0 % Bio data lymphoma
load LymphomaCov.mat
S=covlymph+eye(size(covlymph))*1e-10;
[d,ix]=sort(diag(S),'descend');S=S(ix,ix);
F=chol(S);
k=size(S,1);
kp=k;
end

if 1 % Compute lower and upper bound on variance
[vars,rhobreaks,res]=FullPathCov(S);
[bnds,rhov,dualvals]=UpperBounds(F,S,res(:,1:kp));toc;
end

if 1 % Plot with optimal points. Uncomment to merge multiple plots
epsr=1e-4;
error=max(bnds(1:kp)-vars(1:kp),zeros(size(vars(1:kp))));
optpi=find((error./vars(1:kp))<=epsr);optpv=vars(find((error./vars(1:kp))<=epsr));
%errorl=max(bndsl-varsl(1:kp),zeros(size(vars(1:kp))));
%optpil=find((errorl./varsl(1:kp))<=epsr);optpvl=varsl(find((errorl./varsl(1:kp))<=epsr));
plot(1:kp,vars(1:kp),'-b',1:kp,bnds,':k','LineWidth',1.5);hold on;
%plot(1:kp,varsl(1:kp),'-b',1:kp,bndsl,':k','LineWidth',1.5);
%plot(optpil,optpvl,'b.','MarkerSize',20);
plot(optpi,optpv,'b.','MarkerSize',20);
xlabel('card');ylabel('var');
%semilogy(error./vars,'bo');xlabel('Card');ylabel('Relative Error');axis([1 kp 1e-6 1e2]);
end