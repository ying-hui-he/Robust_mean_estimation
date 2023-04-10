% Compare gaps from Approx. Greedy Path and DSPCA subsets
if 1
% Initialize parameters ****************
n=90;               % Dimension
ratio=10;         % "Signal to noise" ratio
rand('state',25);   % Fix random seed
% Form test matrix as: rank one sparse + noise
testvec=[ones(1,n/3),1./(1:(n/3)),zeros(1,n/3)];
testvec=testvec/(norm(testvec));
S=rand(n,n);
S=S'*S/n+ratio*testvec'*testvec;
kp=60;
end

if 1
% Compute approx. regularization path
[d,ix]=sort(diag(S),'descend');S=S(ix,ix);
A=chol(S);
[vars,rhobreaks,res]=FullPathCov(S);
[bnds,rhov]=UpperBounds(A,S,res(:,1:kp));
error=max(bnds-vars(1:kp),zeros(1,kp));
end

if 1 % Fill in the blanks using DSPCA
% DSPCA code params
maxiter=1000;    % Maximum Number of Iterations
gapchange=1e-2;           % Target precision 
info=100;          % Control amount of output
testrho=rhobreaks(find(error>1e-1)); % TEST: adjust this to get more optimal points? TEST
dualsD=[];cardD=[];rawvars=[];patresD=[];resopt=[];
for rhov=testrho'
    [U,X,x]=DSPCA(S,rhov,gapchange,maxiter,info,1);
    patt=(abs(x)>gapchange);cardd=sum(patt);Ddval=max(eig(S+U));rawvars=[rawvars,x'*S*x];
    dualsD=[dualsD,Ddval];cardD=[cardD,cardd];patresD=[patresD,[find(patt);zeros(n-sum(patt),1)]];
end
dualvals=dualsD;rhov=testrho; 
dualvals=[dualvals,max(eig(S))];rhov=[rhov;0];
dualvals=[dualvals,0];rhov=[rhov;max(diag(S))];
cards=sum(patresD>0);
primalbnds=dualvals'*ones(1,length(cards))+rhov*cards;
Dbnds=min(primalbnds);
Dcards=sum(patresD>0);
[sDcards,dix]=sort(Dcards);
end

epsr=1e-4;
error=max(bnds(1:kp)-vars(1:kp),zeros(size(vars(1:kp))));
optpi=find((error./vars(1:kp))<=epsr);optpv=vars(1,find((error./vars(1:kp))<=epsr));
plot(1:kp,vars(1:kp),'-b',Dcards(dix),Dbnds(dix),'--r','LineWidth',2); hold on;
plot(1:kp,bnds,':k','LineWidth',3);
plot(optpi,optpv,'b.','MarkerSize',25);hold off;
xlabel('card');ylabel('var');
axis([1 kp min(vars) max(vars)])