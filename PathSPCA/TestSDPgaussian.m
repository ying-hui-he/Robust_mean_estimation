% This script generates upper and lower bounds on sparse eigenvalues of random matrices
close all;clear all;

% Initialize parameters ****************
rand('state',25);       % Fix random seed

if 0
n=10; % Dimension
kp=n;
ratio=10; % "Signal to noise" ratio
% Form test matrix as: rank one sparse + noise
testvec=1./((1:n)); 
testvec=testvec/(norm(testvec));
S=rand(n,n);
S=S'*S/n+ratio*testvec'*testvec;
[d,ix]=sort(diag(S),'descend');S=S(ix,ix);S=S/S(1,1);
end

if 1 % Random matrices
p=8;
n=10;
kp=8;
randn('state',33);
% Generate random matrix
F=randn(p,n)/sqrt(p); % Gaussian
%F=sign(randn(p,n))/sqrt(p); % Bernoulli
S=F'*F;
S=S+eye(size(S))*1e-14;
[d,ix]=sort(diag(S),'descend');S=S(ix,ix);F=chol(S);
end

% Compute approx. greedy path
F=chol(S);tic;
[pathvars,rhobreaks,res]=FullPathCov(S);
[bnds,rhov,dualvals]=UpperBounds(F,S,res(:,1:kp));
error=max(bnds-pathvars(1:kp),zeros(size(pathvars(1:kp))));

% Compute fully greedy path
[greedyres,sol,greedyvars,greedyrhos,res]=FullPathGreedyFB(F,S,kp,1,1);
[greedybnds,greedyrhov,greedydualvals]=UpperBounds(F,S,greedyres(:,1:kp));

% Exhaustive search
indices=ind2subv(ones(n,1)*2,1:2^n)-1;
exvars=zeros(1,n);exres=zeros(n,n);exrhosmin=zeros(1,n);exrhosmax=zeros(1,n);
for v=indices(2:end,:)'
    subset=find(v>0);
    [vv,sv]=maxeig(S(subset,subset));
    if exvars(sum(v))<sv
        exvars(sum(v))=sv;exres(:,sum(v))=[subset;zeros(n-length(subset),1)];
        vsol=zeros(n,1);vsol(subset)=vv;
        x=F(:,subset)*vv;x=x/norm(x);
        aiTx=(x'*F(:,subset)).^2;
        outset=1:n;outset(subset)=[];ajTx=(x'*F(:,outset)).^2;
        exrhosmax(sum(v))=min(aiTx);
        exrhosmin(sum(v))=max([ajTx,0]);
    end
end

% Upper bounds from exhaustive search points
[greedyres,sol,greedyvars,greedyrhos,res]=FullPathGreedyFB(F,S,kp,1,1);
[greedybnds,greedyrhov,greedydualvals]=UpperBounds(F,S,greedyres(:,1:kp));

% Launch SDP relaxation
sdpres=[1;zeros(n-1,1)];i=2;sdpvars=[S(1,1)];sdpdual=[];tic;
rhotest=(S(1,1)-(1:25)*(S(1,1)-exrhosmax(kp))/25)';
%rhotest=rhov(2:kp);
for rho=rhotest'
    if ~isnan(rho)
        [y,x,u,bufm,obj]=sdprelax(S,rho);
        bufm=(bufm+bufm')/2;
        xm=reshape(x(end-(n^2)+1:end),n,n);
        [V,D] = eig(xm);[Va,Da]=eig(S);
        [me,es]=sort(diag(D),'descend');
        xv=V(:,es(1));
        disp('-------------------------------------------------------------------------------');
        disp(['Target card: ',num2str(i),'   Eigs. of X: ',num2str(sort(diag(D)','Descend'),'%.2f ')]);
        vm=V(:,n);vm=vm*sign(mean(vm));
        A=chol(S);vpat=(A'*xv).^2-rho; % Compute sparsity pattern
        zpat=vpat>0;disp(['Solution pattern:   ',num2str(zpat')]);
        if sum(zpat)>0
            sdpres=[sdpres,[find(zpat);zeros(n-sum(zpat),1)]];i=i+1;
            [vv,svv]=maxeig(S(find(zpat),find(zpat)));sdpvars=[sdpvars,svv];sdpdual=[sdpdual,obj];
        else
            sdpres=[sdpres,[Inf*ones(n,1)]];i=i+1;
            sdpvars=[sdpvars,-Inf];sdpdual=[sdpdual,Inf];
        end
    else
        sdpres=[sdpres,[Inf*ones(n,1)]];i=i+1;
        sdpvars=[sdpvars,-Inf];sdpdual=[sdpdual,Inf];
    end
end
% Get upper bounds by duality
rhotest=[rhotest;0;S(1,1)];
sdpdual=[sdpdual,max(eig(S)),0];
sdpcards=sum(sdpres>0);
sdpprimalbnds=sdpdual'*ones(1,length(sdpcards))+rhotest*sdpcards;
sdpup=min(sdpprimalbnds);sdpmvars=-Inf*zeros(1,kp);
for i=1:length(sdpcards)
    sdpmvars(sdpcards(i))=max(sdpmvars(sdpcards(i)),sdpvars(i));
end
toc;

if 1 % Try DSPCA
% DSPCA code params
maxiter=1000;    % Maximum Number of Iterations
gapchange=1e-2;           % Target precision 
info=100;          % Control amount of output
dualsD=[];cardD=[];rawvars=[];patresD=[];resopt=[];
Dtestrho=rhotest(1:end-2);
for rhod=Dtestrho'
    [U,X,x]=DSPCA(S,rhod,gapchange,maxiter,info,1);
    patt=(abs(x)>2*gapchange);cardd=sum(patt);Ddval=max(eig(S+U));rawvars=[rawvars,x'*S*x];
    dualsD=[dualsD,Ddval];cardD=[cardD,cardd];patresD=[patresD,[find(patt);zeros(n-sum(patt),1)]];
end
dualvals=dualsD; 
dualvals=[dualvals,max(eig(S))];Dtestrho=[Dtestrho;0];
dualvals=[dualvals,0];Dtestrho=[Dtestrho;max(diag(S))];
cards=sum(patresD>0);
primalbnds=dualvals'*ones(1,length(cards))+Dtestrho*cards;
Dbnds=min(primalbnds);
Dcards=sum(patresD>0);
[sDcards,dix]=sort(Dcards);
end

% Plot result
plot(1:kp,exvars(1:kp),'k','LineWidth',2);hold on;
plot(1:kp,pathvars(1:kp),':b','LineWidth',2);
plot(1:kp,greedyvars(1:kp),'--b','LineWidth',2);
plot(1:kp,sdpmvars,':ro','LineWidth',2,'MarkerSize',10);
plot(1:kp,greedybnds(1:kp),'ks','LineWidth',2,'MarkerSize',10);
plot(sdpcards,sdpup,'b*','LineWidth',2,'MarkerSize',12);
plot(Dcards,Dbnds,'gd','LineWidth',2,'MarkerSize',12);
legend({'Exhaustive','App. Greedy','Greedy','SDP var.','Dual Greedy','Dual SDP','Dual DSPCA'},'Location','SouthEast','FontSize',16)
xlabel('card','FontSize',16);ylabel('var','FontSize',16);axis([1 kp S(1,1) 1.05*max(greedybnds)])
hold off;