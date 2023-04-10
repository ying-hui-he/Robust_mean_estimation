% Initialize parameters 
n=150;               % Dimension
rand('state',25);   % Fix random seed
ratiolist=[10,50,100];
kp=n;

vars=[];bnds=[];
for ratio=ratiolist
    % Form test matrix as: rank one sparse + noise
    testvec=[ones(1,n/3),1./(1:(n/3)),zeros(1,n/3)];
    testvec=testvec/(norm(testvec));
    S=rand(n,n);
    S=S'*S/n+ratio*testvec'*testvec;
    % Sort A w.r.t. diag
    [d,ix]=sort(diag(S),'descend');S=S(ix,ix);
    % Cholesky of S
    A=chol(S);
    % Compute regularization path
    tic;
    [varsr,rhobreaks,res]=FullPathCov(S);
    [bndsr,rhov]=UpperBounds(A,S,res(:,1:kp));
    toc;
    vars=[vars;varsr];bnds=[bnds;bndsr];
end

if 1
epsr=1e-4;
plot(1:kp,vars(1,1:kp),'-b','LineWidth',2);hold on;
plot(1:kp,bnds(1,:),':k','LineWidth',3);
error=max(bnds(1,1:kp)-vars(1,1:kp),zeros(size(vars(1,1:kp))));
optpi=find((error./vars(1,1:kp))<=epsr);optpv=vars(1,find((error./vars(1,1:kp))<=epsr));
plot(optpi,optpv,'b.','MarkerSize',20);
for i=2:length(ratiolist)
    plot(1:kp,vars(i,1:kp),'-b','LineWidth',2);
    plot(1:kp,bnds(i,:),':k','LineWidth',3);
    error=max(bnds(i,1:kp)-vars(i,1:kp),zeros(size(vars(i,1:kp))));
    optpi=find((error./vars(i,1:kp))<=epsr);optpv=vars(i,find((error./vars(i,1:kp))<=epsr));
    plot(optpi,optpv,'b.','MarkerSize',20);
end
xlabel('card');
ylabel('var');
hold off;
end