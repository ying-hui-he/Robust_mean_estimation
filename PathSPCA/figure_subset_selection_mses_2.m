clear all
addpath PathSparsePCA
addpath lars


% Initialize parameters ****************

nrepPB = 100;
nrep= 50;
n=16;                   % Dimension
nJ = 8;
p=1000;                  % number of datapoints;
noises=0.01;                % "Signal to noise" ratio

iPB = 2;
seed=iPB;
rand('state',seed);       % Fix random seed
randn('state',seed);       % Fix random seed

covmat = randn(n,n);
covmat = covmat * covmat';
covmat = diag(diag(covmat).^-.5) * covmat * diag(diag(covmat).^-.5);
G = chol(covmat);
w0=[sign(randn(1,nJ)), zeros(1,n-nJ)]';
% normalize w0
w0 = w0 / norm(w0);

J = find(abs(w0))
Jc = find(sign(w0)==0)
% compute consistency condition
consistency_cond = max(covmat(Jc,J) * ( covmat(J,J) \ sign( w0(J) ) ))

    
for irep=1:nrep
seed=irep;
rand('state',seed);       % Fix random seed
randn('state',seed);       % Fix random seed
 
    X = randn(p,n) * G;

    % generate w0 with certain sparsity pattern
    % w0 = 1./sqrt((1:n))';

for inoise=1:length(noises)
    noise = noises(inoise);
seed=inoise;
rand('state',seed);       % Fix random seed
randn('state',seed);       % Fix random seed

    y = X*w0 + noise * randn(p,1);

    w = inv(X'*X)*X'*y;
    full_error = (y'*y - y'*X*inv(X'*X)*X'*y)/p;



    % full greedy optimization
    subset_greedy = []; l2cost_greedy=[];

    % compute multiplier to make the matrix in SPCA positive
    [temp,mu0] =maxeig(X'*X);
    mu0 = mu0 * y'*y;
    optimal = zeros(n,1);
    true_optimal = zeros(n,1);
    for i=1:n
        set=1:n;set(subset_greedy)=[];
        temp=[];
        for j=1:n-i+1
            newset = [ subset_greedy set(j) ];
            temp(j) = (y'*y - y'*X(:,newset)*inv(X(:,newset)'*X(:,newset))*X(:,newset)'*y)/p;
        end
        [a,b]=min(temp);
        l2cost_greedy = [l2cost_greedy a];
        subset_greedy = [ subset_greedy set(b) ];

        % check optimality of solutions by transforming into SPCA
        s0 = y'*X(:,subset_greedy)*inv(X(:,subset_greedy)'*X(:,subset_greedy))*X(:,subset_greedy)'*y;
        S = X'*y*y'*X-s0*X'*X+mu0*eye(n);
        trS = trace(S);
        S = S / trS;
        S=(S+S')/2;

        S0=S;
        [d,ix]=sort(diag(S),'descend');S=S(ix,ix);
        rix = 1:n;
        rix(ix)=1:n;
        subset_greedy0 = rix(subset_greedy);
        A=chol(S);

        [isopt, rho] = TestOptimalityDual(A,S,subset_greedy0)
        optimal(i)=isopt;
true_optimal(i)=length(find(subset_greedy<=nJ));
        
        %  % compute bounds (very high?!?)
        %  [bndsr,rhov,dualvals]=UpperBounds(A,S,subset_greedy0');
        %  bounds(i) = bndsr*trS-mu0;

    end
    % add first:
    l2cost_greedy = [ y'*y/p, l2cost_greedy];
    optimal = [1; optimal];
    true_optimal = [1; true_optimal];
    optimal
    % full greedy optimization

    % full greedy optimization - BACKWARD
    subset_greedy_BW = [1:n]; l2cost_greedy_BW=[];

    % compute multiplier to make the matrix in SPCA positive
    optimal_BW = zeros(n-1,1);
    true_optimal_BW = zeros(n-1,1);
    for i=1:n-1
        temp=[];
        for j=1:n-i+1
            newset = subset_greedy_BW;
            newset(j)=[];
            temp(j) = (y'*y - y'*X(:,newset)*inv(X(:,newset)'*X(:,newset))*X(:,newset)'*y)/p;
        end
        [a,b]=min(temp);
        l2cost_greedy_BW = [l2cost_greedy_BW a];
        subset_greedy_BW(b) = [];

        % check optimality of solutions by transforming into SPCA
        s0 = y'*X(:,subset_greedy_BW)*inv(X(:,subset_greedy_BW)'*X(:,subset_greedy_BW))*X(:,subset_greedy_BW)'*y;
        S = X'*y*y'*X-s0*X'*X+mu0*eye(n);
        trS = trace(S);
        S = S / trS;
        S=(S+S')/2;

        S0=S;
        [d,ix]=sort(diag(S),'descend');S=S(ix,ix);
        rix = 1:n;
        rix(ix)=1:n;
        subset_greedy_BW0 = rix(subset_greedy_BW);
        A=chol(S);

        [isopt, rho] = TestOptimalityDual(A,S,subset_greedy_BW0)
        cards_BW(i) = length(subset_greedy_BW0);
        optimal_BW(i)=isopt;
        true_optimal_BW(i)=length(find(subset_greedy_BW<=nJ));
        
    end
    % add first and last
    l2cost_greedy_BW = [ l2cost_greedy(end), l2cost_greedy_BW,y'*y/p, ];
    optimal_BW = [1; optimal_BW; 1];
    optimal_BW = flipud(optimal_BW);
    true_optimal_BW = [1; true_optimal_BW; 1];
    true_optimal_BW = flipud(true_optimal_BW);
    l2cost_greedy_BW = fliplr(l2cost_greedy_BW);



    % LASSO
    beta = lars(X, y, 'LASS0');
    beta = abs(sign(beta(2:end,:)));
    true_optimal_lasso = zeros(n,1);
    optimal_lasso = zeros(n,1);
    number_sel = sum(sign(beta),2);
l2cost_lasso = [];
    for i=1:n
        newset = find(beta(max(find(number_sel<=i)),:)); % if more than one value with given card
        l2cost_lasso(i) = (y'*y - y'*X(:,newset)*inv(X(:,newset)'*X(:,newset))*X(:,newset)'*y)/p;

        % check optimality of solutions by transforming into SPCA
        s0 = y'*X(:,newset)*inv(X(:,newset)'*X(:,newset))*X(:,newset)'*y;
        S = X'*y*y'*X-s0*X'*X+mu0*eye(n);
        S = S / trace(S);
        S=(S+S')/2;

        S0=S;
        [d,ix]=sort(diag(S),'descend');S=S(ix,ix);
        rix = 1:n;
        rix(ix)=1:n;
        newset0 = rix(newset);
        A=chol(S);

        [isopt, rho] = TestOptimalityDual(A,S,newset0)
        optimal_lasso(i)=isopt;
true_optimal_lasso(i)=length(find(newset<=nJ));
        


    end

% compute optimal subsets
if n<=20
    indices = ind2subv(ones(n,1)*2,1:2^n)-1;
    bestsubset = Inf*ones(n+1,1);
    for i=1:2^n
        
        newset = find(indices(i,:));
        if ~isempty(newset)
        temp = (y'*y - y'*X(:,newset)*inv(X(:,newset)'*X(:,newset))*X(:,newset)'*y)/p;
        bestsubset(length(newset)+1)=min(bestsubset(length(newset)+1),temp);
        else
            bestsubset(1) = y'*y/p;
        end

    end
end
    OPT_L2(inoise,irep,:) = bestsubset;



    l2cost_lasso = [ y'*y/p, l2cost_lasso];
    optimal_lasso = [1; optimal_lasso];
    true_optimal_lasso = [1; true_optimal_lasso];


    TRUE_GREEDY_OPT(inoise,irep,:) = max(true_optimal_BW,true_optimal)';
    GREEDY_OPT(inoise,irep,:) = or(optimal_BW,optimal)';
    LASSO_OPT(inoise,irep,:) =optimal_lasso';
    TRUE_LASSO_OPT(inoise,irep,:) =true_optimal_lasso';
    
    GREEDY_L2_FD(inoise,irep,:) =l2cost_greedy;
    GREEDY_L2_BW(inoise,irep,:) = l2cost_greedy_BW;
    GREEDY_L2(inoise,irep,:) = min(l2cost_greedy,l2cost_greedy_BW);
    LASSO_L2(inoise,irep,:) = l2cost_lasso';
end
end

clear indices

plot(0:n,squeeze(mean(GREEDY_L2_FD(:,:,:),2))','bx-','linewidth',3); hold on
plot(0:n,squeeze(mean(GREEDY_L2_BW(:,:,:),2))','kx-','linewidth',3); hold on
plot(0:n,squeeze(mean(LASSO_L2(:,:,:),2))','gx-','linewidth',3); hold on
plot(0:n,squeeze(mean(OPT_L2(:,:,:),2))','rx-','linewidth',3); hold off
title('Performance of estimation methods');
xlabel('subset cardinality');
ylabel('MSE')
legend('forward','backward','lasso','optimal');

save figure_subset_selection_mses_2

