function [isopt, rho] = TestOptimalityDual(A,S,subset)
% Test the optimality of subset 
% Input: a covariance matrix S and its factorization A such that S=A'*A, a subset of variables.
% Output: isopt=1 if optimal, also returns the rho penalty corresponding to the best duality gap found for that subset

ds=diag(S);
if any(ds(1:end-1)-ds(2:end)<0)
    disp('Error in optimality test: diagonal of input matrix should be decreasing');
    isopt=0;rho=NaN;return;
end
if norm(A'*A-S)>1e-12;
    disp('Error in optimality test: A not factor of S');
    isopt=0;rho=NaN;return;    
end

n = size(S,1);
eps=1e-7;
isopt = 0;
rho = NaN;
bestgap=Inf;
b=Inf;J=[];M=[];Jopt=[];
A=[A;zeros(max(0,n-size(A,1)),n)];

% Basic checks
if length(subset)==1 
    % if it is the first one it is optimal
    if subset==1,
        isopt=1;
        rho=S(1,1);
        b=0;
    else
        disp('S matrix not diagonal decreasing...');
        isopt=0;
        rho=NaN;
    end
    return;
end

% Full subset is always optimal
if length(subset)==n;
    isopt=1;
    rho=0;
    return;
end

% Compute dominant eigenvector
subset_c=1:n; subset_c(subset)=[];
[v,mv]=maxeig(S(subset,subset));
x=A(:,subset)*v;v=v/norm(x);x=x/norm(x);

% Compute first interval on rho (from basic consistency condition)
temp = S(:,subset)*v; 
rhomin_active=max( temp(subset_c).^2 );
rhomax_active=min( temp(subset).^2 );
if ( rhomin_active >= rhomax_active )
    % the interval is empty -> subset cannot be optimal
    txt=sprintf('Card: %d, basic consistency failed',length(subset));disp(txt);
    isopt = 0;rho = NaN;
    return
end

% Remove variables such that ||a_j||^2<rho_min_active (i.e. with Y_j=0 in the dual)
allowed_c=subset_c(find(diag(S(subset_c,subset_c))>rhomin_active ));

% Initialize
aiTx=A'*x;  % NB : also equal to S(:,subset)*v

% Find optimal rho
txt=sprintf('Card: %d, minimizing gap...',length(subset));disp(txt);
I = subset;
% Define gap minimization interval in rho
local_rhomax=min(rhomax_active)-eps; % NUMISSUE: cannot be too close to the limit
local_rhomin=max(rhomin_active)+eps; % NUMISSUE: cannot be too close to the limit
% Compute gap on both ends
dvmin=dualvalue(A,S,local_rhomin,subset);pmin=sum(aiTx(I).^2-local_rhomin);
dvmax=dualvalue(A,S,local_rhomax,subset);pmax=sum(aiTx(I).^2-local_rhomax);
% Test if one of them is optimal
if (dvmin-pmin<=eps)
    rho=local_rhomin+eps; % NUMISSUE: cannot be too close to the boundary
    isopt=1;Jopt=subset_c(find(diag(S(subset_c,subset_c))>rho));
end
if (dvmax-pmax<=eps)
    rho=local_rhomax-eps;% NUMISSUE: cannot be too close to the boudnary
    isopt=1;Jopt=subset_c(find(diag(S(subset_c,subset_c))>rho));;
end

% Otherwise launch binary search on duality gap in rho
if isopt==0
    nrhos = 10;
    Ap=A(:,J)-x*x'*A(:,J);
    Gm=-Ap*diag(sum(Ap.^2).^(-1))*Ap'+eye(n)*length(subset);
    for irho=1:nrhos
        % Compute primal and dual values, and corresponding eigenvector
        rho_candidate=(local_rhomin+local_rhomax)/2;
        J=subset_c(find(diag(S(subset_c,subset_c))>rho_candidate));
        [dval,tv]=dualvalue(A,S,rho_candidate,subset);pval=sum(aiTx(I).^2-rho_candidate);gap=dval-pval;
        % Compute derivative of gap w.r.t. rho
        alphai=aiTx(I).^2-rho_candidate;
        vi=A(:,I)*diag(aiTx(I))-rho_candidate*x*ones(1,length(I));
        vip=-x*ones(1,length(I));
        Gm=Gm+vi*diag(alphai.^-1)*vip'+vip*diag(alphai.^-1)*vi'+vi*diag(alphai.^-2)*vi';
        if ~isempty(J)
            alphaj=-(aiTx(J).^2-rho_candidate);
            vj=A(:,J)*diag(aiTx(J))-x*(aiTx(J).^2)';
            Gm=Gm-vj*diag(alphaj.^-2)*vj';
        end
        grad_lmax=tv'*Gm*tv;
        % Check if optimal, otherwise find new rho candidate
        if (gap<=eps)
            rho=rho_candidate;
            isopt=1;Jopt=J;
            break;
        elseif grad_lmax<0
            local_rhomin = rho_candidate;
        else
            local_rhomax = rho_candidate;
        end
    end
    if (isopt==0)&(gap<bestgap);
        rho=rho_candidate; % Best rho found so far
        Jopt=J;
    end
end    