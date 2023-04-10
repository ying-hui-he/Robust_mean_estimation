function [u,sigma,flag]=maxeig(S)
% Compute maximum eigenvalue and eigenvector of S
n=size(S,1);
algo=1;

if n==1
    u=1;sigma=S;
else
    if algo==0
        [V,D]=eig(S);
        [v,idx]=sort(diag(D),'descend');
        u=V(:,idx(1));sigma=v(1);
    else
        options.disp=0;
        options.issym=1;
        [u,sigma,flag]=eigs(0.5*(S+S'),1,'la',options); % ARPACK: Faster for large problems but relatively unstable
        flag=0;
        % restart eigs twice if not converged
        if flag
            disp('ARPACK: eigenvalue decomposition restarted');
            [u,sigma,flag]=eigs(0.5*(S+S'),1,'la',options);
            if flag
                [u,sigma,flag]=eigs(0.5*(S+S'),1,'la',options);
            end
        end
    end
end