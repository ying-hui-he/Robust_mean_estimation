function [dv,tv]=dualvalue(A,S,rho,pattern)
% Computes dual value for a given rho and a pattern (when basic consitency condition is satisifed)
tv=[];

ds=diag(S);
if any(ds(1:end-1)-ds(2:end)<0)
    disp('Error in dualvalue input: diagonal of input matrix should be decreasing');
    dv=0;tv=[];return;
end
if norm(A'*A-S)>1e-12;
    disp('Error in dualvalue input: A not factor of S');
    dv=0;tv=[];return;    
end

n = size(S,1);
set=1:n;set(pattern)=[];

if length(pattern)==1
    dv=S(1,1)-rho;tv=[1;zeros(n-1,1)];return;
end

% build x and v
[v,mv]=maxeig(S(pattern,pattern));
x = A(:,pattern) * v;
v = v / norm(x);
x = x / norm(x);
temp = S(:,pattern)*v;
if isempty(set)
    rhomin_active = 0;
else
    rhomin_active =  max( temp(set).^2 );
end
rhomax_active=min( temp(pattern).^2 );

if rhomin_active >= rhomax_active
    % No possible rhos, exit
    dv = Inf;
    return;
end

% Check rho
if (rho<rhomin_active-1e-6)|(rho>rhomax_active+1e-6)
    disp('Error in computing dual value: inconsistent rho');
    dv = Inf;
    return;
end

% Otherwise, build dual solution matrix as a function of rho
Mtest=zeros(size(S));
pr=find(diag(S-rho)>0);
prset=intersect(pr,set);
if length(prset)>0
%     for j=prset
%         prjx=A(:,j)-(x'*A(:,j))*x;
%         Mtest=Mtest+rho*((A(:,j)'*A(:,j)-rho)/(rho-(A(:,j)'*x)^2))*prjx*prjx'/(prjx'*prjx);
%     end
%     for i=pattern
%         bix=A(:,i)*A(:,i)'*x-rho*x;
%         Mtest=Mtest+bix*bix'/(x'*bix);
%     end
    Ap=A(:,prset)-x*x'*A(:,prset); % same as above but faster...
    Mtest=Ap*diag(rho*(sum(A(:,prset).^2)-rho)./(sum(Ap.^2).*(rho-(x'*A(:,prset)).^2)))*Ap';
    Bix=A(:,pattern)*diag(x'*A(:,pattern))-rho*x*ones(1,length(pattern));
    Mtest=Mtest+Bix*diag(1./(x'*Bix)')*Bix';
    [tv,dv]=maxeig(Mtest); % NUMISSUE: Nan when rho is close to the boundary
else % All other variables can be excluded right away
    Bix=A(:,pattern)*diag(x'*A(:,pattern))-rho*x*ones(1,length(pattern));
    Mtest=Mtest+Bix*diag(1./(x'*Bix)')*Bix';
    [tv,dv]=maxeig(Mtest); % NUMISSUE: Nan when rho is close to the boundary
end