function [y,x,u,bufm,obj]=sdprelax(Sigma,rho)
% Compute SDP relaxation of sparse PCA problem
A=chol(Sigma);
[m,n]=size(A);

Amat=[];bvec=[];objevc=[];
for i=1:n
    Amat=[Amat;[zeros(m^2,(i-1)*m^2),-eye(m^2),zeros(m^2,(n-i)*m^2+1)]];
    bvec=[bvec;zeros(m^2,1)];
    Amat=[Amat;[zeros(m^2,(i-1)*m^2),-eye(m^2),zeros(m^2,(n-i)*m^2+1)]];
    bvec=[bvec;vec(-A(:,i)*A(:,i)'+rho*eye(m))];
end
Amat=[Amat;kron(ones(1,n),eye(m^2)),-vec(eye(m))];
bvec=[bvec;zeros(m^2,1)];
objvec=[zeros(n*m^2,1);-1];

K.s=[m*ones(2*n,1);m]';
pars.fid=0;
[x,ysol]=sedumi(Amat',objvec,bvec,K,pars);
y=ysol(1:end-1);
obj=ysol(end);
bufm=reshape(kron(ones(1,n),eye(m^2))*y,m,m);
[V,D]=eig(bufm);
u=V(:,end);