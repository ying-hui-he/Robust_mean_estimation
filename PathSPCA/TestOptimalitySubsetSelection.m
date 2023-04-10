function isopt =  TestOptimalitySubsetSelection(X,y,subset)
% check optimality of solutions of subset selection by transforming it into SPCA
[temp,mu0] =maxeig(X'*X);
n = size(X,2);
mu0 = mu0 * y'*y;
s0 = y'*X(:,subset)*inv(X(:,subset)'*X(:,subset))*X(:,subset)'*y;
S = X'*y*y'*X-s0*X'*X+mu0*eye(n);
trS = trace(S);
S = S / trS;
S=(S+S')/2;

S0=S;
[d,ix]=sort(diag(S),'descend');S=S(ix,ix);
rix = 1:n;
rix(ix)=1:n;
subset0 = rix(subset);
A=chol(S);

[isopt, rho] = TestOptimalityDual(A,S,subset0)
optimal(i)=isopt;