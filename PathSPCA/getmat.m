function M=getmat(u,a,x)
M=u(1)*x*x'+u(2)*(x*a'+a*x')+u(3)*a*a';