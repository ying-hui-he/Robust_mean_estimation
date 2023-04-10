function [subres,sol,vars,rhobreaks,res]=FullPathGreedyFB(A,S,k,depth,algo)
% Given decomposition, compute full greedy path
% Warning: diagonal of S should be decreasing!
% DEBUG: sol and rhobreaks should be fixed

% Process inputs
n=size(S,1);
if nargin<3
    k=n-1;
else
    k=min(k,n-1);
end
if nargin<4
    depth=0;
end
if nargin<5
    algo=0;
end


% Initialize path
subset=[1];subres=[subset';zeros(n-length(subset),1)];
res=[];rhobreaks=[sum(A(:,1).^2)];sol=[];vars=[];

% Initialize forward/backward search params
i=1;iback=1;dirct=1; 

% Scan path
while i<=k
    if dirct<0 % Go backward
        if algo==0 % Approx greedy algo
            % Compute solution at current subset
            [v,mv]=maxeig(S(subset,subset));
            vsol=zeros(n,1);vsol(subset)=v;
            % Compute x at current subset
            x=A(:,subset)*v;x=x/norm(x);
            res=[res,x];
            % Compute next rho breakpoint
            set=1:n;set(subset)=[];
            vals=(x'*A(:,subset)).^2;
            [rhomax,vpos]=min(vals);
            rhobreaks(end)=[];sol=sol(:,1:end-1);vars(end)=[];
            subset(vpos)=[];subres=[subres(:,1:end-2),[subset';zeros(n-length(subset),1)]];
            iback=iback-1;
        else % Use pure greedy algo instead
            subres=subres(:,1:end-2);
            vbuf=[];
            for j=1:length(subset)
                bufset=subset;bufset(j)=[];
                [v,mv]=maxeig(S(bufset,bufset));
                vbuf=[vbuf;mv];
            end
            [m,idx]=max(vbuf);vars(end)=[];
            subset(min(idx))=[];subres=[subres,[subset';zeros(n-length(subset),1)]];
            iback=iback-1;
        end
    else % Go forward
        if algo==0 % Approx greedy algo
            % Compute solution at current subset
            [v,mv]=maxeig(S(subset,subset));
            vsol=zeros(n,1);vsol(subset)=v;
            sol=[sol,vsol];vars=[vars,mv];
            % Compute x at current subset
            x=A(:,subset)*v;x=x/norm(x);
            res=[res,x];
            % Find point to remove
            set=1:n;set(subset)=[];
            vals=(x'*A(:,set)).^2;
            [rhomax,vpos]=max(vals);
            rhobreaks=[rhobreaks;rhomax];
            subset=[subset,set(vpos)];subres=[subres,[subset';zeros(n-length(subset),1)]];
            iback=iback+1;
        else % Use pure greedy algo instead
            [v,mv]=maxeig(S(subset,subset));
            vars=[vars,mv];
            set=1:n;set(subset)=[];vbuf=[];
            for j=set
                bufset=union(subset,j);[v,mvbuf]=maxeig(S(bufset,bufset));
                vbuf=[vbuf;mvbuf];
            end
            [mv,idx]=max(vbuf);
            subset=[subset,set(min(idx))];subres=[subres,[subset';zeros(n-length(subset),1)]];
            iback=iback+1;
        end
    end
    %txt=fprintf('i: %d  iback: %d  #vars: %d',i,iback,length(vars));disp(txt);
    if (depth>0)&(i>=3)
        if iback>i % Control search direction
            i=i+1;dirct=-1;
        elseif iback<=max(i-depth,2)
            dirct=1;
        end
    else
        i=i+1;
    end
end
if k==n-1
    [v,mv]=maxeig(S);
    vars=[vars,mv];
end