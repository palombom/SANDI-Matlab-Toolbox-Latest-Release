function r = drchrnd(a,n)

% Draw samples from Dirichlet distribution

p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r./repmat(sum(r,2),1,p);

end