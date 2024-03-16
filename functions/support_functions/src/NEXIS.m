function S = NEXIS(p,b,delta,smalldel,Dsoma)

% p(1) = fsoma
% p(2) = fneurite': fneurite = (1-p(1))*p(2)
% p(3) = Din
% p(4) = Rsoma
% p(5) = De
% p(6) = tex

Dis = Dsoma;
t = delta - smalldel./3;
Ssphere = @(p,x) exp( -my_murdaycotts(delta,smalldel,p(4),Dis,x) );
S = p(1).*Ssphere(p,b) + (1-p(1)).*NEXI(b,t,p(6),p(3),p(5),p(5),p(2));

end