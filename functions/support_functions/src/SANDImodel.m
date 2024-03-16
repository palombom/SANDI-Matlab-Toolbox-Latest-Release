function S = SANDImodel(p,x,delta, smalldel, Dsoma)

Dis = Dsoma; 
fellipse = @(p,x) exp(-x.*p(2)).*sqrt(pi./(4.*x.*(p(1)-p(2)))).*erf(sqrt(x.*(p(1)-p(2))));
fstick = @(p,x) fellipse([p,0],x);
fsphere = @(p,x) exp(-my_murdaycotts(delta,smalldel,p,Dis,x) );
fball = @(p,x) exp(-p.*x);        

 %S =  p(2).*fstick(p(3),x) + ...
 %     p(1).*fsphere(p(4),x) + ...
 %     (1 - p(1) - p(2)).*fball(p(5),x);

 S =  cos(p(1)).^2.*fstick(p(3),x) + ...
       (1-cos(p(1)).^2).*cos(p(2)).^2.*fsphere(p(4),x) + ...
       (1 - cos(p(1)).^2 - (1-cos(p(1)).^2).*cos(p(2)).^2).*fball(p(5),x);

% S =  cos(p(1)).^2.*fstick(1.7,x) + ...
%      (1-cos(p(1)).^2).*cos(p(2)).^2.*fsphere(p(4),x) + ...
%      (1 - cos(p(1)).^2 - (1-cos(p(1)).^2).*cos(p(2)).^2).*fellipse([p(3),p(5).*p(3)],x);

end