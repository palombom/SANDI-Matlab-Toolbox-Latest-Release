function S = NEXI(b,t,tex,Da,De,Dp,f)

% full KM
D1 = @(x,b,Da) Da*b'*x.^2;
D2 = @(x,b,De,Dp) b'*Dp + (De-Dp)*b'*x.^2;
Discr = @(x,b,t,tex,f,Da,De,Dp) (t./tex.*(1-2*f)+D1(x,b,Da)-D2(x,b,De,Dp)).^2 + 4*t.^2./tex.^2.*f.*(1-f);
lp = @(x,b,t,tex,f,Da,De,Dp) (t./tex+D1(x,b,Da)+D2(x,b,De,Dp))/2 + sqrt(Discr(x,b,t,tex,f,Da,De,Dp))/2;
lm = @(x,b,t,tex,f,Da,De,Dp) (t./tex+D1(x,b,Da)+D2(x,b,De,Dp))/2 - sqrt(Discr(x,b,t,tex,f,Da,De,Dp))/2;
Pp = @(x,b,t,tex,Da,De,Dp,f) (f*D1(x,b,Da) + (1-f)*D2(x,b,De,Dp) - lm(x,b,t,tex,f,Da,De,Dp))./sqrt(Discr(x,b,t,tex,f,Da,De,Dp));
Pm = @(x,b,t,tex,Da,De,Dp,f) 1 - Pp(x,b,t,tex,Da,De,Dp,f);
M = @(x,b,t,tex,Da,De,Dp,f) Pp(x,b,t,tex,Da,De,Dp,f).*exp(-lp(x,b,t,tex,f,Da,De,Dp)) + Pm(x,b,t,tex,Da,De,Dp,f).*exp(-lm(x,b,t,tex,f,Da,De,Dp));

S = integral(@(x) M(x,b,t,tex,Da,De,Dp,f), 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true)';

end
