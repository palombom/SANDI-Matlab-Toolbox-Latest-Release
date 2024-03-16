function mu = RiceMean(nu, sig)
% This code is from Jelle Veraart - NYU

if sig>0

x = - nu.^2 ./ (2*sig.^2);

I0 = besseli(0, x/2, 1).*exp(abs(real(x/2)));
I1 = besseli(1, x/2, 1).*exp(abs(real(x/2)));

K = exp(x/2).*(x.*I1  + (1-x).*I0);

mu = 1.2533 .* sig .* K;

nanlocs = find(isnan(mu) | ~isfinite(mu));
mu(nanlocs) = nu(nanlocs);

else
    mu = nu;
end

mu = mu./mu(1); % Provide signal normilized by the b=0 signal

end