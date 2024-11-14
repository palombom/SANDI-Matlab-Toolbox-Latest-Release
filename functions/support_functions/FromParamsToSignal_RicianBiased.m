function S = FromParamsToSignal_RicianBiased(params, b, delta, smalldelta, sigma)

% params are the SANDI estimated model parameters, specifically:
%
% params(:,1) = fenurite
% params(:,2) = fsoma
% params(:,3) = Din in um2/ms
% params(:,4) = Rsoma in um
% params(:,5) = De in um2/ms
%
% b: the b value, in ms/um2
% delta: the diffusion gradient pulse separation in ms
% smalldelta: the diffusion gradient pulse duration in ms
% sigma: noise level as estimated by MPPCA, normalized by the b=0 signal -
% it is a vector of length equal to size(params,1).
%
%
% Author: Marco Palombo
% October 2024


frandomellipse = @(p,x) exp(-x.*p(2)).*sqrt(pi./(4.*x.*(p(1)-p(2)))).*erf(sqrt(x.*(p(1)-p(2))));
frandomstick = @(p,x) frandomellipse([p,0],x);
Dis = 3; % intra-soma diffusivity fixed to 3 um2/ms
fsphere = @(p,x) exp(-my_murdaycotts(delta,smalldelta,p,Dis,x));
fball = @(p,x) exp(-p.*x);

fsandi = @(p,x) p(1).*frandomstick(p(3), x) + p(2).*fsphere(p(4),x) + (1 - p(1) - p(2)).*fball(p(5),x);

S = zeros(size(params,1), numel(b));

parfor i =1:size(params,1)
    S(i,:) = RiceMean(fsandi(params(i,:), b), sigma(i));
end

end

