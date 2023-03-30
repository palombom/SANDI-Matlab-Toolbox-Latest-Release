function [sphericalMean, sigma] = SphericalMeanFromSH(dirs, y, order)

y(y<eps) = eps;
[coef, X] = llsFitSH(dirs, y,order);

sphericalMean = coef(1, :) ./ sqrt(pi*4);
residuals = X*coef - y;
w = ones(size(residuals));

[n, ~] = size(y); m = size(coef, 1);
sigma = sqrt(n./(n-m)).*1.4826.*mad(residuals.*w, 1, 1);

end







