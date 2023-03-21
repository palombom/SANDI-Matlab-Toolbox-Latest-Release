function mpgMean = apply_RF_matlab(signal, Mdl)
% Apply pretrained Random Forest regressor
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 8th December 2021
% Email: palombom@cardiff.ac.uk

tic

mpgMean = zeros(size(signal,1), numel(Mdl));

parfor i=1:numel(Mdl)
    mpgMean(:,i) = predict(Mdl{i},signal);
end

tt = toc;

fprintf('DONE - RF fitted in %3.0f sec.\n', tt)

end