function mpgMean = apply_RF_matlab(signal, trainedML, MLdebias)
% Apply pretrained Random Forest regressor
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 8th December 2021
% Email: palombom@cardiff.ac.uk

tic

Mdl = trainedML.Mdl;
Slope = trainedML.Slope;
Intercept = trainedML.Intercept;

mpgMean = zeros(size(signal,1), numel(Mdl));

parfor i=1:numel(Mdl)
    mpgMean(:,i) = predict(Mdl{i},signal);
    if MLdebias==1
        if i==1 || i==3 || i==5
            mpgMean(:,i) = (mpgMean(:,i) - Intercept(i))./Slope(i);
        end
    end
end

tt = toc;

fprintf('DONE - RF fitted in %3.0f sec.\n', tt)

end
