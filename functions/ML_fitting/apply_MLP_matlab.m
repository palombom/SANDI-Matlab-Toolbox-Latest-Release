function mpgMean = apply_MLP_matlab(signal, trainedML, MLdebias)

% Apply pretrained Multi Layer Perceptron regressor
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 8th December 2021
% Email: palombom@cardiff.ac.uk

method = 0;

Mdl = trainedML.Mdl;
Slope = trainedML.Slope;
Intercept = trainedML.Intercept;

tic

if method == 1
    mpgMean = zeros(size(signal,1), 5, numel(Mdl));

    for j=1:numel(Mdl)
        net = Mdl{j};
        mpgMean(:,:,j) = net(signal');
        if MLdebias==1
            for i = [1,3,5]
                mpgMean(:,i,j) = (mpgMean(:,i,j) - Intercept(i,j))./Slope(i,j);
            end
        end
    end
        
    mpgMean = mean(mpgMean,3);
    mpgMean = mpgMean';

else
    
mpgMean = zeros(size(signal,1), size(Mdl,1), size(Mdl,2));

for j=1:size(Mdl,2)
    for i=1:size(Mdl,1)
        net = Mdl{i,j};
        mpgMean(:,i,j) = net(signal');
        if MLdebias==1
            if i==1 || i==3 || i==5
                mpgMean(:,i,j) = (mpgMean(:,i,j) - Intercept(i,j))./Slope(i,j);
            end
        end
    end
end

mpgMean = mean(mpgMean,3);

end

tt = toc;

fprintf('DONE - MLP fitted in %3.0f sec.\n', tt)

end
