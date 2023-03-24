function trainedML = train_RF_matlab(database_train, params_train, n_trees)
% Train a Random Forest regressor for SANDI fitting
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 8th December 2021
% Email: palombom@cardiff.ac.uk

tic

trainedML = struct;
Mdl = cell(size(params_train,2),1);
MLprediction = zeros(size(params_train));
Rsq = zeros(size(params_train,2),1);
Slope = zeros(size(params_train,2),1);
Intercept = zeros(size(params_train,2),1);

rng(1);

parfor i=1:size(params_train,2)
    
    Mdl{i} = TreeBagger(n_trees,database_train,params_train(:,i),'Method','regression','OOBPrediction','On');
    MLprediction(:,i) = predict(Mdl{i},database_train);
    [Rsq(i),Slope(i),Intercept(i)] = regression(params_train(:,i)', MLprediction(:,i)');

end

trainedML.Mdl = Mdl;
trainedML.Rsq = Rsq;
trainedML.Slope = Slope;
trainedML.Intercept = Intercept;

tt = toc;

fprintf('DONE - RF trained in %3.0f sec.\n', tt)

end
