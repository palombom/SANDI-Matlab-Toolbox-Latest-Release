function Mdl = train_RF_matlab(database_train, params_train, n_trees)
% Train a Random Forest regressor for SANDI fitting
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 8th December 2021
% Email: palombom@cardiff.ac.uk

tic

Mdl = cell(size(params_train,2),1);

rng(1);

parfor i=1:size(params_train,2)
    
    Mdl{i} = TreeBagger(n_trees,database_train,params_train(:,i),'Method','regression','OOBPrediction','On');

end

tt = toc;

fprintf('DONE - RF trained in %3.0f sec.\n', tt)

end