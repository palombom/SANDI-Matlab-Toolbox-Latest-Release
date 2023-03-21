function [Mdl, training_performances] = train_MLP_matlab(database_train, params_train, n_layers, n_neurons, n_MLPs)

% Train a Multi Layer Perceptron regressor for SANDI fitting
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 4th August 2022
% Email: palombom@cardiff.ac.uk

tic

method = 0; % If set to '0' it will create and train an MLP for each model parameter. If '1' it will create and train a single MLP for the prediciton of all the model parameters.

net_structure = zeros(1,n_layers);

for i=1:n_layers
    net_structure(i) = n_neurons;
end

rng(1);

if method == 1
    Mdl = cell(n_MLPs,1);
    training_performances = cell(n_MLPs,1);

    for j = 1:n_MLPs

        Mdl{j} = feedforwardnet(net_structure, 'trainlm');
        Mdl{j}.performParam.regularization = 0;
        Mdl{j}.trainParam.showWindow = false;
        Mdl{j}.trainParam.showCommandLine = false;
        Mdl{j}.trainParam.max_fail = 10;
                    nlayers = numel(Mdl{j}.layers);
            for ii=1:nlayers-1
                Mdl{j}.layers{ii}.transferFcn = 'logsig'; % Keep better the output is within rage [0 1]
            end

        [Mdl{j}, training_performances{j}] = train(Mdl, database_train', params_train');

        disp(['   - MLP ' num2str(j) '/' num2str(n_MLPs) ' trained'])
    end
else

    training_performances = cell(size(params_train,2),n_MLPs);
    Mdl = cell(size(params_train,2),n_MLPs);

    for j = 1:n_MLPs


        for i = 1:size(params_train,2)
            Mdl{i,j} = feedforwardnet(net_structure, 'trainlm');
            Mdl{i,j}.performParam.regularization = 0;
            Mdl{i,j}.trainParam.showWindow = false;
            Mdl{i,j}.trainParam.showCommandLine = false;
            Mdl{i,j}.trainParam.max_fail = 10;

            nlayers = numel(Mdl{i,j}.layers);
            for ii=1:nlayers-1
                Mdl{i,j}.layers{ii}.transferFcn = 'logsig'; % Keep better the output is within rage [0 1]
            end

        end

        parfor i=1:size(params_train,2)

            [Mdl{i,j}, training_performances{i,j}] = train(Mdl{i,j}, database_train', params_train(:,i)');

        end

        disp(['   - MLP ' num2str(j) '/' num2str(n_MLPs) ' trained'])

    end


end

tt = toc;

fprintf('   - DONE! MLP trained in %3.0f sec.\n', tt)

end
