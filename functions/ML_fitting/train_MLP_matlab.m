function trainedML = train_MLP_matlab(database_train, params_train, n_layers, n_neurons, n_MLPs, SANDIinput)

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

trainedML = struct;

if method == 1
    Mdl = cell(n_MLPs,1);
    training_performances = cell(n_MLPs,1);
    
    MLprediction = zeros(size(params_train,1),size(params_train,2), n_MLPs);
    Rsq = zeros(size(params_train,2),n_MLPs);
    Slope = zeros(size(params_train,2),n_MLPs);
    Intercept = zeros(size(params_train,2),n_MLPs);
    
    
    for j = 1:n_MLPs
        
        Mdl{j} = feedforwardnet(net_structure);
        %Mdl{j}.performParam.regularization = 0;
        Mdl{j}.trainParam.showWindow = false;
        Mdl{j}.trainParam.showCommandLine = false;
        %Mdl{j}.trainParam.max_fail = 10;
        %Mdl{j}.inputs{1}.processFcns = {'mapminmax'};
        
        %nlayers = numel(Mdl{j}.layers);
        %for ii=1:nlayers-1
        %    Mdl{j}.layers{ii}.transferFcn = 'logsig'; % Keep better the output is within rage [0 1]
        %end
        
        %[Mdl{j}, training_performances{j}] = train(Mdl, database_train', params_train', 'useGpu','yes','useParallel', 'yes');
        [Mdl{j}, training_performances{j}] = train(Mdl, database_train', params_train', 'useParallel', 'yes');
        
        MLprediction(:,:,j) = Mdl(database_train')';
        
        for i = 1:size(params_train,2)
            beta = fitlm(params_train(:,i)', MLprediction(:,i)');
            Slope(i) = beta.Coefficients.Estimate(2);
            Intercept(i) = beta.Coefficients.Estimate(1);
            Rsq(i) = beta.Rsquared.Ordinary;
        end
        
        disp(['   - MLP ' num2str(j) '/' num2str(n_MLPs) ' trained'])
    end
else
    
    training_performances = cell(size(params_train,2),n_MLPs);
    Mdl = cell(size(params_train,2),n_MLPs);
    
    MLprediction = zeros(size(params_train,1),size(params_train,2), n_MLPs);
    Rsq = zeros(size(params_train,2),n_MLPs);
    Slope = zeros(size(params_train,2),n_MLPs);
    Intercept = zeros(size(params_train,2),n_MLPs);
    
    for j = 1:n_MLPs
        
        
        for i = 1:size(params_train,2)
            Mdl{i,j} = feedforwardnet(net_structure);
            %Mdl{i,j}.performParam.regularization = 0;
            Mdl{i,j}.trainParam.showWindow = false;
            Mdl{i,j}.trainParam.showCommandLine = false;
            %Mdl{i,j}.trainParam.max_fail = 10;
            %Mdl{i,j}.inputs{1}.processFcns = {'mapminmax'};
            
            %nlayers = numel(Mdl{i,j}.layers);
            %for ii=1:nlayers-1
            %    Mdl{i,j}.layers{ii}.transferFcn = 'logsig'; % Keep better the output is within rage [0 1]
            %end
            
        end
        
        for i=1:size(params_train,2)
            disp(['   - MLP for model parameter ' num2str(i) '/' num2str(size(params_train,2)) ' training'])
            fprintf(SANDIinput.LogFileID,['   - MLP for model parameter ' num2str(i) '/' num2str(size(params_train,2)) ' training\n']);
            
            %[Mdl{i,j}, training_performances{i,j}] = train(Mdl{i,j}, database_train', params_train(:,i)', 'useGpu','yes','useParallel', 'yes');
            [Mdl{i,j}, training_performances{i,j}] = train(Mdl{i,j}, database_train', params_train(:,i)', 'useParallel', 'yes');
            MLprediction(:,i,j) = Mdl{i,j}(database_train');
            beta = fitlm(params_train(:,i)', MLprediction(:,i)');
            Slope(i) = beta.Coefficients.Estimate(2);
            Intercept(i) = beta.Coefficients.Estimate(1);
            Rsq(i) = beta.Rsquared.Ordinary;
            
        end
        
        disp(['   - MLP ' num2str(j) '/' num2str(n_MLPs) ' trained'])
        fprintf(SANDIinput.LogFileID,['   - MLP for model parameter ' num2str(i) '/' num2str(size(params_train,2)) ' training\n']);
        
    end
    
    
end

trainedML.Mdl = Mdl;
trainedML.training_performances = training_performances;
trainedML.Rsq = Rsq;
trainedML.Slope = Slope;
trainedML.Intercept = Intercept;

tt = toc;

disp(['DONE - MLP training took ' num2str(round(tt)) ' sec.'])
fprintf(SANDIinput.LogFileID,['DONE - MLP training took ' num2str(round(tt)) ' sec.\n']);

end
