function SANDIinput = TrainMachineLearningModel(SANDIinput)

%%%%%%%%%%%%%%%%%%%%%%%%%%% USER DEFINED INFO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the parameters to build the training set. This can (and should) change according to the acquisition!

% UB stands for upper bound. Model parameters values will be chosen randomly between the intervals:
% fsoma and fneurite = [0 1]
% Din and De = [Dsoma/12 Din_UB] and [Dsoma/12 De_UB],
% Rsoma = [1 Rsoma_UB]

SANDIinput.seed = 1; % for reproducibility

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train the ML model
if SANDIinput.WithDot==1
    SANDIinput = setup_and_run_model_training_with_dot(SANDIinput); % Funciton that train the ML model. For options 'RF' and 'MLP', the training performance (mean squared error as a function of #trees or training epochs, respectively) are also saved in 'train_perf', which is the second output.
else
    SANDIinput = setup_and_run_model_training(SANDIinput); % Funciton that train the ML model. For options 'RF' and 'MLP', the training performance (mean squared error as a function of #trees or training epochs, respectively) are also saved in 'train_perf', which is the second output.
end

if SANDIinput.DoTestPerformances==1
    SANDIinput = investigate_exchange_effectes_NEXI_SANDI_RicianNoise(SANDIinput); % it estimates the extent of potantial bias due to unaccounted inter-compartmental exchange using the NEXI model (https://doi.org/10.1016/j.neuroimage.2022.119277)
end
end