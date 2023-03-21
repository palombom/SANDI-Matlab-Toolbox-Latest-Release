function SANDIinput = TrainMachineLearningModel(SANDIinput)

%%%%%%%%%%%%%%%%%%%%%%%%%%% USER DEFINED INFO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the parameters to build the training set. This can (and should) change according to the acquisition!

% UB stands for upper bound. Model parameters values will be chosen randomly between the intervals:
% fsoma and fneurite = [0 1]
% Din and De = [Dsoma/12 Din_UB] and [Dsoma/12 De_UB],
% Rsoma = [1 Rsoma_UB]

SANDIinput.Dsoma =  []; % in micrometers^2/ms, if empty it is set to by default to 3
SANDIinput.Din_UB =  []; % in micrometers^2/ms, if empty it is set to by default to 3
SANDIinput.Rsoma_UB = []; % in micrometers, if empty it is set to by default to a max value given the diffusion time and the set Dsoma
SANDIinput.De_UB = []; % in micrometers^2/ms, if empty it is set to by default to 3

SANDIinput.seed = 1; % for reproducibility

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train the ML model
SANDIinput = setup_and_run_model_training(SANDIinput); % Funciton that train the ML model. For options 'RF' and 'MLP', the training performance (mean squared error as a function of #trees or training epochs, respectively) are also saved in 'train_perf', which is the second output.

end