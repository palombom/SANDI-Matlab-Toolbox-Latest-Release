function SANDIinput = SANDI_batch_analysis(ProjectMainFolder, Delta, smalldelta, SNR)

% Main script to perform the SANDI analysis (on one or more datasets) using machine learning, as described in Palombo M. et al. Neuroimage 2020: https://doi.org/10.1016/j.neuroimage.2020.116835

% The code assumes that data are organized in the following way:
%
% - ProjectMainFolder
% |-> - derivatives
%     |--> - preprocessed
%          |---> - sub-01
%                |----> - ses-01
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.nii.gz
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bval
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bvec
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_mask.nii.gz
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_noisemap.nii.gz
%                   ...
%                |----> - ses-n
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.nii.gz
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bval
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bvec
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_mask.nii.gz
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_noisemap.nii.gz
%            ...
%          |---> - sub-n
%                |----> - ses-01
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.nii.gz
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bval
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bvec
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_mask.nii.gz
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_noisemap.nii.gz
%                   ...
%                |----> - ses-n
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.nii.gz
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bval
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bvec
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_mask.nii.gz
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_noisemap.nii.gz

% The OUTPUT of the analysis will be stored in a new folder
% 'ProjectMainFolder -> derivatives -> SANDI_analysis -> sub-XXX -> ses-XXX -> SANDI_Output'
% for each subject and session

% NOTE: several improvements have been introduced since the original work
% on Neuroimage 2020
% These comprises:
% 1. Calculation of the Spherical Mean signal using Spherical Harmonics
% fitting (zeroth-order SH coefficient)
% 2. The training set is built in a more accurate way:
%    (i) Neurite and soma signal fraction are now sampled to cover
%    uniformily the simplex fneurite + fsoma <=1
%    (ii) The noise is added to simulated signals in a more realistic way:
%    first the noiseless signal for a random fibre direction is simulated
%    using the SANDI model, then the Rician noise floor is added using the
%    RiceMean function with sigma as estimated by MPPCA (noisemap), if
%    provided, then Gaussian noise is added with sigma equal to the std. of
%    the residuals from the SH fit, finally the spherical mean signal is computed averaging the signal over all the directions.
%    (iii) The training can be done in two ways now: minimizing the MSE
%    between a) the ground truth model parameters used to simulate the training
%    set and the ML prediction, or b) the model parameters estimated by NLLS with Rician likelihood and the ML prediction

% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% March 2024
% Email: palombom@cardiff.ac.uk

% Add the path to main and support functions used for SANDI analysis

addpath(genpath(fullfile(pwd, 'functions')));

%% Initialize analysis
SANDIinput = InitializeSANDIinput(ProjectMainFolder, Delta, smalldelta, SNR); % Edit this function to change the default options of the SANDI Toolbox

disp('*****   SANDI analysis using Machine Learning based fitting method   ***** ')
dt = char(datetime("now"));
disp(['*****                          ' dt '                  ***** '])
fprintf(SANDIinput.LogFileID,'*****   SANDI analysis using Machine Learning based fitting method   ***** \n');
fprintf(SANDIinput.LogFileID,'*****                          %s                  ***** \n', dt);

%% STEP 1 - Preprocess the data: calculate the spherical mean signal and estimate noise distributions
SANDIinput = ProcessAllDatasets(SANDIinput); % Process all the datasets, one by one

%% STEP 2 - Train the Machine Learning (ML) model
SANDIinput = TrainMachineLearningModel(SANDIinput); % trains the ML model on synthetic data

%SANDIinput =
%investigate_exchange_effectes_NEXI_SANDI_RicianNoise(SANDIinput); % Runs
%tests to estimate the bias due to unaccounted exchange between SANDI
%compartments, using NEXI model https://doi.org/10.1016/j.neuroimage.2022.119277

% Saving the Training Set
Signals_train = SANDIinput.database_train_noisy;
Params_train = SANDIinput.params_train;
Performance_train = SANDIinput.train_perf;
Bvals_train = SANDIinput.model.bvals;
Sigma_mppca_train = SANDIinput.model.sigma_mppca;
Sigma_SHresiduals_train = SANDIinput.model.sigma_SHresiduals;

mkdir(fullfile(SANDIinput.StudyMainFolder, 'Report_ML_Training_Performance'));

save(fullfile(SANDIinput.StudyMainFolder, 'Report_ML_Training_Performance','TrainingSet.mat'), 'Signals_train',...
    'Params_train','Performance_train','Bvals_train', 'Sigma_mppca_train', 'Sigma_SHresiduals_train');

%% STEP 3 - SANDI fit each subject
SANDIinput = AnalyseAllDatasets(SANDIinput); % Analyse all the datasets, one by one

fclose(SANDIinput.LogFileID);
end
