function SANDIinput = SANDI_batch_analysis(ProjectMainFolder, Delta, smalldelta)

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
%                |----> - ses-02
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
%          |---> - sub-02
%                |----> - ses-01
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.nii.gz
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bval
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bvec
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_mask.nii.gz
%                       |-----> sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_noisemap.nii.gz
%                |----> - ses-02
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
%                |----> - ses-02
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
% March 2023
% Email: palombom@cardiff.ac.uk

% Add the path to main and support functions used for SANDI analysis

addpath(genpath(fullfile(pwd, 'functions')));

%% Initialize analysis
disp('*****   SANDI analysis using Machine Learning based fitting method   ***** ')
SANDIinput = InitializeSANDIinput(ProjectMainFolder, Delta, smalldelta);

%% STEP 1 - Preprocess the data: calculate the spherical mean signal and estimate noise distributions
SANDIinput = ProcessAllDatasets(SANDIinput); % Start processing all the datasets, one by one

%% STEP 2 - Train the Machine Learning (ML) model
SANDIinput = TrainMachineLearningModel(SANDIinput);

%% STEP 3 - SANDI fit each subject
% Here each subject can be preprocessed to compute the direction averaged
% signal and then used for SANDI model estimation by inserting the code below within a for loop over each subject
SANDIinput = AnalyseAllDatasets(SANDIinput);

end