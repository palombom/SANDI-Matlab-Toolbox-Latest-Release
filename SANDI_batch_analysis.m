% Main script to perform the SANDI analysis (on one or more datasets) using machine learning, as described in Palombo M. et al. Neuroimage 2020: https://doi.org/10.1016/j.neuroimage.2020.116835

% The code assumes that data are organized in the following way:
%
% - StudyMainFolder
% |---> - Dataset_1
%       |---> dwi.nii.gz
%       |---> bvals.bval
%       |---> bvecs.bvec
%       |---> mask.nii.gz
%       |---> noisemap.nii.gz
% |---> - Dataset_2
%       |---> dwi.nii.gz
%       |---> bvals.bval
%       |---> bvecs.bvec
%       |---> mask.nii.gz
%       |---> noisemap.nii.gz
%    ...
% |---> - Dataset_n
%       |---> dwi.nii.gz
%       |---> bvals.bval
%       |---> bvecs.bvec
%       |---> mask.nii.gz
%       |---> noisemap.nii.gz

% The OUTPUT of the analysis will be stored in a new folder 'SANDI_Output'
% within the corresponding 'Dataset_xxx' folder

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

clear all
close all
clc

% Add the path to main and support functions used for SANDI analysis

addpath(genpath(fullfile(pwd, 'functions')));

SANDIinput = struct; % Initialize the variable containing all the SANDI analysis info

%% Initialize analysis

disp('*****   SANDI analysis using Machine Learning based fitting method   ***** ')

%%%%%%%%%%%%%%%%%%%%%%%%%%% USER DEFINED INFO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SANDIinput.StudyMainFolder = ; % <---- USER DEFINED: Path to the main folder where the data is

SANDIinput.SNR = []; % <---- USER DEFINED: if a noisemap from MPPCA denoising is provided, leave it empty. If no noisemap from MPPCA denoising is available, provide the SNR computed on an individual representative b=0 image
SANDIinput.delta = ; %  <---- USER DEFINED: the diffusion gradient separation ( for PGSE sequence ) in ms. This is assumed the same for all the dataset within the same study.
SANDIinput.smalldel = ; % <---- USER DEFINED: the diffusion gradient duration ( for PGSE sequence ) in ms. This is assumed the same for all the dataset within the same study.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SANDIinput.Nset = 1e4; % Size of the training set. Recommended values: 1e4 for testing the performance and between 1e5 and 1e6 for the analysis. Do not use values < 1e4. Values >1e6 might lead to 'out of memory'. 
SANDIinput.MLmodel = 'MLP'; % can be 'RF' for Random Forest (default); 'MLP' for multi-layer perceptron
SANDIinput.FittingMethod = 0; % can be '0': minimizes MSE between ground truth model parameters and ML predictions. It has higher precision but lower accuracy; or '1': minimizes MSE between NLLS estimates of model parameters and ML predictions. It has higher accuracy but lower precision.
SANDIinput.MLdebias = 1; % can be '0' or '1'. When '1', it will estimate slope and intercept from the prediciton vs ground truth relation (from training set) and correct the prediciton to follow the unit line. 

SANDIinput.FWHM = 0.001; % size of the 3D Gaussian smoothing kernel. If needed, this smooths the input data before analysis.

SANDIinput.DoTestPerformances = 1; % If '1' then it compares the performances of ML estimation with NLLS estimation on unseen simulated signals and write HTML report
SANDIinput.diagnostics = 1; % can be '0' or '1'. When '1', figures will be plotted to help checking the results

%% STEP 1 - Preprocess the data: calculate the spherical mean signal and estimate noise distributions
SANDIinput = ProcessAllDatasets(SANDIinput); % Start processing all the datasets, one by one

%% STEP 2 - Train the Machine Learning (ML) model
SANDIinput = TrainMachineLearningModel(SANDIinput);

%% STEP 3 - SANDI fit each subject
% Here each subject can be preprocessed to compute the direction averaged
% signal and then used for SANDI model estimation by inserting the code below within a for loop over each subject
SANDIinput = AnalyseAllDatasets(SANDIinput);
