function SANDIinput = InitializeSANDIinput(ProjectMainFolder, Delta, smalldelta, SNR)

SANDIinput = struct; % Initialize the variable containing all the SANDI analysis info

%%%%%%%%%%%%%%%%%%%%%%%%%%% USER DEFINED INFO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SANDIinput.StudyMainFolder = ProjectMainFolder; % <---- USER DEFINED: Path to the main folder where the data is

SANDIinput.SNR = SNR; % <---- USER DEFINED: if a noisemap from MPPCA denoising is provided, leave it empty. If no noisemap from MPPCA denoising is available, provide the SNR computed on an individual representative b=0 image
SANDIinput.delta = Delta; %  <---- USER DEFINED: the diffusion gradient separation ( for PGSE sequence ) in ms. This is assumed the same for all the dataset within the same study.
SANDIinput.smalldel = smalldelta; % <---- USER DEFINED: the diffusion gradient duration ( for PGSE sequence ) in ms. This is assumed the same for all the dataset within the same study.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SANDIinput.Dsoma =  3; % in micrometers^2/ms, if empty it is set to by default to 3
SANDIinput.Din_UB =  3; % in micrometers^2/ms, if empty it is set to by default to 3
SANDIinput.Rsoma_UB = []; % in micrometers, if empty it is set to by default to a max value given the diffusion time and the set Dsoma
SANDIinput.De_UB = 3; % in micrometers^2/ms, if empty it is set to by default to 3

SANDIinput.Nset = 1e5; % Size of the training set. Recommended values: 1e4 for testing the performance and between 1e5 and 1e6 for the analysis. Do not use values < 1e4. Values >1e6 might lead to 'out of memory'. 
SANDIinput.MLmodel = 'RF'; % can be 'RF' for Random Forest; 'MLP' for multi-layer perceptron
SANDIinput.FittingMethod = 0; % can be '0': minimizes MSE between ground truth model parameters and ML predictions. It has higher precision but lower accuracy; or '1': minimizes MSE between NLLS estimates of model parameters and ML predictions. It has higher accuracy but lower precision.
SANDIinput.MLdebias = 0; % can be '0' or '1'. When '1', it will estimate slope and intercept from the prediciton vs ground truth relation (from training set) and correct the prediciton to follow the unit line. 

SANDIinput.FWHM = 0.001; % size of the 3D Gaussian smoothing kernel. If needed, this smooths the input data before analysis.

SANDIinput.UseDirectionAveraging = 1; % If set equal to 1, it calculates the powder-averaged signal as aritmetic mean over the directions instead of using the order zero SH.

SANDIinput.DoTestPerformances = 0; % If '1' then it compares the performances of ML estimation with NLLS estimation on unseen simulated signals and write HTML report
SANDIinput.diagnostics = 0; % can be '0' or '1'. When '1', figures will be plotted to help checking the results

SANDIinput.WithDot = 0; % if 1 add 'dot' compartment in case needed, for example, in some cases to process ex vivo data

SANDIinput.LogFileFilename = fullfile(ProjectMainFolder,'SANDI_analysis_LogFile.txt');
SANDIinput.LogFileID = fopen(SANDIinput.LogFileFilename,'w');
end

