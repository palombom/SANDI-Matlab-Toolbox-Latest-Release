function SANDIinput = InitializeSANDIinput(ProjectMainFolder, Delta, smalldelta)

SANDIinput = struct; % Initialize the variable containing all the SANDI analysis info

%%%%%%%%%%%%%%%%%%%%%%%%%%% USER DEFINED INFO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SANDIinput.StudyMainFolder = ProjectMainFolder; % <---- USER DEFINED: Path to the main folder where the data is

SANDIinput.SNR = []; % <---- USER DEFINED: if a noisemap from MPPCA denoising is provided, leave it empty. If no noisemap from MPPCA denoising is available, provide the SNR computed on an individual representative b=0 image
SANDIinput.delta = Delta; %  <---- USER DEFINED: the diffusion gradient separation ( for PGSE sequence ) in ms. This is assumed the same for all the dataset within the same study.
SANDIinput.smalldel = smalldelta; % <---- USER DEFINED: the diffusion gradient duration ( for PGSE sequence ) in ms. This is assumed the same for all the dataset within the same study.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SANDIinput.Nset = 1e4; % Size of the training set. Recommended values: 1e4 for testing the performance and between 1e5 and 1e6 for the analysis. Do not use values < 1e4. Values >1e6 might lead to 'out of memory'. 
SANDIinput.MLmodel = 'RF'; % can be 'RF' for Random Forest; 'MLP' for multi-layer perceptron
SANDIinput.FittingMethod = 0; % can be '0': minimizes MSE between ground truth model parameters and ML predictions. It has higher precision but lower accuracy; or '1': minimizes MSE between NLLS estimates of model parameters and ML predictions. It has higher accuracy but lower precision.
SANDIinput.MLdebias = 1; % can be '0' or '1'. When '1', it will estimate slope and intercept from the prediciton vs ground truth relation (from training set) and correct the prediciton to follow the unit line. 

SANDIinput.FWHM = 0.001; % size of the 3D Gaussian smoothing kernel. If needed, this smooths the input data before analysis.

SANDIinput.UseDirectionAveraging = 1; % It calculates the powder-averaged signal as aritmetic mean over the directions instead of using the order zero SH.

SANDIinput.DoTestPerformances = 1; % If '1' then it compares the performances of ML estimation with NLLS estimation on unseen simulated signals and write HTML report
SANDIinput.diagnostics = 1; % can be '0' or '1'. When '1', figures will be plotted to help checking the results

end

