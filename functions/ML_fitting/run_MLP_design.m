function [Mdl, train_perf, mse, ss, uu] = run_MLP_design(SANDIinput)

% Main script to setup and train the Random
% Forest (RF) or multi-layers perceptron (MLP) regressors used to fit the
% SANDI model
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 4th August 2022
% Email: palombom@cardiff.ac.uk

% Read the input
data_filename = SANDIinput.data_filename;
mask_filename = SANDIinput.mask_filename;
bvalues_filename = SANDIinput.bvalues_filename;
bvecs_filename = SANDIinput.bvecs_filename ;
delta = SANDIinput.delta ;
smalldel = SANDIinput.smalldel ;
noisemap_filename = SANDIinput.noisemap_filename ;
SNR = SANDIinput.SNR ;
output_folder = SANDIinput.output_folder;
Dsoma = SANDIinput.Dsoma;
Din_UB = SANDIinput.Din_UB ;
Rsoma_UB = SANDIinput.Rsoma_UB ;
De_UB = SANDIinput.De_UB ;
seed = SANDIinput.seed ;
FittingMethod = SANDIinput.FittingMethod ;
noiseModel = SANDIinput.noiseModel ;

rng(seed) % for reproducibility

if ~isempty(output_folder)
    mkdir(output_folder) % create the output folder
end

if isempty(Dsoma)
    Dsoma = 3;
end
if isempty(Din_UB)
    Din_UB = 3;
end
if isempty(Rsoma_UB)
    Rsoma_UB = round(sqrt(6*Dsoma*(delta-smalldel/3))*0.7); % Authomatically set the Rsoma to a max value given the diffusion time and the set Dsoma.
end
if isempty(De_UB)
    De_UB = 3;
end

if isempty(FittingMethod)
    FittingMethod=0;
end

if isempty(noiseModel)
    noiseModel='gaussian-RandomSticks';
end

% MLP and sample sizes to test

sampleSizes = [1e4 1e5];
MLPunits = [10 20 30 40 50];

[ss, uu] = ndgrid(sampleSizes, MLPunits);
ss = ss(:); uu = uu(:);

Mdl = cell(numel(ss),1);
train_perf = cell(numel(ss),1);
mse = zeros(numel(ss),5);

titles = {'fneurite', 'fsoma', 'Din', 'Rsoma', 'De'};

figure,

for i=1:numel(ss)
    disp(['* Testing configuration ' num2str(i) '/' num2str(numel(ss))])

%% Build the model and the corresponding training set

% Build the SANDI model to fit (as in the Neuroimage 2020) sampling the signal fraction from a Dirichlet distribution, to guarantee that they sum up to 1 and cover uniformely the area of the triangle defined by them

model = struct;
model.Nset = ss(i); % After some testing, between 1e4 and 1e5 is large enough to converge to satisfactory performances. By default we choose 1e4 to keep the training faster, but it can and should be changed according to the dataset.

% If a noisemap is provided, it will use the distribution of noise variances within the volume to train the model. If a noisemap is not provided, it will use the user defined SNR.

if ~isempty(noisemap_filename)

    noisemap = normalize_noisemap(noisemap_filename, data_filename, mask_filename, bvalues_filename); % Load the noisemap previously obtained by MP-PCA denoising and normalizes it by dividing it for the b=0 image.

    sigma = noisemap(:);
    sigma = sigma(sigma>0);
    sigma(sigma<0) = 0;
    sigma(sigma>1) = 1;
    
    sigma = median(sigma); % Use the median of the noisemap within the brain region as sigma
    sigma = sigma.*ones(model.Nset,1);
else
    sigma = 1/SNR;
    sigma = sigma.*ones(model.Nset,1);
end
model.sigma = sigma;

model.delta = delta;
model.smalldel = smalldel;
model.Dsoma = Dsoma;
model.noise = noiseModel;
model.paramsrange = [0 1; 0 1; Dsoma/12 Din_UB; 1 Rsoma_UB; Dsoma/12 De_UB];
model.Nparams = size(model.paramsrange,1);
model.boost = FittingMethod;

% Build the training set
disp(['   - Building training set: #samples = ' num2str(model.Nset)])
[~, database_train_noisy, params_train, ~, ~] = build_training_set(model, bvalues_filename, bvecs_filename, output_folder);

        %% MLP train

        % --- Using Matlab
        n_MLPs = 1; % Training is performed using n_MLPs randomly initiailized for each model parameter. The final prediciton is the average prediciton among the n_MLPS. This shuold mitigate issues with local minima during training according to the "wisdom of crowd" principle.
        n_layers = 3;
        n_units = uu(i); % 5*min(size(database_train,1),size(database_train,2)); % Recommend network between 3 x number of b-shells and 5 x number of b-shells

        disp(['   - Training ' num2str(n_MLPs) ' MLP(s) with ' num2str(n_layers) ' hidden layer(s) and ' num2str(n_units) ' units per layer for each model parameter...'])

        [Mdl{i}, train_perf{i}] = train_MLP_matlab(database_train_noisy, params_train, n_layers, n_units, n_MLPs);

        % save([output_folder '/trained_MLPmodel.mat'], 'Mdl', '-v7.3')

        
        for j=1:5
            mse(i,j) = train_perf{i}{j}.best_perf;
        end
        
        plot(1:numel(ss), sum(log10(mse),2), '.-'), ylabel('Sum of model parameters log10(mse)'), xlabel('Network configuration'), legend(titles)
        
        hold off
end


end