function [database_train, database_train_noisy, params_train, sigma_mppca, bunique, Ndirs_per_Shell] = build_training_set(SANDIinput)

% Builds the dataset for supervised training of the machine learning models
% for SANDI fitting.
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 4th August 2022
% Email: palombom@cardiff.ac.uk

tic

%% Build the training set


bval_filename = SANDIinput.bvalues_filename;
bvec_filename = SANDIinput.bvecs_filename;
output_folder = SANDIinput.output_folder;

model = SANDIinput.model;

Nparams = model.Nparams;
sigma_mppca = model.sigma_mppca;
sigma_SHresiduals = model.sigma_SHresiduals;
Nset = model.Nset;
delta = model.delta;
smalldel = model.smalldel;

params = zeros(Nset, Nparams);

 T = drchrnd([1 1 1], Nset);
 params(:,1:2) = T(:,1:2);
   
% Sample the other model paramters from a uniform distribution

  for i = 3:Nparams
      params(:,i) = (model.paramsrange(i,2)-model.paramsrange(i,1)).*rand(Nset,1) + model.paramsrange(i,1);
  end

Dis = model.Dsoma;

fstick = @(p,x,costheta) exp(-x.*p(1).*costheta.^2);
fsphere = @(p,x) exp(-my_murdaycotts(delta,smalldel,p,Dis,x) );
fball = @(p,x) exp(-p.*x);

f = @(p, x, costheta) p(1).*fstick(p(3),x,costheta) + ...
    p(2).*fsphere(p(4),x) + ...
    (1 - p(1) - p(2)).*fball(p(5),x);

% Load bvals and bvecs

bvals = importdata(bval_filename);
bvals = round(bvals/100).*100;
bvals(bvals==0) = 1E-6;
bunique = unique(bvals);

Ndirs_per_Shell = zeros(numel(bunique),1);
for i = numel(bunique)
    Ndirs_per_Shell(i) = sum(bvals == bunique(i));
end

bvecs = importdata(bvec_filename); % Read bvecs
bvecs = bvecs./dot(bvecs, bvecs);  % Normalize bvecs
bvecs(isnan(bvecs)) = 0;
bvecs(isinf(bvecs)) = 0;

disp(['   - Generating ' num2str(Nset) ' random fibre directions'])
phi = rand(Nset,1).*2.*pi;
u = 2.*rand(Nset,1)-1; % cos(theta): ranging from -1 and +1
fibre_orientation = [sqrt(1 - u.^2).*cos(phi), sqrt(1 - u.^2).*sin(phi), u]';

costheta = zeros(Nset,numel(bvals));

disp('   - Calculating angles between fibres and gradients')

parfor i=1:Nset
    costheta(i,:) = dot(repmat(fibre_orientation(:,i), [1 numel(bvals)]), bvecs);
end

database_dir = zeros(Nset, numel(bvals));
database_dir_with_rician_bias = zeros(Nset, numel(bvals));
database_dir_with_rician_bias_noisy = zeros(Nset, numel(bvals));

disp(['   - Calculating signals per diffusion gradient direction and add Rician bias following sigma distribution from MPPCA, with median SNR = ' num2str(nanmedian(1./sigma_mppca)) ' to the signal for each direction'])
disp(['   - Adding Gaussian noise following the distribution from SH residuals, with median SNR = ' num2str(nanmedian(1./sigma_SHresiduals)) ' to the signal for each direction'])

parfor i=1:Nset
    database_dir(i,:) = f(params(i,:), bvals./1000, costheta(i,:));
    database_dir_with_rician_bias(i,:) = RiceMean(database_dir(i,:), sigma_mppca(i));
    database_dir_with_rician_bias_noisy(i,:) =  database_dir_with_rician_bias(i,:) + sigma_SHresiduals(i).*randn(size(database_dir_with_rician_bias(i,:)));
end

database_train_noisy = zeros(size(params,1), numel(bunique));
database_train = zeros(size(params,1), numel(bunique));

% Identify b-shells and direction-average per shell
disp('   - Direction-averaging the signals')
for i=1:numel(bunique)
    database_train_noisy(:,i) = nanmean(database_dir_with_rician_bias_noisy(:,bvals==bunique(i)),2);
    database_train(:,i) = nanmean(database_dir(:,bvals==bunique(i)),2);
end

params_train = params;

database_train_noisy = database_train_noisy./database_train_noisy(:,1); % Normalize by the b=0

bunique = bunique./1000;

if ~isempty(output_folder), save(fullfile(output_folder, 'database_training_set.mat'), 'database_train', 'params_train'); end

tt = toc;

disp(['   - DONE! Set built in ' num2str(round(tt)) ' sec.'])

end