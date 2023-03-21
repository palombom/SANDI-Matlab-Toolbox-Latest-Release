function SANDIinput = make_direction_average(SANDIinput)

% Calculate the direction-average of the data 
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 8th December 2021
% Email: palombom@cardiff.ac.uk

data_filename = SANDIinput.data_filename;
mask_filename = SANDIinput.mask_filename;
bvalues_filename = SANDIinput.bvalues_filename;
bvecs_filename = SANDIinput.bvecs_filename ;
noisemap_mppca_filename = SANDIinput.noisemap_mppca_filename ; % Load the distribution of noise sigmas from MPPCA to determine the Rician noise floor bias to add during training
output_folder = SANDIinput.output_folder ;
FWHM = SANDIinput.FWHM ;
SNR = SANDIinput.SNR ;

disp(['* Computing the Spherical Mean for dataset: ' data_filename])


if isempty(noisemap_mppca_filename)
    disp('    Noisemap was not provided; using the provided average SNR')
end

diagnostics = SANDIinput.diagnostics ;

tic

mkdir(output_folder)

% Load the imaging data
disp('   - Loading the data')

tmp = load_untouch_nii(mask_filename);
mask = double(tmp.img(:));

if ~isempty(noisemap_mppca_filename)
    tmp = load_untouch_nii(noisemap_mppca_filename);
    tmp_img = double(tmp.img(:));
    sigma_mppca = tmp_img(mask==1)';
end

tmp = load_untouch_nii(data_filename);
I = double(abs(tmp.img));
tmp.img = [];

[sx, sy, sz, vols] = size(I);

sigma = FWHM/(2*sqrt(2*log(2)));

for i=1:vols
    I(:,:,:,i) = imgaussfilt3(squeeze(I(:,:,:,i)), sigma);
end

% Load bvals and bvecs
bvecs = importdata(bvecs_filename);
bvals = importdata(bvalues_filename);
bvals = round(bvals/100).*100;

bunique = unique(bvals);

Save = zeros(sx, sy, sz, numel(bunique));
S0mean = nanmean(double(I(:,:,:,bvals<=100)),4);
S0mean_vec = S0mean(:);
S0mean_vec = S0mean_vec(mask==1);

% Identify b-shells and direction-average per shell
estimated_sigma = [];
for i=1:numel(bunique)
    if i==1

        Save(:,:,:,i) = S0mean;
        if isempty(noisemap_mppca_filename)

                sigma_mppca = 1./SNR.*S0mean_vec';
                estimated_sigma = [estimated_sigma; sigma_mppca];
        else

            estimated_sigma = [estimated_sigma; nan(1,sum(mask))];
        end

    elseif i>1

            disp(['   - Calculating powder average signal by estimating the zero-order SH - shell ' num2str(i-1) ' of ' num2str(numel(bunique)-1)])
            dirs = bvecs(:,bvals==bunique(i))';
            y_tmp = I(:,:,:,bvals==bunique(i));
            y_tmp = reshape(y_tmp,[sx*sy*sz, size(y_tmp,4)])';
            y = y_tmp(:,mask==1);
            order = 4; % We expect datasets for SANDI to have at least 2 b-shells at b>3 ms/um^2. Given the complexity of the ODF at such high b, we recommend order between 4 and 6. By default we set it to 4.
            [Save_tmp, sigma_tmp] = SphericalMeanFromSH(dirs, y, order);
            estimated_sigma = [estimated_sigma; sigma_tmp];
            tmp_img = zeros(sx*sy*sz, 1);
            tmp_img(mask==1) = Save_tmp;
            Save(:,:,:,i) = reshape(tmp_img,[sx sy sz]);
    end
end

sigma_SHresiduals = nanmean(estimated_sigma,1);

Save = Save./S0mean;
% Save the direction-averaged data in NIFTI
tmp.img = Save; % We will use the normalized spherical mean signal
tmp.hdr.dime.dim(5) = size(tmp.img,4);
tmp.hdr.dime.dim(1) = 4;
tmp.hdr.dime.datatype = 16;
tmp.hdr.dime.bitpix = 32;
save_untouch_nii(tmp, fullfile(output_folder, 'diravg_signal.nii.gz'))

noisemap = zeros(sx*sy*sz, 1);
noisemap(mask==1) = sigma_SHresiduals;
tmp.img = reshape(noisemap,[sx sy sz]); % We will use the normalized spherical mean signal
tmp.hdr.dime.dim(5) = size(tmp.img,4);
tmp.hdr.dime.dim(1) = 3;
tmp.hdr.dime.datatype = 16;
tmp.hdr.dime.bitpix = 32;
noisemap_SHresiduals_filename = fullfile(output_folder, 'noisemap_from_SHfit.nii.gz');
save_untouch_nii(tmp, noisemap_SHresiduals_filename)

SANDIinput.Save = Save;
SANDIinput.bunique = bunique;

SANDIinput.noisemap_SHresiduals_filename = fullfile(output_folder, 'noisemap_from_SHfit.nii.gz');

noisemap_mppca = normalize_noisemap(noisemap_mppca_filename, data_filename, mask_filename, bvalues_filename); % Load the noisemap previously obtained by MP-PCA denoising and normalizes it by dividing it for the b=0 image.
sigma_mppca = noisemap_mppca(:);
sigma_mppca = sigma_mppca(sigma_mppca>0);
sigma_mppca(sigma_mppca<0) = 0;
sigma_mppca(sigma_mppca>1) = 1;

noisemap_SHresiduals = normalize_noisemap(noisemap_SHresiduals_filename, data_filename, mask_filename, bvalues_filename); % Load the noisemap previously obtained by MP-PCA denoising and normalizes it by dividing it for the b=0 image.
sigma_SHresiduals = noisemap_SHresiduals(:);
sigma_SHresiduals = sigma_SHresiduals(sigma_SHresiduals>0);
sigma_SHresiduals(sigma_SHresiduals<0) = 0;
sigma_SHresiduals(sigma_SHresiduals>1) = 1;

SANDIinput.sigma_mppca = [SANDIinput.sigma_mppca, sigma_mppca];
SANDIinput.sigma_SHresiduals = [SANDIinput.sigma_SHresiduals, sigma_SHresiduals];

% Make and save the schemefile associated with the direction-averaged data
%protocol = make_protocol(bunique./1E3, delta, smalldel);
%ProtocolToScheme(protocol, [outputfolder '/diravg.scheme'])

if diagnostics==1
    
    h = figure('Name', 'Spherical mean signal'); hold on
   
    [~, ~, slices, ~] = size(Save);
    slice_to_show = round(slices/2);
    slice_to_show(slice_to_show==0) = 1;
    img_to_show = squeeze(Save(:,:,slice_to_show,:));
    for vv = 1:size(img_to_show,3)
        subplot(4,4,vv), imshow(img_to_show(:,:,vv),[0 0.7]), colorbar
    %tmp = imtile(squeeze(Save(:,:,slice_to_show,:)));
    %imshow(tmp, [0 0.7]), colorbar
    end
    savefig(h, fullfile(output_folder, 'Spherical_mean_signal.fig'));
    close(h);

end

tt = toc;
disp(['DONE - Spherical mean computed in ' num2str(round(tt)) ' sec.'])

end