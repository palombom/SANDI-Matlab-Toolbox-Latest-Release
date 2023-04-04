function [] = run_model_fitting(SANDIinput)

% Main script to fit the SANDI model using Random
% Forest (RF) and/or multi-layers perceptron (MLP)
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 8th December 2021
% Email: palombom@cardiff.ac.uk

mask_data = SANDIinput.mask_filename ;
output_folder = SANDIinput.output_folder ;
Mdl = SANDIinput.Mdl ;
trainedML = SANDIinput.trainedML;
MLmodel = SANDIinput.MLmodel ;
model = SANDIinput.model ;
diagnostics = SANDIinput.diagnostics ;

%% Load direction-averaging with gaussian smoothing given the provided FHWM

img_data = fullfile(output_folder, 'diravg_signal.nii.gz'); % data to process: direction-averaged signals for each subject

%% Load data

if ~isempty(mask_data)
    tmp = load_untouch_nii(mask_data);
    mask = double(tmp.img);
end

tmp = load_untouch_nii(img_data);
nifti_struct = tmp;

I = double(tmp.img);
[sx, sy, sz, vol] = size(I);

disp(['Data ' img_data ' loaded:'])
disp(['  - matrix size = ' num2str(sx) ' x ' num2str(sy) ' x ' num2str(sz)])
disp(['  - volumes = ' num2str(vol) ])

delta = model.delta;
smalldel = model.smalldel;
bvals = model.bvals;

disp('Protocol loaded:')
disp(['  - gradient pulse duration ~ ' num2str(round(smalldel)) ' ms'])
disp(['  - gradient pulse separation ~ ' num2str(round(delta)) ' ms'])
disp(['  - diffusion time ~ ' num2str(round(delta  - smalldel/3)) ' ms'])
disp(['  - b values ~ ' num2str(round(bvals.*1000)) ' s/mm^2'])
disp(['  - #' num2str(sum(round(bvals.*1000)>=100)) ' b shells'])

if isempty(mask_data), mask = ones(sx,sy,sz); end

% Prepare ROI for fitting
ROI = reshape(I, [sx*sy*sz vol]);
m = reshape(mask, [sx*sy*sz 1]);
signal = (ROI(m==1,:));

% Remove nan or inf and impose that the normalised signal is >= 0
signal(isnan(signal)) = 0; signal(isinf(signal)) = 0; signal(signal<0) = 0;

%% Fitting the model to the data using pretrained models

if isempty(MLmodel), MLmodel='RF'; end

MLdebias = SANDIinput.MLdebias;

switch MLmodel

    case 'RF'
        %% RF fit

        disp('Fitting using a Random Forest regressor implemented in matlab ')

        % --- Using Matlab

        disp('Applying the Random Forest...')
        mpgMean = apply_RF_matlab(signal, trainedML, MLdebias);

    case 'MLP'

        %% MLP fit

        disp('Fitting using a MLP regressor implemented in matlab ')

        % --- Using Matlab

        disp('Applying the MLP...')
        mpgMean = apply_MLP_matlab(signal, trainedML, MLdebias);

end

%% Calculate and save SANDI parametric maps

names = {'fneurite', 'fsoma', 'Din', 'Rsoma', 'De', 'fextra', 'Rsoma_Low_fsoma_Filtered', 'Din_Low_fsoma_Filtered'};
bounds = [0 0.7; 0 0.7; model.paramsrange(3:5,:).*1; 0 1];

mpgMean = abs(mpgMean);

fneu = mpgMean(:,1);
fneu(fneu<0) = 0;

fsom = mpgMean(:,2);
fsom(fsom<0) = 0;

fe = 1 - fneu - fsom;
fe(fe<0) = 0;

fneurite = fneu ./ (fneu + fsom + fe);
fsoma = fsom ./ (fneu + fsom + fe);
fextra = fe ./ (fneu + fsom + fe);

disp('Saving SANDI parametric maps')

try
    if diagnostics==1
        r = SANDIinput.report{SANDIinput.subj_id};
    end

    for i=1:size(mpgMean,2)+3

        itmp = zeros(sx*sy*sz,1);

        if i==1
            itmp(mask==1) = fneurite;
        elseif i==2
            itmp(mask==1) = fsoma;
        elseif i==size(mpgMean,2)+1
            itmp(mask==1) = fextra;
        elseif i==size(mpgMean,2)+2
            Rsoma_tmp = mpgMean(:,4);
            Rsoma_tmp(fsoma<=0.15) = 0;
            itmp(mask==1) = Rsoma_tmp;
        elseif i==size(mpgMean,2)+3
            Din_tmp = mpgMean(:,3);
            Din_tmp(fneurite<0.10) = 0;
            itmp(mask==1) = Din_tmp;
        else
            mpgMean(mpgMean(:,i)<0,i) = 0;
            itmp(mask==1) = mpgMean(:,i);
        end

        itmp = reshape(itmp,[sx sy sz]);

        if diagnostics==1
            if i<=size(mpgMean,2)+1
                h = figure('Name',['SANDI ' names{i} ' map']); hold on
                tmp = imtile(itmp);
                imshow(tmp,bounds(i,:)), title(names{i}), colorbar, colormap parula

                T = getframe(h);
                imwrite(T.cdata, fullfile(output_folder , 'SANDIreport', ['SANDI ' names{i} ' map.tiff']))

                r.section(['SANDI ' names{i} ' map']);
                r.add_text(['The plot shows the SANDI ' names{i} ' map.']);
                r.add_figure(gcf,['SANDI ' names{i} ' map. Units are um for maps indicating radius and um^2/ms for maps indicating diffusivities.'],'left');
                r.end_section();
                

                savefig(h, fullfile(output_folder, 'SANDIreport', ['SANDI ' names{i} ' map.fig']));
                close(h);
            end
        end

        nifti_struct.img = itmp;
        nifti_struct.hdr.dime.dim(5) = size(nifti_struct.img,4);
        if size(nifti_struct.img,4)==1
            nifti_struct.hdr.dime.dim(1) = 3;
        else
            nifti_struct.hdr.dime.dim(1) = 4;
        end
        nifti_struct.hdr.dime.datatype = 16;
        nifti_struct.hdr.dime.bitpix = 32;

        save_untouch_nii(nifti_struct,fullfile(output_folder, ['SANDI-fit_' names{i} '.nii.gz']));
        disp(['  - ' output_folder '/SANDI-fit_' names{i} '.nii.gz'])

    end

    if diagnostics==1
        r.close();
    end
    
catch

    if diagnostics==1
        r = SANDIinput.report{SANDIinput.subj_id};
        h = figure('Name','SANDI maps for a representative slice'); hold on
    end

    for i=1:size(mpgMean,2)+3

        itmp = zeros(sx*sy*sz,1);

        if i==1
            itmp(mask==1) = fneurite;
        elseif i==2
            itmp(mask==1) = fsoma;
        elseif i==size(mpgMean,2)+1
            itmp(mask==1) = fextra;
        elseif i==size(mpgMean,2)+2
            Rsoma_tmp = mpgMean(:,4);
            Rsoma_tmp(fsoma<=0.15) = 0;
            itmp(mask==1) = Rsoma_tmp;
        elseif i==size(mpgMean,2)+3
            Din_tmp = mpgMean(:,3);
            Din_tmp(fneurite<0.10) = 0;
            itmp(mask==1) = Din_tmp;
        else
            mpgMean(mpgMean(:,i)<0,i) = 0;
            itmp(mask==1) = mpgMean(:,i);
        end

        itmp = reshape(itmp,[sx sy sz]);

        if diagnostics==1
            if i<=size(mpgMean,2)+1
                [~, ~, slices, ~] = size(itmp);
                slice_to_show = round(slices/2);
                slice_to_show(slice_to_show==0) = 1;
                subplot(2,3,i), hold on, imshow(itmp(:,:,slice_to_show), bounds(i,:)), title(names{i}), colorbar, colormap parula
            end
        end

        nifti_struct.img = itmp;
        nifti_struct.hdr.dime.dim(5) = size(nifti_struct.img,4);
        if size(nifti_struct.img,4)==1
            nifti_struct.hdr.dime.dim(1) = 3;
        else
            nifti_struct.hdr.dime.dim(1) = 4;
        end
        nifti_struct.hdr.dime.datatype = 16;
        nifti_struct.hdr.dime.bitpix = 32;

        save_untouch_nii(nifti_struct,fullfile(output_folder, ['SANDI-fit_' names{i} '.nii.gz']));
        disp(['  - ' output_folder '/SANDI-fit_' names{i} '.nii.gz'])

    end

    if diagnostics==1

        T = getframe(h);
        imwrite(T.cdata, fullfile(output_folder , 'SANDIreport', 'SANDI_Maps.tiff'))

        r.section('SANDI maps for a representative slice');
        r.add_text('The plot shows the SANDI parametetric maps for a representative slice.');
        r.add_figure(gcf,'SANDI parametetric maps for a representative slice','left');
        r.end_section();
        r.close();

        savefig(h, fullfile(output_folder, 'SANDIreport', 'SANDI_Maps.fig'));
        close(h);
    end

end

end

