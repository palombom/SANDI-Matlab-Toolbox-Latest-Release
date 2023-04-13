function SANDIinput = ProcessAllDatasets(SANDIinput)

StudyMainFolder = fullfile(SANDIinput.StudyMainFolder,'derivatives','preprocessed');
SubList = dir(fullfile(StudyMainFolder,'sub-*'));
SANDIinput.DatasetList = SubList;
SANDIinput.report = [];

for subj_id = 1:numel(SubList)
    
    SesList = dir(fullfile(StudyMainFolder,SubList(subj_id).name,'ses-*'));
    
    for ses_id = 1:numel(SesList)
        
        bvalues_filename = dir(fullfile(SesList(ses_id).folder, SesList(ses_id).name, '*_dwi.bval'));
        SANDIinput.bvalues_filename = fullfile(bvalues_filename(1).folder, bvalues_filename(1).name);

        bvecs_filename = dir(fullfile(SesList(ses_id).folder, SesList(ses_id).name, '*_dwi.bvec'));
        SANDIinput.bvecs_filename = fullfile(bvecs_filename(1).folder, bvecs_filename(1).name);

        data_filename = dir(fullfile(SesList(ses_id).folder, SesList(ses_id).name, '*_dwi.nii.gz'));
        SANDIinput.data_filename = fullfile(data_filename(1).folder, data_filename(1).name);
        
        mask_filename = dir(fullfile(SesList(ses_id).folder, SesList(ses_id).name, '*_mask.nii.gz'));
        SANDIinput.mask_filename = fullfile(mask_filename(1).folder, mask_filename(1).name);
        
        noisemap_mppca_filename = dir(fullfile(SesList(ses_id).folder, SesList(ses_id).name, '*_noisemap.nii.gz'));
        SANDIinput.noisemap_mppca_filename = fullfile(noisemap_mppca_filename(1).folder, noisemap_mppca_filename(1).name);
        
        if subj_id==1 && ses_id==1
            SANDIinput.sigma_mppca = [];
            SANDIinput.sigma_SHresiduals = [];
        end
        
        % Compute Direction-averaged signal
        SANDIinput.output_folder = fullfile(SANDIinput.StudyMainFolder, 'derivatives', 'SANDI_analysis', SubList(subj_id).name, SesList(ses_id).name, 'SANDI_Output'); % <---- Folder where the direction-averaged signal and SANDI fit results for each subject will be stored;
        SANDIinput.report(subj_id, ses_id).r = report_generator([], fullfile(SANDIinput.output_folder,'SANDIreport'));
        SANDIinput.subj_id = subj_id;
        SANDIinput.ses_id = ses_id;
        SANDIinput = make_direction_average(SANDIinput);
    end
end
end