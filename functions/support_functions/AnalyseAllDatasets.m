function SANDIinput = AnalyseAllDatasets(SANDIinput)

StudyMainFolder = fullfile(SANDIinput.StudyMainFolder,'derivatives','SANDI_analysis');
PreprocMainFolder = fullfile(SANDIinput.StudyMainFolder,'derivatives','preprocessed');

SubList = dir(fullfile(StudyMainFolder,'sub-*'));
SANDIinput.DatasetList = SubList;

for subj_id = 1:numel(SubList)
    
    SesList = dir(fullfile(StudyMainFolder,SubList(subj_id).name,'ses-*'));
    PreprocSesList = dir(fullfile(PreprocMainFolder,SubList(subj_id).name,'ses-*'));
    
    for ses_id = 1:numel(SesList)
        % Load the data
        SANDIinput.output_folder = fullfile(SesList(ses_id).folder, SesList(ses_id).name, 'SANDI_Output');
        
        data_filename = dir(fullfile(PreprocSesList(ses_id).folder, PreprocSesList(ses_id).name, '*_dwi.nii.gz'));
        SANDIinput.data_filename = fullfile(data_filename(1).folder, data_filename(1).name);
        
        mask_filename = dir(fullfile(PreprocSesList(ses_id).folder, PreprocSesList(ses_id).name, '*_mask.nii.gz'));
        SANDIinput.mask_filename = fullfile(mask_filename(1).folder, mask_filename(1).name);
        
        % Fit the data
        SANDIinput.subj_id = subj_id;
        SANDIinput.ses_id = ses_id;
        
        tic
        if SANDIinput.WithDot == 1
                run_model_fitting_with_dot(SANDIinput);
        else
                run_model_fitting(SANDIinput);
        end
        
        tt = toc;
        disp(['DONE - Dataset analysed in in ' num2str(round(tt)) ' sec.'])
        fprintf(SANDIinput.LogFileID,['DONE - Dataset analysed in in ' num2str(round(tt)) ' sec.\n']);

    end
end
end