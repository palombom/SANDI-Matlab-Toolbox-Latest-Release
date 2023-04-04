function SANDIinput = AnalyseAllDatasets(SANDIinput)

StudyMainFolder = SANDIinput.StudyMainFolder;
FoldersList = dir(fullfile(StudyMainFolder,'Dataset_*'));

for subj_id = 1:numel(FoldersList)
    % Load the data
    SANDIinput.output_folder = fullfile(StudyMainFolder, FoldersList(subj_id).name, 'SANDI_Output'); 
    SANDIinput.data_filename = fullfile(StudyMainFolder, FoldersList(subj_id).name, 'dwi.nii.gz');
    SANDIinput.mask_filename = fullfile(StudyMainFolder, FoldersList(subj_id).name, 'mask.nii.gz');
    % Fit the data
    SANDIinput.subj_id = subj_id;
    run_model_fitting(SANDIinput);
end

end