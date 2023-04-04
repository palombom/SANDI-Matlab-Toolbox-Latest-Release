function SANDIinput = ProcessAllDatasets(SANDIinput)

StudyMainFolder = SANDIinput.StudyMainFolder;
FoldersList = dir(fullfile(StudyMainFolder,'Dataset_*'));
SANDIinput.DatasetList = FoldersList;
SANDIinput.report = cell(numel(FoldersList),1);

for subj_id = 1:numel(FoldersList)

    SANDIinput.bvalues_filename = fullfile(StudyMainFolder, FoldersList(subj_id).name, 'bvals.bval');
    SANDIinput.bvecs_filename = fullfile(StudyMainFolder, FoldersList(subj_id).name, 'bvecs.bvec');
    SANDIinput.noisemap_mppca_filename = fullfile(StudyMainFolder, FoldersList(subj_id).name, 'noisemap.nii.gz');
    SANDIinput.data_filename = fullfile(StudyMainFolder, FoldersList(subj_id).name, 'dwi.nii.gz');
    SANDIinput.mask_filename = fullfile(StudyMainFolder, FoldersList(subj_id).name, 'mask.nii.gz');

    if subj_id==1
        SANDIinput.sigma_mppca = [];
        SANDIinput.sigma_SHresiduals = [];
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Compute Spherical Mean signal using Spherical Harmonics
    SANDIinput.output_folder = fullfile(StudyMainFolder, FoldersList(subj_id).name, 'SANDI_Output'); % <---- Folder where the direction-averaged signal and SANDI fit results for each subject will be stored;
    SANDIinput.report{subj_id} = report_generator([], fullfile(SANDIinput.output_folder,'SANDIreport'));
    SANDIinput.subj_id = subj_id;
    SANDIinput = make_direction_average(SANDIinput);

end

end