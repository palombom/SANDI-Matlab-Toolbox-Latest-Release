function SANDIinput = ProcessAllDatasets(SANDIinput)

StudyMainFolder = SANDIinput.StudyMainFolder;
FoldersList = dir(fullfile(StudyMainFolder,'Dataset_*'));
SANDIinput.DatasetList = FoldersList;
SANDIinput.report = cell(numel(FoldersList),1);

for subj_id = 1:numel(FoldersList)

    SANDIinput.bvalues_filename = fullfile(StudyMainFolder, ['Dataset_' num2str(subj_id)], 'bvals.bval');
    SANDIinput.bvecs_filename = fullfile(StudyMainFolder, ['Dataset_' num2str(subj_id)], 'bvecs.bvec');
    SANDIinput.noisemap_mppca_filename = fullfile(StudyMainFolder, ['Dataset_' num2str(subj_id)], 'noisemap.nii.gz');
    SANDIinput.data_filename = fullfile(StudyMainFolder, ['Dataset_' num2str(subj_id)], 'dwi.nii.gz');
    SANDIinput.mask_filename = fullfile(StudyMainFolder, ['Dataset_' num2str(subj_id)], 'mask.nii.gz');

    if subj_id==1
        SANDIinput.sigma_mppca = [];
        SANDIinput.sigma_SHresiduals = [];
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Compute Spherical Mean signal using Spherical Harmonics
    SANDIinput.output_folder = fullfile(StudyMainFolder, ['Dataset_' num2str(subj_id)], 'SANDI_Output'); % <---- Folder where the direction-averaged signal and SANDI fit results for each subject will be stored;
    SANDIinput.report{subj_id} = report_generator([], fullfile(SANDIinput.output_folder,'SANDIreport'));
    SANDIinput.subj_id = subj_id;
    SANDIinput = make_direction_average(SANDIinput);

end

end