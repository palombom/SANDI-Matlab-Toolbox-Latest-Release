# SANDI Matlab Toolbox: Latest Release

![SANDIToolbox_v2_Logo2](https://github.com/palombom/SANDI-Matlab-Toolbox-Latest-Release/assets/33546086/3d8bb4b3-0177-4dcc-9d50-77014ad96968)

This repository contains the latest release of the SANDI Matlab Toolbox.

The "***SANDI (Soma And Neurite Density Imaging) Matlab Toolbox***" enables model-based estimation of MR signal fraction of brain cell bodies (of all cell types, from neurons to glia, namely soma) and cell projections (of all cell types, from dentrites and myelinated axons to glial processes, namely neurties ) as well as apparent MR cell body radius and intraneurite and extracellular apparent diffusivities from a suitable diffusion-weighted MRI acquisition using Machine Learning (see the original SANDI paper for model assumptions and acquisition requirements DOI: https://doi.org/10.1016/j.neuroimage.2020.116835).

For queries about the toolbox or suggestions on how to improve it, please email: palombom@cardiff.ac.uk

## Dependencies
To use SANDI Matlab Toolbox you will need a MATLAB distribution with the Parallel Computing Toolbox, the Statistics and Machine Learning Toolbox and the Optimization Toolbox. Additionally, you will also need an external repository that is already included in the SANDI Matlab Toolbox:
* [Tools for NIfTI and ANALYZE image] Jimmy Shen (2022). Tools for NIfTI and ANALYZE image (https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image), MATLAB Central File Exchange. Retrieved April 16, 2022.

## Download 
To get the SANDI Matlab Toolbox clone this repository. The tools include all the necessary dependencies and should be ready for you to run.

If you use Linux or MacOS:

1. Open a terminal;
2. Navigate to your destination folder;
3. Clone SANDI Matlab Toolbox:
```
$ git clone https://github.com/palombom/SANDI-Matlab-Toolbox-Latest-Release.git 
```
4. Add the SANDI-Matlab-Toolbox-Latest-Release folder and subfolder to your Matlab path list. The main function is called "SANDI_batch_analysis" and it analyses one or more datasets with the SANDI model. 
5. You should now be able to use the code. 

## Usage
The function "SANDI_batch_analysis" represents the core of the toolbox. It performs the SANDI analysis on one or more datasets. It assumes that data are organized following the BIDS standard. Create a single main folder, e.g. 'ProjectMainFolder' per group and strictly use the structure:  'ProjectMainFolder/derivatives/preprocessed/sub-01/ses-01' for the dataset corresponding to the first subject and the first session of this subject. Generalize the structure to n subjects and n sessions, as illustrated below. In each of the folders 'ProjectMainFolder/derivatives/preprocessed/sub-XXX/ses-YYY' data must be named making sure that they end as explained below (including the file extension!):

- ProjectMainFolder
  - derivatives
    - preprocessed
      - sub-01
        - ses-01
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.nii.gz
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bval
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bvec
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_mask.nii.gz
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_noisemap.nii.gz
        - ...
        - ses-n
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.nii.gz
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bval
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bvec
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_mask.nii.gz
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_noisemap.nii.gz
      - ...
      - sub-n
        - ses-01
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.nii.gz
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bval
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bvec
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_mask.nii.gz
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_noisemap.nii.gz
        - ...
        - ses-n
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.nii.gz
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bval
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_dwi.bvec
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_mask.nii.gz
          - sub-<>_ses-<>_acq-<>_run-<>_desc-preproc_noisemap.nii.gz

**INPUT** to the "SANDI_batch_analysis" are: the 'ProjectMainFolder', alongside the gradient pulse separation 'Delta' and duration 'smalldelta' in ms (milliseconds). 

**OUTPUT** of the analysis will be stored in a new folder 'ProjectMainFolder -> derivatives -> SANDI_analysis -> sub-XXX -> ses-XXX -> SANDI_Output' for each subject and session.

**REPORT** - A report of the steps performed is displaied in the matlab command window and saved in the txt file 'SANDI_analysis_LogFile.txt' within the 'ProjectMainFolder' folder. 

**TEST PERFORMANCE** If the flags "SANDIinput.DoTestPerformances" and "SANDIinput.diagnostics" are set to '1' (we recommend to always do so the first time you run the toolbox for a study), then the toolbox will run a series of tests to quantify the performance of the model fitting, given the acquisition protocol used and the noise distributions of the datasets. This includes: test of SANDI model parameters cross-correlations; accuracy and precision of each model parameter estimation; bias due to unaccounted inter-compartmental exchange; sensitivity analysis (based on a nominal change of model parameters of 10%). Images of the output of the performance analysis are saved in "ProjectMainFolder -> Report_ML_Training_Performance" folder and in each "ProjectMainFolder -> derivatives -> SANDI_analysis -> sub-XXX -> ses-XXX -> SANDI_Output -> SANDIreport" folders. 

## Citation
If you use SANDI Matlab Toolbox, please remember to cite our main SANDI work:

1. Marco Palombo, Andrada Ianus, Michele Guerreri, Daniel Nunes, Daniel C. Alexander, Noam Shemesh, Hui Zhang, "SANDI: A compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI", NeuroImage 2020, 215: 116835, ISSN 1053-8119, DOI: https://doi.org/10.1016/j.neuroimage.2020.116835. 

and our preclinical optimization:

2. Andrada Ianuş, Joana Carvalho, Francisca F. Fernandes, Renata Cruz, Cristina Chavarrias, Marco Palombo, Noam Shemesh, "Soma and Neurite Density MRI (SANDI) of the in-vivo mouse brain and comparison with the Allen Brain Atlas", NeuroImage 2022, 254: 119135, ISSN 1053-8119, DOI: https://doi.org/10.1016/j.neuroimage.2022.119135.


## License
SANDI Matlab Toolbox is distributed under the BSD 2-Clause License (https://github.com/palombom/SANDI-Matlab-Toolbox/blob/main/LICENSE), Copyright (c) 2022 Cardiff University and University College London. All rights reserved.

**The use of SANDI Matlab Toolbox MUST also comply with the individual licenses of all of its dependencies.**

## Acknowledgements
The development of SANDI was supported by EPSRC (EP/G007748, EP/I027084/01, EP/L022680/1, EP/M020533/1, EP/N018702/1, EP/M507970/1) and European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Starting Grant, agreement No. 679058). Dr. Marco Palombo is currently supported by the UKRI Future Leaders Fellowship MR/T020296/2.

