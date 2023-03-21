function I = Gauss_smoothing_3D(I, fwhm, gauss_std)

% Function to apply a 3D smoothing to the image I with Gaussian kernel of defined FWHM,
% or given standard deviation gauss_std
%
% Author:
% Dr. Marco Palombo
% Cardiff University Brain Research Imaging Centre (CUBRIC)
% Cardiff University, UK
% 8th December 2021
% Email: palombom@cardiff.ac.uk

[~,~,~,vol] = size(I);

if isempty(gauss_std) && ~isempty(fwhm)
    gauss_std = fwhm/sqrt(8*log(2));
end

for i=1:vol
    I(:,:,:,i) = imgaussfilt3(I(:,:,:,i),gauss_std);
end
end