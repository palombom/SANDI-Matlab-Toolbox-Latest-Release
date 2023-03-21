function noisemap = normalize_noisemap(noisemap_filename, data_filename, mask_filename, bvalues_filename)

tmp = load_untouch_nii(noisemap_filename);
noisemap = double(tmp.img);

tmp = load_untouch_nii(mask_filename);
mask = double(tmp.img);
se = strel('cube',9);
mask = imerode(mask, se);

tmp = load_untouch_nii(data_filename);
dwi = double(tmp.img);

bvals = importdata(bvalues_filename);

b0mean = nanmean(dwi(:,:,:,bvals<100), 4);

noisemap = noisemap./b0mean.*mask;

end