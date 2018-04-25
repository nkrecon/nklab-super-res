clear all;
addpath /usr/share/matlab/nifti
nii = load_nii('RatBrain_T2FSEref_40.nii');
img = nii.img;
sz = size(img);
xs = sz(1);
ys = sz(2);
zs = sz(3);
newnii=zeros(xs, ys, zs, 2);
newnii(:,:,:,1)=img;
newnii(:,:,:,2)=img;
nii.hdr.hk.regular = 'r';
nii.hdr.dime.dim(1)=4;
nii.hdr.dime.dim(5)=2;
nii.img=newnii;
nii.original.hdr = nii.hdr;
save_nii(nii,'RatBrain_T2FSEref_40_4D.nii');

clear all;
addpath /usr/share/matlab/nifti
nii = load_nii('RatBrain_T2FSEref_41.nii');
img = nii.img;
sz = size(img);
xs = sz(1);
ys = sz(2);
zs = sz(3);
newnii=zeros(xs, ys, zs, 2);
newnii(:,:,:,1)=img;
newnii(:,:,:,2)=img;
nii.hdr.hk.regular = 'r';
nii.hdr.dime.dim(1)=4;
nii.hdr.dime.dim(5)=2;
nii.img=newnii;
nii.original.hdr = nii.hdr;
save_nii(nii,'RatBrain_T2FSEref_41_4D.nii');

clear all;
addpath /usr/share/matlab/nifti
nii = load_nii('RatBrain_T2FSEref_42.nii');
img = nii.img;
sz = size(img);
xs = sz(1);
ys = sz(2);
zs = sz(3);
newnii=zeros(xs, ys, zs, 2);
newnii(:,:,:,1)=img;
newnii(:,:,:,2)=img;
nii.hdr.hk.regular = 'r';
nii.hdr.dime.dim(1)=4;
nii.hdr.dime.dim(5)=2;
nii.img=newnii;
nii.original.hdr = nii.hdr;
save_nii(nii,'RatBrain_T2FSEref_42_4D.nii');