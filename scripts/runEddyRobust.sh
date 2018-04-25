#!/bin/bash

singcmd=singularity
eddy=eddy_cuda8.0
sing_outdir=/opt/data
sing_indir=/tmp
singimg="/media/chidi/NTFS-SHARE/Repos/nklab-neuro-tools/nklab-neuro-tools.simg"
inputdir=/media/chidi/NTFS-SHARE/Repos/nklab-super-res/outputfiles
outdir=/media/chidi/NTFS-SHARE/Repos/nklab-super-res/outputfiles

dwiname=input_of_eddy_correct.nii
dwieddyout=${sing_outdir}/output_of_eddy_correct
dwiout=${sing_outdir}/${dwiname}
dwi=${sing_indir}/${dwiname}

index=${sing_outdir}/index.txt
acqparams=${sing_outdir}/acqparams.txt

bvals=${sing_indir}/bvalsr
bvecs=${sing_indir}/bvecsr

if [ ! -f ${inputdir}/bvalsr ]
then
  bvals=${sing_indir}/bvalsr
fi

if [ ! -f ${input_dir}/bvec$sr ]
then
  bvecs=${sing_indir}/bvecsr
fi



cd ${outdir}

#STEP 1 - EDDY CORRECTION
#if [ ! -f ${outdir}/${4}.nii.gz -o $overwrite -eq 1 ]
#then

# 1a. obtain the first b0 values for creating mask
$singcmd run -B ${outdir}:${sing_outdir} -B ${inputdir}:${sing_indir} ${singimg} fslroi ${dwi} ${dwiout}0 0 1

# 1b. create mask using b0 image
$singcmd run -B ${outdir}:/opt/data ${singimg} bet ${dwiout}0 ${dwiout}_brain -R -f 0.1 -g 0 -m

#remove extraneous files
rm ${outdir}/${dwiname}0.nii.gz
rm ${outdir}/${dwiname}_brain.nii.gz

#create dummy index file because blip-up, blip-down acquisitions not obtained for TOPUP
indx=""
directions=$(fslinfo ${inputdir}/${dwiname} | grep dim4 | head -1 | awk ' {print $2} ')
for ((i=1; i<=$directions; i+=1)); do indx="$indx 1"; done
echo $indx > ${outdir}/index.txt 

#create dummy acquisition file
echo "0 1 0 0.1" > ${outdir}/acqparams.txt
echo "0 1 0 0.1" >> ${outdir}/acqparams.txt

#run Eddy
$singcmd run --nv -B ${outdir}:/opt/data -B ${inputdir}:${sing_indir} ${singimg} $eddy --imain=${dwi} --mask=${dwiout}_brain_mask --acqp=${acqparams} --index=${index} --bvecs=${bvecs} --bvals=${bvals} --out=${dwieddyout}

rm ${outdir}/index.txt
rm ${outdir}/acqparams.txt
rm ${outdir}/${dwiname}_brain_mask.nii.gz
#fi