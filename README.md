# nklab-super-res
Experimental Julia based super resolution pipeline for dwi images 

## Super Resolution without eddy correction

Select the 3 input files in order of acquisition for SR. To do this sort the files by date ascending by modified date. So for example if image1, image2 and image 3 were acquired in that order then they should be sorted in the input screen as:

`image1`

`image2`

`image3`

Then select the output folder and specify a name for your output file abd click run.


## Super Resolution without eddy correction

`julia NKSRgui.ji`


## Super Resolution with eddy correction usind FSL's eddy_correct

`julia NKSRgui-eddycorrect`

To adapt for your host change the following values in `NKSRgui-eddycorrect`:

`SINGIMG="....../nklab-neuro-tools.simg"`

`singcmd="singularity"`


## Super Resolution with eddy correction using FSL's EDDY for input images, and eddy_correct on the derived image `input-to-eddy-correct`

`julia NKSRgui-EDDY.jl`

To adapt for your host change the following values in `NKSRgui-EDDY.jl`:

`SINGIMG="....../nklab-neuro-tools.simg"`

`singcmd="singularity"`

and change the following for `runEddyRobust.sh`

`singcmd=singularity`

`eddy=eddy_cuda8.0`



## Super Resolution with eddy correction using FSL's EDDY for all input images and derived image

`julia NKSRgui-allEDDY.jl`

To adapt for your host change the following values in `NKSRgui-allEDDY.jl`:

`SINGIMG="....../nklab-neuro-tools.simg"`

`singcmd="singularity"`

and change the following for `runEddyRobust.sh`

`singcmd=singularity`

`eddy=eddy_cuda8.0`


## Testing times
Testing was done on personal laptop (intel i7, 16GB mem, 2.8Ghx x 8 core +  Geoforce GTX 1050). Times should be better on XenialChen and Fourier.

`NKSRgui 								1668 seconds`

`NKSRgui_eddycorrect					2900 seconds`

`NKSRgui_EDDY + eddy-correct (cuda8.0)	2990 seconds`

`NKSRgui_allEDDY (cuda8.0)				3661 seconds`

`NKSRgui_EDDY (openmp)					4878 seconds`