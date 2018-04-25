addprocs(Sys.CPU_CORES-nprocs())

# this program reconstructs data with 200 micrometer resolution
push!(LOAD_PATH,"library");
using Read_NIfTI1_real
@everywhere using myFun
@everywhere using Interpolations
using Gtk
using Gtk.ShortNames
using JLD

function runSR(filenameA::String,filenameB::String,filenameC::String,outputfile::String)

#perform eddy correction using singularity image
tic()
print("SuperResolution is using ", nprocs(), " cores\n")
dirA=rsplit(filenameA,"/";limit=2)[1]
fileA=rsplit(filenameA,"/";limit=2)[2]
dirB=rsplit(filenameB,"/";limit=2)[1]
fileB=rsplit(filenameB,"/";limit=2)[2]
dirC=rsplit(filenameC,"/";limit=2)[1]
fileC=rsplit(filenameC,"/";limit=2)[2]
dirOut=rsplit(outputfile,"/";limit=2)[1]
fileOut=rsplit(outputfile,"/";limit=2)[2]

#Location of Singularity image
SINGIMG="/media/chidi/NTFS-SHARE/Repos/nklab-neuro-tools/nklab-neuro-tools.simg"
singcmd="singularity"

#perform eddy correction using singularity image
print("Performing Eddy Correction on ",filenameA,"\n")
run(`$singcmd run -B $dirA:/opt/data -B $dirOut:/tmp $SINGIMG eddy_correct /opt/data/$fileA /tmp/eddyA 0`)
fileA_eddy="$dirOut/eddyA.nii.gz"

print("Performing Eddy Correction on ",filenameB,"\n")
run(`$singcmd run -B $dirB:/opt/data -B $dirOut:/tmp $SINGIMG eddy_correct /opt/data/$fileB /tmp/eddyB 0`)
fileB_eddy="$dirOut/eddyB.nii.gz"

print("Performing Eddy Correction on ",filenameC,"\n")
run(`$singcmd run -B $dirC:/opt/data -B $dirOut:/tmp $SINGIMG eddy_correct /opt/data/$fileC /tmp/eddyC 0`)
fileC_eddy="$dirOut/eddyC.nii.gz"

headerinfoA = load_niigz_header(fileA_eddy); # see the source code in library/Read_NIfTI1_real.jl
dataA = load_niigz_data(fileA_eddy, headerinfoA); 
headerinfoB = load_niigz_header(fileB_eddy); 
dataB = load_niigz_data(fileB_eddy, headerinfoB); 
headerinfoC = load_niigz_header(fileC_eddy); 
dataC = load_niigz_data(fileC_eddy, headerinfoC); 
xdim,ydim,zdim,ddim=size(dataA);

print("Converting from Int16 to Float32\n")
#Converting data from Int16 to Float32
data1f = convert(Array{Float32},dataA);
data2f = convert(Array{Float32},dataB);
data3f = convert(Array{Float32},dataC);

#Denoising three sets of low-resolution images
print("Denoising eddy corrected LR images\n")

data1f_sm = denoisingDWI(data1f); # see denoisingDWI source code in library/myFun.jl
data2f_sm = denoisingDWI(data2f);
data3f_sm = denoisingDWI(data3f);

#A better version of image interpolation
print("Creating interpolated image\n")
data1f_int = SharedArray{Float32,4}((xdim,ydim,zdim*3+2,ddim)); 
data2f_int = SharedArray{Float32,4}((xdim,ydim,zdim*3+2,ddim)); 
data3f_int = SharedArray{Float32,4}((xdim,ydim,zdim*3+2,ddim)); 

@everywhere function interpme(zdim, array1D)
    A_x = 1f0:1f0:Float32(zdim)
    itp = interpolate(array1D, BSpline(Cubic(Line())), OnCell())
    sitp = scale(itp, A_x)
    return sitp[collect(1f0:1//3:Float32(zdim))]
end

@time @sync @parallel for cntx = 1:xdim
    for cnty = 1:ydim
        for cntd = 1:ddim
            tmp01 = data1f_sm[cntx,cnty,:,cntd]
            data1f_int[cntx,cnty,2:end-3,cntd] = interpme(zdim,tmp01)
        end
    end
end

@time @sync @parallel for cntx = 1:xdim
    for cnty = 1:ydim
        for cntd = 1:ddim
            tmp01 = data2f_sm[cntx,cnty,:,cntd]
            data2f_int[cntx,cnty,3:end-2,cntd] = interpme(zdim,tmp01)
        end
    end
end

@time @sync @parallel for cntx = 1:xdim
    for cnty = 1:ydim
        for cntd = 1:ddim
            tmp01 = data3f_sm[cntx,cnty,:,cntd]
            data3f_int[cntx,cnty,4:end-1,cntd] = interpme(zdim,tmp01)
        end
    end
end

data1f_int = data1f_int / 3f0;
data2f_int = data2f_int / 3f0;
data3f_int = data3f_int / 3f0;

data1f2f3f_int = (data1f_int+data2f_int+data3f_int);


print("Creating and Saving intermediate image as input to eddy correct\n")
headerinfo = deepcopy(headerinfoA)
headerinfo["dim"][4] = zdim*3+2
headerinfo["dim"][5] = ddim*3
headerinfo["pixdim"][4] = headerinfo["pixdim"][4]/3f0
headerinfo["srow_z"][3] = headerinfo["srow_z"][3]/3f0
headerinfo["srow_z"][4] = headerinfo["srow_z"][4]/3f0
tmpdata = zeros(Int16, xdim,ydim,zdim*3+2,ddim*3);
tmpdata[:,:,:,1:ddim] = convert(Array{Int16}, round.(data1f_int));
tmpdata[:,:,:,ddim+1:ddim*2] = convert(Array{Int16}, round.(data2f_int));
tmpdata[:,:,:,2*ddim+1:ddim*3] = convert(Array{Int16}, round.(data3f_int));
newFileName ="$dirOut/input_of_eddy_correct.nii"
write_nii_header(newFileName, headerinfo);
fid = open(newFileName,"a");
write(fid, tmpdata);
close(fid);

print("Performing Eddy Correction on ",newFileName,"\n")
run(`$singcmd run -B $dirOut:/opt/data $SINGIMG eddy_correct /opt/data/input_of_eddy_correct.nii /opt/data/output_of_eddy_correct 0`)

#Loading eddy_corrected data back to julia
filename_int_eddy = "$dirOut/output_of_eddy_correct.nii.gz";
headerinfo_int_eddy = load_niigz_header(filename_int_eddy); # see the source code in library/Read_NIfTI1_real.jl
data_int_eddy = load_niigz_data(filename_int_eddy, headerinfo_int_eddy); 


#Converting data to Float32; and to 3 sets of low-res images
data_interpolation_eddy = convert(Array{Float32},data_int_eddy);
data1f_int_eddy = data_interpolation_eddy[:,:,:,1:ddim];
data2f_int_eddy = data_interpolation_eddy[:,:,:,ddim+1:ddim*2];
data3f_int_eddy = data_interpolation_eddy[:,:,:,2*ddim+1:ddim*3];
data1f2f3f_int_eddy = data3f_int_eddy + data3f_int_eddy + data3f_int_eddy;
data1f_inteddy_2_lowres = data1f_int_eddy[:,:,1:3:end-2,:]+data1f_int_eddy[:,:,2:3:end-2,:]+data1f_int_eddy[:,:,3:3:end-2,:];
data2f_inteddy_2_lowres = data2f_int_eddy[:,:,2:3:end-1,:]+data2f_int_eddy[:,:,3:3:end-1,:]+data2f_int_eddy[:,:,4:3:end-1,:];
data3f_inteddy_2_lowres = data3f_int_eddy[:,:,3:3:end-0,:]+data3f_int_eddy[:,:,4:3:end-0,:]+data3f_int_eddy[:,:,5:3:end-0,:];
data_sr = SharedArray{Float32,4}((xdim,ydim,zdim*3+2,ddim)); # this is for parallel computation
lambda = 0.5f0
data1f2f3f_int_eddy[:,:,1,:] = data1f2f3f_int_eddy[:,:,3,:]/1.2f0;
data1f2f3f_int_eddy[:,:,2,:] = data1f2f3f_int_eddy[:,:,3,:]/1.1f0;
slice_profile = convert(Array{Float32},[0.33, 0.34, 0.33]);

print("Performing Super Resolution\n")

@sync @parallel for cntd = 1:ddim
    datacombined = superresolution_lambda_better_boundary_condition(data1f_inteddy_2_lowres[:,:,:,cntd],data2f_inteddy_2_lowres[:,:,:,cntd],data3f_inteddy_2_lowres[:,:,:,cntd],data1f2f3f_int_eddy[:,:,:,cntd],slice_profile, lambda);
    data_sr[:,:,:,cntd] = datacombined;
end

# superresolution_lambda source code is in library/myFun.jl
#tmp01 = abs.(data1f2f3f_int_eddy-data_sr)./data1f2f3f_int_eddy;
#L = find(tmp01.>0.1f0);
#data_sr_ac = deepcopy(data_sr); # ac means artifact correction
#data_sr_ac[L] = data1f2f3f_int_eddy[L];
#data_sr_ac[:,:,45:end,:] = data_sr[:,:,45:end,:];

data_sr_cp = copy(data_sr);
print("Denoising intermediate Super Resolution results\n")
#@time data_sr_ac_sm = denoisingDWI(data_sr_ac_cp);
@time data_sr_sm = denoisingDWI(data_sr_cp);

#Saving the SR results
data_sr_sm = 32000f0 * data_sr_sm / maximum(data_sr_sm[:]);
data_sr_sm = convert(Array{Int16}, round.(data_sr_sm));
headerinfo = deepcopy(headerinfoA)
headerinfo["dim"][4] = zdim*3+2
headerinfo["pixdim"][4] = headerinfo["pixdim"][4]/3f0
headerinfo["srow_z"][3] = headerinfo["srow_z"][3]/3f0
headerinfo["srow_z"][4] = headerinfo["srow_z"][4]/3f0
headerinfo["cal_max"] = Float32(32000);
newFileName = "$dirOut/$fileOut.nii"
write_nii_header(newFileName, headerinfo);
fid = open(newFileName,"a");
write(fid, data_sr_sm);
close(fid);
print("Super Resolution results saved as ",newFileName,"\n")

#Saving the interpolation results
data1f2f3f_int_eddy = 32000f0 * data1f2f3f_int_eddy / maximum(data1f2f3f_int_eddy[:]);
data1f2f3f_int_eddy = convert(Array{Int16}, round.(data1f2f3f_int_eddy));
headerinfo = deepcopy(headerinfoA)
headerinfo["dim"][4] = zdim*3+2
headerinfo["pixdim"][4] = headerinfo["pixdim"][4]/3f0
headerinfo["srow_z"][3] = headerinfo["srow_z"][3]/3f0
headerinfo["srow_z"][4] = headerinfo["srow_z"][4]/3f0
headerinfo["cal_max"] = Float32(32000);
newFileName = string("$dirOut/$fileOut","_INT.nii")
write_nii_header(newFileName, headerinfo);
fid = open(newFileName,"a");
write(fid, data1f2f3f_int_eddy);
close(fid);
print("Interpolation results saved as ",newFileName,"\n")

#Saving mean DWI
data1f2f3f_int_eddy = convert(Array{Float32},data1f2f3f_int_eddy);
dwi = mean(data1f2f3f_int_eddy[:,:,:,2:21],4);
dwi = 32000. * dwi / maximum(dwi[:]);
dwi = convert(Array{Int16}, round.(dwi));
println(headerinfo["dim"])
headerinfo["dim"][1]=3
headerinfo["dim"][5]=1
newFileName = string("$dirOut/$fileOut","_mean_DWI.nii")
write_nii_header(newFileName, headerinfo);
fid = open(newFileName,"a");
write(fid, dwi);
close(fid);
print("mean DWI saved as ",newFileName,"\n")
print("Output files successfully created. Super Resolution completed.")
toc()
print("...Now...Cleaning up temporary files\n")
  try
    run(`rm $dirOut/eddyA.nii.gz`)
    run(`rm $dirOut/eddyB.nii.gz`)
    run(`rm $dirOut/eddyC.nii.gz`)
    run(`rm $dirOut/eddyA.ecclog`)
    run(`rm $dirOut/eddyB.ecclog`)
    run(`rm $dirOut/eddyC.ecclog`)
    run(`rm $dirOut/input_of_eddy_correct.nii`)
    run(`rm $dirOut/output_of_eddy_correct.nii.gz`)
    run(`rm $dirOut/output_of_eddy_correct.ecclog`)
  catch
    print("Error cleaning up files. Please check $dirOut and manually remove unrequired files.")
  end
print("Done.")
end



#GUI code begins below
win = @Window("Julia NK Denoise")
f = @Frame("Select files:")
vbox = @Box(:v)
push!(win, f)
push!(f, vbox)
runbutton = @Button("Run")
choose = @Button("Choose LR Files")
textbuffer = @TextBuffer()
textview = @TextView()
textstring = "No files selected"
setproperty!(textbuffer,:text,textstring)
setproperty!(textview,:buffer,textbuffer)

textbuffer2 = @TextBuffer()
textview2 = @TextView()
textstring2 = "No location defined"
setproperty!(textbuffer2,:text,textstring2)
setproperty!(textview2,:buffer,textbuffer2)

shift=3

savebutton = @Button("Choose HR Location")

push!(vbox, choose)
push!(vbox,  textview)
push!(vbox, savebutton)
push!(vbox,  textview2)
push!(vbox, runbutton)
ready=false
location=false
showall(win)

inputfiles=["file1"]
outputfile = "file2"


idrun = signal_connect(runbutton, "clicked") do widget
  global ready
  global location
  println(widget, " was clicked!")

  if ready && location
    print("\nRunning SuperResolution\n")
    runSR(inputfiles[1],inputfiles[2],inputfiles[3],outputfile)
   else
    print("File inputs and output location not correctly set.")
   end
end

idchoose = signal_connect(choose, "clicked") do widget
  global ready
  global shift
  global inputfiles
  selection = open_dialog("Pick LR niftii", GtkNullContainer(), ("*.nii","*.nii.gz"), select_multiple=true)

  if size(selection)[1]==shift
    ready=true
    inputfiles=selection
    textstring="Following files have been selected: "
     for d=1:size(selection)[1]
        textstring=string(textstring,"\n",selection[d])
     end
  else
    ready=false
    textstring = string("Please select exactly ",string(shift), " LR files")
  end
  
  setproperty!(textbuffer,:text,textstring)
end

idlocation= signal_connect(savebutton, "clicked") do widget
  global location
  global outputfile
  selection = save_dialog("Pick HR location", GtkNullContainer(),("*.nii","*.nii.gz"))

  if ~(isempty(selection))
    location=true
    outputfile=selection
    textstring2=string("Following location has been selected:\n",selection)
    filesplit=split(outputfile,".")
    if filesplit[end]=="gz"
      if filesplit[end-1]=="nii"
        outputfile=String(filesplit[end-2])
      else
        outputfile=String(filesplit[end-1])
      end
    elseif filesplit[end]=="nii"
      outputfile=String(filesplit[end-1])
    end
  else
    location=false
    textstring2 = "No location defined"
  end
  setproperty!(textbuffer2,:text,textstring2)
end


if !isinteractive()
        c = Condition()
        signal_connect(win, :destroy) do widget
            notify(c)
        end
        wait(c)
end



