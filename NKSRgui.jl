# this program reconstructs data with 200 micrometer resolution
push!(LOAD_PATH,"library");
using Read_NIfTI1_real
using myFun
using Gtk
using Gtk.ShortNames

addprocs(Sys.CPU_CORES-nprocs())
# Step 1a: denoise 3 sets of low-resolution images
## define function
@everywhere function qft1d(input::Array{Complex{Float32},2})
    output = fftshift(fft(fftshift(input,2),2),2)
end

@everywhere function qift1d(input::Array{Complex{Float32},2})
    output = ifftshift(ifft(ifftshift(input,2),2),2)
end

@everywhere function smooth_tract_selective_PCA(tmparray::Array{Float32,1},tmparray2D::Array{Float32,4},Lx::Int64,Ly::Int64,Lz::Int64,ddim::Int64, magicnumber::Int64)
    tmparray2Dr = reshape(tmparray2D,ddim,Lx*Ly*Lz)
    tmparraydiff = abs(tmparray2Dr.-tmparray)
    tmparraydiff = sum(tmparraydiff,1)                  
    tmpcorrI = sortperm(tmparraydiff[:])
    tmparray_chosen = tmparray2D[:,tmpcorrI[1:magicnumber]]
    mean_tmparray_chosen = mean(tmparray_chosen,1);
    tmparray_chosen = tmparray_chosen .- mean_tmparray_chosen;
    u,s,v = svd(tmparray_chosen);
    s[2]=s[2]/2f0;
    s[3:end]=0;
    tmparray_chosen2 = u*Diagonal(s)*v';
    tmparray_chosen2 = tmparray_chosen2 .+ mean_tmparray_chosen;
    seed_smoothed_pca = tmparray_chosen2[:,1];
    return seed_smoothed_pca
end

# Step 3: perform super-resolution reconstruction

@everywhere function superresolution(d1::Array{Float32,3},d2::Array{Float32,3},d3::Array{Float32,3},d123::Array{Float32,3}, m0::BitArray{2})
    xs = size(d1)[1];
    ys = size(d1)[2];
    zs = size(d1)[3];
    zsize_new = zs*3 + 2;
    lambda = Float32(0.5); ### HERE 
    #A1 = eye(Float32,zsize_new).*slice_profile[1];
    A1 = eye(Float32,zsize_new).*slice_profile[1] + (circshift(eye(Float32,zsize_new),[0 1]).*slice_profile[2]) + (circshift(eye(Float32,zsize_new),[0 2]).*slice_profile[3]);
    A1[end-1,1]=Float32(0);
    A1[end,1:2]=Float32(0);
    A2 = lambda * eye(Float32, zsize_new);
    A = vcat(A1,A2);
    data_S = zeros(Float32,xs,ys,zsize_new);
    B1 = zeros(Float32,zsize_new,1);
    B2 = zeros(Float32,zsize_new,1);
    for cntx = 1:xs
        for cnty = 1:ys
            if m0[cntx,cnty]                
                array1 = d1[cntx,cnty,:];
                array2 = d2[cntx,cnty,:];
                array3 = d3[cntx,cnty,:];
                array123 = d123[cntx,cnty,:];
                B1[1:3:end-2,1] = array1;
                B1[2:3:end-1,1] = array2;
                B1[3:3:end-0,1] = array3;
                B2[:,1] = lambda*array123;
                B = vcat(B1,B2);
                x = A\B;
                data_S[cntx,cnty,:] = x;
            end
        end
    end
    return data_S
end


# Step 1: load data 
function runSR(filenameA::String,filenameB::String,filenameC::String,outputfile::String)

tic()
headerinfoA = load_nii_header(filenameA); # see the source code in library/Read_NIfTI1_real.jl
dataA = load_nii_data(filenameA, headerinfoA); 
headerinfoB = load_nii_header(filenameB); 
dataB = load_nii_data(filenameB, headerinfoB); 
headerinfoC = load_nii_header(filenameC); 
dataC = load_nii_data(filenameC, headerinfoC); 
xsize,ysize,zsize,dsize=size(dataA);
outputfile=string(outputfile,".nii");

data1f = convert(Array{Float32},dataA);
data2f = convert(Array{Float32},dataB);
data3f = convert(Array{Float32},dataC);

magicnumber = 64;
const xr = 5
const yr = 5
const zr = 2

datadxyz = permutedims(data1f,[4 1 2 3]);
datadxyz_sm = SharedArray(Float32,dsize,xsize,ysize,zsize);

@sync @parallel for cntz = 1:zsize
    for cnty = 1:ysize
        #@inbounds @fastmath @simd for cntx = 1:xdim
        for cntx = 1:xsize
            if datadxyz[1,cntx,cnty,cntz] != Float32(0)
                tmparray = datadxyz[:,cntx,cnty,cntz];
                minx = max(cntx-xr,1);
                maxx = min(cntx+xr,xsize);
                miny = max(cnty-yr,1);
                maxy = min(cnty+yr,ysize);
                minz = max(cntz-zr,1);
                maxz = min(cntz+zr,zsize);
                Lx = maxx-minx+1;
                Ly = maxy-miny+1;
                Lz = maxz-minz+1;
                tmparray2D = datadxyz[:,minx:maxx,miny:maxy,minz:maxz];
                tmparray_chosen_recovered = smooth_tract_selective_PCA(tmparray,tmparray2D,Lx,Ly,Lz,dsize, magicnumber);            
                datadxyz_sm[:,cntx,cnty,cntz]=convert(Array{Float32},abs(tmparray_chosen_recovered));
            end
        end
    end
end

data1f_denoised = permutedims(datadxyz_sm,[2,3,4,1]);

datadxyz = permutedims(data2f,[4 1 2 3]);
datadxyz_sm = SharedArray(Float32,dsize,xsize,ysize,zsize);

@sync @parallel for cntz = 1:zsize
    for cnty = 1:ysize
        #@inbounds @fastmath @simd for cntx = 1:xdim
        for cntx = 1:xsize
            if datadxyz[1,cntx,cnty,cntz] != Float32(0)
                tmparray = datadxyz[:,cntx,cnty,cntz];
                minx = max(cntx-xr,1);
                maxx = min(cntx+xr,xsize);
                miny = max(cnty-yr,1);
                maxy = min(cnty+yr,ysize);
                minz = max(cntz-zr,1);
                maxz = min(cntz+zr,zsize);
                Lx = maxx-minx+1;
                Ly = maxy-miny+1;
                Lz = maxz-minz+1;
                tmparray2D = datadxyz[:,minx:maxx,miny:maxy,minz:maxz];
                tmparray_chosen_recovered = smooth_tract_selective_PCA(tmparray,tmparray2D,Lx,Ly,Lz,dsize, magicnumber);            
                datadxyz_sm[:,cntx,cnty,cntz]=convert(Array{Float32},abs(tmparray_chosen_recovered));
            end
        end
    end
end
data2f_denoised = permutedims(datadxyz_sm,[2,3,4,1]);

datadxyz = permutedims(data3f,[4 1 2 3]);
datadxyz_sm = SharedArray(Float32,dsize,xsize,ysize,zsize);
@sync @parallel for cntz = 1:zsize
    for cnty = 1:ysize
        #@inbounds @fastmath @simd for cntx = 1:xdim
        for cntx = 1:xsize
            if datadxyz[1,cntx,cnty,cntz] != Float32(0)
                tmparray = datadxyz[:,cntx,cnty,cntz];
                minx = max(cntx-xr,1);
                maxx = min(cntx+xr,xsize);
                miny = max(cnty-yr,1);
                maxy = min(cnty+yr,ysize);
                minz = max(cntz-zr,1);
                maxz = min(cntz+zr,zsize);
                Lx = maxx-minx+1;
                Ly = maxy-miny+1;
                Lz = maxz-minz+1;
                tmparray2D = datadxyz[:,minx:maxx,miny:maxy,minz:maxz];
                tmparray_chosen_recovered = smooth_tract_selective_PCA(tmparray,tmparray2D,Lx,Ly,Lz,dsize, magicnumber);            
                datadxyz_sm[:,cntx,cnty,cntz]=convert(Array{Float32},abs(tmparray_chosen_recovered));
            end
        end
    end
end

data3f_denoised = permutedims(datadxyz_sm,[2,3,4,1]);

data1f = data1f_denoised;
data2f = data2f_denoised;
data3f = data3f_denoised;

mask = sum(sum(data1f + data2f + data3f,4),3);
mask2d = mask .> 0.1*maximum(mask[:]);


# slice_profile [0.33, 0.34, 0.33] works well;
@everywhere slice_profile = convert(Array{Float32},[0.335, 0.33, 0.335]);

# Step 2: simply combine 3 data sets to create a big-matrix data set 
data1f_new = zeros(Float32,xsize,ysize,zsize*3+2,dsize);
data2f_new = zeros(Float32,xsize,ysize,zsize*3+2,dsize);
data3f_new = zeros(Float32,xsize,ysize,zsize*3+2,dsize);
data1f_new[:,:,1:3:end-2,:] = data1f*slice_profile[1];
data1f_new[:,:,2:3:end-2,:] = data1f*slice_profile[2];
data1f_new[:,:,3:3:end-2,:] = data1f*slice_profile[3];
data2f_new[:,:,2:3:end-1,:] = data2f*slice_profile[1];
data2f_new[:,:,3:3:end-1,:] = data2f*slice_profile[2];
data2f_new[:,:,4:3:end-1,:] = data2f*slice_profile[3];
data3f_new[:,:,3:3:end-0,:] = data3f*slice_profile[1];
data3f_new[:,:,4:3:end-0,:] = data3f*slice_profile[2];
data3f_new[:,:,5:3:end-0,:] = data3f*slice_profile[3];
data1f2f3f_new = (data1f_new+data2f_new+data3f_new);


zsize_new = zsize*3 + 2;

#data_all = zeros(Float32,xsize,ysize,zsize_new,dsize);
#datadxyz_sm = SharedArray{Float32,4}((ddim,xdim,ydim,zdim))
data_all = SharedArray(Float32,xsize,ysize,zsize_new,dsize);


@sync @parallel for cntd = 1:dsize
    datacombined = superresolution(data1f[:,:,:,cntd],data2f[:,:,:,cntd],data3f[:,:,:,cntd],data1f2f3f_new[:,:,:,cntd],mask2d[:,:,1,1]);
    data_all[:,:,:,cntd] = datacombined;
end




# Step 4: denoising

datadxyz = permutedims(data_all,[4 1 2 3]);
datadxyz_sm = SharedArray(Float32,dsize,xsize,ysize,zsize_new);

@sync @parallel for cntz = 1:zsize_new
    for cnty = 1:ysize
        #@inbounds @fastmath @simd for cntx = 1:xdim
        for cntx = 1:xsize
            if datadxyz[1,cntx,cnty,cntz] != Float32(0)
                tmparray = datadxyz[:,cntx,cnty,cntz];
                minx = max(cntx-xr,1);
                maxx = min(cntx+xr,xsize);
                miny = max(cnty-yr,1);
                maxy = min(cnty+yr,ysize);
                minz = max(cntz-zr,1);
                maxz = min(cntz+zr,zsize_new);
                Lx = maxx-minx+1;
                Ly = maxy-miny+1;
                Lz = maxz-minz+1;
                tmparray2D = datadxyz[:,minx:maxx,miny:maxy,minz:maxz];
                tmparray_chosen_recovered = smooth_tract_selective_PCA(tmparray,tmparray2D,Lx,Ly,Lz,dsize, magicnumber);            
                datadxyz_sm[:,cntx,cnty,cntz]=convert(Array{Float32},abs(tmparray_chosen_recovered));
            end
        end
    end
end
data_sm = permutedims(datadxyz_sm,[2,3,4,1]);

data_sm = 32000. * data_sm / maximum(data_sm[:]);
data_sm = convert(Array{Int16}, round.(data_sm));
headerinfo = deepcopy(headerinfoA)
headerinfo["dim"][4] = zsize*3+2
headerinfo["pixdim"][4] = headerinfo["pixdim"][4]/3f0
headerinfo["srow_z"][3] = headerinfo["srow_z"][3]/3f0
headerinfo["srow_z"][4] = headerinfo["srow_z"][4]/3f0
headerinfo["cal_max"] = Float32(32000);
write_nii_header(outputfile, headerinfo);
fid = open(outputfile,"a");
write(fid, data_sm);
close(fid);
print(outputfile, " successfully created. Super Resolution completed.")
toc()
end



#GUI
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
    print("running")
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



