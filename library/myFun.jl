module myFun
    
    export qft1d
    export qift1d
    export smooth_tract_selective_PCA
    export denoisingDWI
    export superresolution
    export superresolution_lambda
    export superresolution_lambda_better_boundary_condition

    function qft1d(input::Array{Complex{Float32},2})
        output = fftshift(fft(fftshift(input,2),2),2)
    end
    function qift1d(input::Array{Complex{Float32},2})
        output = ifftshift(ifft(ifftshift(input,2),2),2)
    end
    function smooth_tract_selective_PCA(tmparray::Array{Float32},tmparray2Dr::Array{Float32},magicnumber::Int64)
        tmparraydiff = abs.(tmparray2Dr.-tmparray)
        tmparraydiff = sum(tmparraydiff,1)                  
        tmpcorrI = sortperm(tmparraydiff[:])
        tmparray_chosen = tmparray2Dr[:,tmpcorrI[1:magicnumber]]
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
    function denoisingDWI(data::Array{Float32,4})
        magicnumber = 64;
        const xr = 5
        const yr = 5
        const zr = 2
        xdim,ydim,zdim,ddim=size(data);
        datadxyz = permutedims(data,[4 1 2 3]);
        datadxyz_sm = SharedArray{Float32,4}((ddim,xdim,ydim,zdim)); # this is for parallel computation
#         datadxyz_sm = zeros(Float32, ddim,xdim,ydim,zdim); 
        @sync @parallel for cntz = 1:zdim
            for cnty = 1:ydim
                @inbounds @fastmath @simd for cntx = 1:xdim
                    tmparray = datadxyz[:,cntx,cnty,cntz];
                    minx = max(cntx-xr,1);
                    maxx = min(cntx+xr,xdim);
                    miny = max(cnty-yr,1);
                    maxy = min(cnty+yr,ydim);
                    minz = max(cntz-zr,1);
                    maxz = min(cntz+zr,zdim);
                    Lx = maxx-minx+1;
                    Ly = maxy-miny+1;
                    Lz = maxz-minz+1;
                    tmparray2D = datadxyz[:,minx:maxx,miny:maxy,minz:maxz];
                    tmparray2Dr = reshape(tmparray2D,ddim,Lx*Ly*Lz);
                    tmparray_chosen_recovered = smooth_tract_selective_PCA(tmparray,tmparray2Dr,magicnumber);            
                    datadxyz_sm[:,cntx,cnty,cntz]=convert(Array{Float32},abs.(tmparray_chosen_recovered));
                end
            end
        end
        data_sm = permutedims(datadxyz_sm,[2,3,4,1]);
        return data_sm
    end

    function superresolution(d1::Array{Float32,3},d2::Array{Float32,3},d3::Array{Float32,3},d123::Array{Float32,3},slice_profile::Array{Float32})
        xs = size(d1)[1];
        ys = size(d1)[2];
        zs = size(d1)[3];
        zsize_new = zs*3 + 2;
        lambda = Float32(0.5); ### HERE 
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
        return data_S
    end

    function superresolution_lambda(d1::Array{Float32,3},d2::Array{Float32,3},d3::Array{Float32,3},d123::Array{Float32,3},slice_profile::Array{Float32}, lambda::Float32)
        xs = size(d1)[1];
        ys = size(d1)[2];
        zs = size(d1)[3];
        zsize_new = zs*3 + 2;
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
        return data_S
    end

    function superresolution_lambda_better_boundary_condition(d1::Array{Float32,3},d2::Array{Float32,3},d3::Array{Float32,3},d123::Array{Float32,3},slice_profile::Array{Float32}, lambda::Float32)
        xs = size(d1)[1];
        ys = size(d1)[2];
        zs = size(d1)[3];
        zsize_new = zs*3 + 2;
        A1 = eye(Float32,zsize_new).*slice_profile[1] + (circshift(eye(Float32,zsize_new),[0 1]).*slice_profile[2]) + (circshift(eye(Float32,zsize_new),[0 2]).*slice_profile[3]);
        A1 = A1[1:end-2,:];
        A2 = lambda * eye(Float32, zsize_new);
        A = vcat(A1,A2);
        data_S = zeros(Float32,xs,ys,zsize_new);
        B1 = zeros(Float32,zs*3,1);
        B2 = zeros(Float32,zsize_new,1);
        for cntx = 1:xs
            for cnty = 1:ys
                array1 = d1[cntx,cnty,:];
                array2 = d2[cntx,cnty,:];
                array3 = d3[cntx,cnty,:];
                array123 = d123[cntx,cnty,:];
                B1[1:3:end,1] = array1;
                B1[2:3:end,1] = array2;
                B1[3:3:end,1] = array3;
                B2[:,1] = lambda*array123;
                B = vcat(B1,B2);
                x = A\B;
                data_S[cntx,cnty,:] = x;                   
            end
        end
        return data_S
    end



end
