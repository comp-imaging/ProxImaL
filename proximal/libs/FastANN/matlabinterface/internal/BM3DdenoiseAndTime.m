function [ img time ] = BM3DdenoiseAndTime( img, sigma, algorithm, cspace )

tic();
if strcmp(algorithm, 'BM3D')
    [ps,img] = BM3D(img, img, 255*sigma, 'np', 0);
elseif strcmp(algorithm, 'CBM3D')
    [ps,img] = CBM3D(img, img, 255*sigma, 'np', 0, cspace);                           
end
time = toc();

end

