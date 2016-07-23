function [ img time ] = fastdenoiseAndTime( inputImage, sigma, algorithm, blocksize, tilesize )

if(~exist('algorithm', 'var'))
    algorithm = 'BM3DWiener';
end

if(~exist('blocksize', 'var'))
    blocksize = 8;
end

if(~exist('tilesize', 'var'))
    tilesize = 15;
end

[img stats] = fastdenoise_mex(inputImage, sigma, 'Algorithm', algorithm, 'BlockSize', blocksize, 'TileSize', tilesize);
time = stats(1);

end

