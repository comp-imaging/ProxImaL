close all

addpath('BM3D')
addpath('internal')

resultpath = './denoisingtest/result';
inputpath = '../dataset/';

image = im2double(imread('../dataset/bm3d_rgb/image_Lena512rgb.png'));


noisesigma = 0.1;
noisyImage = image + noisesigma*randn(size(image));
noisyImage = max( min(noisyImage, 1), 0);

%  'BlockSize' (int) dimension of a single patch (BlockSizexBlocksize) possible values 4, 8, 16\n");
%  'TileSize' (int) dimension of one tile (11-21)\n");
%  'ClusterSize' (int) maximum number of elements to end up in a cluster (16, 32, 64)\n");
%  'NumCandidates' (int) number of candidates to be considered (8, 16)\n");
%  'Algorithm' denoising algorithm to be used, possible settings:\n      'NlmWeightedAverage', 'BM3D', 'BM3DWiener', 'NlmAverage', 'SlidingDCT'\n");
%  'Format' data format to run the algorithm on, possible settings:\n      'Greyscale', 'RGB', 'RGBNoConvert', 'LumaChroma', 'YUV420', 'YUV420Fast', 'YUV420FastSubpixel' \n");

algorithm = 'BM3DWiener';
format = 'RGB';
[denoisedImage stats] = fastdenoise_mex(noisyImage, noisesigma, 'Algorithm', algorithm, 'Format', format);

psnr = comppsnr(image, denoisedImage);
fprintf('denoising took %fms used %fMB of memory achieved %.4fPSNR\n', 1000*stats(1), stats(2)/1024/1024, psnr);

figure();
imshow(noisyImage);
figure();
imshow(denoisedImage);
