close all

addpath('BM3D')
addpath('internal')

resultpath = './denoisingtest/result';
inputpath = '../dataset/';
save_images = true;

if save_images && ~exist(resultpath, 'dir')
    mkdir(resultpath);
end

noise_sigma = [5 10 15 20 25 30 35 40 50 60 70 75 80 90 100];
    
run_bm3dgreydata = true;
grey_prefix = 'bm3d_grey';
grey_images = { ...
    'C.man', 'Cameraman256.png', ...
    'House', 'house.png', ...
    'Peppers', 'peppers256.png', ...
    'Montage', 'montage.png', ...
    'Lena', 'Lena512.png', ...
    'Barbara', 'barbara.png', ...
    'Boats', 'boat.png', ...
    'F.print', 'fingerprint.png', ...
    'Man', 'man.png', ...
    'Couple', 'couple.png' };


greyscaledenoisers = { @(img,sigma)fastdenoiseAndTime(img, sigma, 'BM3DWiener'), ...
                       @(img,sigma)fastdenoiseAndTime(img, sigma, 'NlmWeightedAverage'), ...
                       @(img,sigma)BM3DdenoiseAndTime(img, sigma, 'BM3D'), ...
                       @(img,sigma)deal(zeros(size(img)), 0) };
greyscaledenoisernames = {'GPUBM3D', 'GPUNLM', 'MATLABBM3D', 'MATLABNLM'};

run_bm3dcolordata = true;
color_prefix = 'bm3d_rgb';
color_images = { ...
    'House', 'image_House256rgb.png', ...
    'Peppers', 'image_Peppers512rgb.png', ...
    'Lena', 'image_Lena512rgb.png', ...
    'Baboon', 'image_Baboon512rgb.png', ...
    'F16', 'image_F16_512rgb.png', ...
    'Kodak1', 'kodim01.png', ...
    'Kodak2', 'kodim02.png', ...
    'Kodak3', 'kodim03.png', ...
    'Kodak12', 'kodim12.png' };
colordenoisers = { @(img,sigma)fastdenoiseAndTime(img, sigma, 'BM3DWiener'), ...
                       @(img,sigma)fastdenoiseAndTime(img, sigma, 'NlmWeightedAverage'), ...
                       @(img,sigma)BM3DdenoiseAndTime(img, sigma, 'CBM3D', 'opp'), ...
                       @(img,sigma)deal(zeros(size(img)), 0) };
colordenoisernames = {'GPUBM3D', 'GPUNLM', 'MATLABBM3D', 'MATLABNLM'};


%run greyscale test
if run_bm3dgreydata
    mkdir(sprintf('%s/%s',resultpath,grey_prefix));
    resultfile = fopen(sprintf('%s/%s/result.csv',resultpath,grey_prefix),'w');
    fprintf(resultfile, ',,PSNR');
    for sigma = noise_sigma
        fprintf(resultfile, ',');
    end
    fprintf(resultfile, 'timing\n');
    fprintf(resultfile, 'dataset,method\sigma');
    for sigma = noise_sigma
        fprintf(resultfile, ',%d',sigma);
    end
    fprintf(resultfile, ',');
    for sigma = noise_sigma
        fprintf(resultfile, ',%d',sigma);
    end
    fprintf(resultfile, '\n');
    for img = 0:(length(grey_images)/2-1)
        imgname = grey_images{2*img+1};
        imgfile = sprintf('%s/%s/%s', inputpath, grey_prefix, grey_images{2*img+2});
        gt = double(imread(imgfile));
        gt = gt(:,:,1)/255;
        
        if(save_images)
            fd = sprintf('%s/%s/%s',resultpath,grey_prefix,imgname);
            if ~exist(fd, 'dir')
                mkdir(sprintf('%s/%s/%s',resultpath,grey_prefix,imgname));
            end
            imwrite(gt, sprintf('%s/%s/%s/gt.png',resultpath,grey_prefix,imgname));
        end
        
        results_psnr = zeros(length(greyscaledenoisers),length(noise_sigma));
        results_timing = zeros(length(greyscaledenoisers),length(noise_sigma));
        fprintf(resultfile, '%s,input', imgname);
        for sigmaid = 1:length(noise_sigma)
            sigma = noise_sigma(sigmaid);
            randn('seed',12345);
            % add noise
            noisy = gt + sigma/255*randn(size(gt));
            if(save_images)
                imwrite(noisy, sprintf('%s/%s/%s/%d_noisy.png',resultpath,grey_prefix,imgname,sigma));
            end
            noisy_psnr = comppsnr(noisy, gt);
            fprintf(resultfile, ',%f', noisy_psnr);

            %run denoiser
            for denoiser = 1:length(greyscaledenoisers)
               [denoised time] = greyscaledenoisers{denoiser}(noisy, sigma/255);
               psnr = comppsnr(denoised, gt);
               results_psnr(denoiser,sigmaid) = psnr;
               results_timing(denoiser,sigmaid) = time;
               if(save_images)
                  imwrite(denoised, sprintf('%s/%s/%s/%d_%s.png',resultpath,grey_prefix,imgname,sigma,greyscaledenoisernames{denoiser}));
               end
               fprintf('%s achieved a PSNR of %f (input %f) for %s noisy with %d\n', greyscaledenoisernames{denoiser}, ...
                   psnr, noisy_psnr, imgname, sigma);
            end
        end
        fprintf(resultfile, '\n');
        
        %write denoiser performance to csv
        for denoiser = 1:length(greyscaledenoisers)
            fprintf(resultfile, '%s,%s', imgname, greyscaledenoisernames{denoiser});
            for sigmaid = 1:length(noise_sigma)
                fprintf(resultfile, ',%f', results_psnr(denoiser,sigmaid));
            end
            fprintf(resultfile, ',');
             for sigmaid = 1:length(noise_sigma)
                fprintf(resultfile, ',%f', results_timing(denoiser,sigmaid));
            end
            fprintf(resultfile, '\n');
        end
    end    
end


%run color test
if run_bm3dcolordata
    mkdir(sprintf('%s/%s',resultpath,color_prefix));
    resultfile = fopen(sprintf('%s/%s/result.csv',resultpath,color_prefix),'w');
    fprintf(resultfile, ',,PSNR');
    for sigma = noise_sigma
        fprintf(resultfile, ',');
    end
    fprintf(resultfile, 'timing\n');
    fprintf(resultfile, 'dataset,method\sigma');
    for sigma = noise_sigma
        fprintf(resultfile, ',%d',sigma);
    end
    fprintf(resultfile, ',');
    for sigma = noise_sigma
        fprintf(resultfile, ',%d',sigma);
    end
    fprintf(resultfile, '\n');
    for img = 0:(length(color_images)/2-1)
        imgname = color_images{2*img+1};
        imgfile = sprintf('%s/%s/%s', inputpath, color_prefix, color_images{2*img+2});
        gt = double(imread(imgfile));
        gt = gt/255;
        
        if(save_images)
            mkdir(sprintf('%s/%s/%s',resultpath,color_prefix,imgname));
            imwrite(gt, sprintf('%s/%s/%s/gt.png',resultpath,color_prefix,imgname));
        end
        
        results_psnr = zeros(length(colordenoisers),length(noise_sigma));
        results_timing = zeros(length(colordenoisers),length(noise_sigma));
        fprintf(resultfile, '%s,input', imgname);
        for sigmaid = 1:length(noise_sigma)
            sigma = noise_sigma(sigmaid);
            randn('seed',12345);
            % add noise
            noisy = gt + sigma/255*randn(size(gt));
            if(save_images)
                imwrite(noisy, sprintf('%s/%s/%s/%d_noisy.png',resultpath,color_prefix,imgname,sigma));
            end
            noisy_psnr = comppsnr(noisy, gt);
            fprintf(resultfile, ',%f', noisy_psnr);

            %run denoiser
            for denoiser = 1:length(colordenoisers)
               [denoised time] = colordenoisers{denoiser}(noisy, sigma/255);
               psnr = comppsnr(denoised, gt);
               results_psnr(denoiser,sigmaid) = psnr;
               results_timing(denoiser,sigmaid) = time;
               if(save_images)
                  imwrite(denoised, sprintf('%s/%s/%s/%d_%s.png',resultpath,color_prefix,imgname,sigma,colordenoisernames{denoiser}));
               end
               fprintf('%s achieved a PSNR of %f (input %f) for %s noisy with %d\n', colordenoisernames{denoiser}, ...
                   psnr, noisy_psnr, imgname, sigma);
            end
        end
        fprintf(resultfile, '\n');
        
        %write denoiser performance to csv
        for denoiser = 1:length(colordenoisers)
            fprintf(resultfile, '%s,%s', imgname, colordenoisernames{denoiser});
            for sigmaid = 1:length(noise_sigma)
                fprintf(resultfile, ',%f', results_psnr(denoiser,sigmaid));
            end
            fprintf(resultfile, ',');
             for sigmaid = 1:length(noise_sigma)
                fprintf(resultfile, ',%f', results_timing(denoiser,sigmaid));
            end
            fprintf(resultfile, '\n');
        end
    end    
end

% image = double(imread('lena.bmp'));
% image = image(:,:,1)/255;
% 
% noisesigma = 20/255;
% noisyImage = image + noisesigma*randn(size(image));
% 
% noisyImage(noisyImage<0) = 0;
% noisyImage(noisyImage>1) = 1;
% 
% 
% [denoisedImage stats] = fastdenoise_mex(noisyImage, noisesigma);
% 
% 
% psnr = comppsnr(image, denoisedImage);
% fprintf('denoising took %fms used %fMB of memory achieved %.4fPSNR\n', 1000*stats(1), stats(2)/1024/1024, psnr);
% 
% figure();
% imshow(noisyImage);
% figure();
% imshow(denoisedImage);
