function [ psnr, mse ] = comppsnr( x, y )
%PSNR Summary of this function goes here
%   Detailed explanation goes here

xl = x(~isnan(x) & ~isnan(y));
yl = y(~isnan(x) & ~isnan(y));

maxval = 1.0;

if max(xl(:)) > 10 && max(yl(:)) > 10
    maxval = 255;
elseif max(xl(:)) > 10 
    xl = xl / 255;
elseif max(yl(:)) > 10
    yl = yl / 255;
end

diff2 = (xl-yl).^2;
mse = mean(diff2(:));
psnr = 10.0 * log10(maxval*maxval / mse);
end
