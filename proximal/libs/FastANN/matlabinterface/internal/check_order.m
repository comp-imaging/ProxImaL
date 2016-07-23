function [n_out, w, trivalwin] = check_order(n_in)
%CHECK_ORDER Checks the order passed to the window functions.
% [N,W,TRIVALWIN] = CHECK_ORDER(N_ESTIMATE) will round N_ESTIMATE to the
% nearest integer if it is not already an integer. In special cases (N is
% [], 0, or 1), TRIVALWIN will be set to flag that W has been modified.

%   Copyright 1988-2002 The MathWorks, Inc.
%   $Revision: 1.6.4.2 $  $Date: 2009/05/23 08:16:17 $

w = [];
trivalwin = 0;

if ~(isnumeric(n_in) & isfinite(n_in)),
    error(generatemsgid('InvalidOrder'),'The order N must be finite.');
end

% Special case of negative orders:
if n_in < 0,
   error(generatemsgid('InvalidOrder'),'Order cannot be less than zero.');
end

% Check if order is already an integer or empty
% If not, round to nearest integer.
if isempty(n_in) | n_in == floor(n_in),
   n_out = n_in;
else
   n_out = round(n_in);
   warning(generatemsgid('InvalidOrder'),'Rounding order to nearest integer.');
end

% Special cases:
if isempty(n_out) | n_out == 0,
   w = zeros(0,1);               % Empty matrix: 0-by-1
   trivalwin = 1; 
elseif n_out == 1,
   w = 1;
   trivalwin = 1;   
end