% Cepstrum Compute function
% e: Signal
% k: dimension
% Author: Mustafa Teke
% mustafa.teke@gmail.com 
function c = cepstrum(e, k)
m = length(e);
for j = 1:k
    c(j) = 0;
    for i = 1:m
        c(j) = c(j) + ( log10(e(i)) * cos(j * (i - 0.5) * pi / m) ) ;
    end
end