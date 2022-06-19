% Function computes melcepstrum
% Author: Mustafa Teke
% mustafa.teke@gmail.com 
function c = melcepstrum(magnitudes, freqs)
bands = extractBands(magnitudes, freqs, 0, 20000, 44000, 24);
energies = energy(bands);
c = cepstrum(energies, 20);