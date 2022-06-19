function bands = extractBands(magnitudes, freqs, startFreq, breakPoint, endFreq, numberOfPieces)
valuesLinear = divideLinearly(startFreq, breakPoint, numberOfPieces /2);
valuesLogarithmic = divideLogarithmically(breakPoint, endFreq, numberOfPieces / 2);
boundaries = [valuesLinear valuesLogarithmic(2:length(valuesLogarithmic))];
boundaryIndices = findIndices(freqs, boundaries);
bands = zeros(length(boundaryIndices) - 1, length(magnitudes));

for i = 1:length(boundaryIndices) - 1
    if(i == 1)    
        l = boundaryIndices(i+1) - boundaryIndices(i) + 1;
        rangeVals = magnitudes(boundaryIndices(i) : boundaryIndices(i+1));
        bands(i,1:l) = rangeVals;
    else
        l = boundaryIndices(i+1) - boundaryIndices(i);
        rangeVals = magnitudes(boundaryIndices(i) + 1 : boundaryIndices(i+1));
        bands(i,1:l) = rangeVals;
    end
end