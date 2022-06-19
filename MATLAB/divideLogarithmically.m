function values = divideLogarithmically(startNum, endNum, numberOfPieces)
% Girilen aral��� logaritmik olarak par�alara ay�r�r
widthOfAPiece = (log10(endNum) - log10(startNum)) / (numberOfPieces);
valuesInLogScale = ([0:numberOfPieces] * widthOfAPiece) + log10(startNum);
values = 10 .^ valuesInLogScale;