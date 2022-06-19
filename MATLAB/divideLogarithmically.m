function values = divideLogarithmically(startNum, endNum, numberOfPieces)
% Girilen aralýðý logaritmik olarak parçalara ayýrýr
widthOfAPiece = (log10(endNum) - log10(startNum)) / (numberOfPieces);
valuesInLogScale = ([0:numberOfPieces] * widthOfAPiece) + log10(startNum);
values = 10 .^ valuesInLogScale;