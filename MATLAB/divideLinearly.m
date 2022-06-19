function values = divideLinearly(startNum, endNum, numberOfPieces)
% Girilen aralýðý lineer olarak parçalara ayýrýr
widthOfAPiece = (endNum - startNum) / numberOfPieces;
values = ([0:numberOfPieces] * widthOfAPiece) + startNum;
