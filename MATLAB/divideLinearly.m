function values = divideLinearly(startNum, endNum, numberOfPieces)
% Girilen aral��� lineer olarak par�alara ay�r�r
widthOfAPiece = (endNum - startNum) / numberOfPieces;
values = ([0:numberOfPieces] * widthOfAPiece) + startNum;
