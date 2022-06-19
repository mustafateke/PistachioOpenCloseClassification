function columnIndices = findLargestElementIndices(x, numOfElements)
maxElements = max(x);
for i=1:numOfElements
    [num, indice] = max(maxElements);
    maxElements(indice) = NaN;
    columnIndices(i) = indice;
end