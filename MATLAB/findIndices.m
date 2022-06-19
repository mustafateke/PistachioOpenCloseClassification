function indices = findIndices(x, values)
%indices = find (x >= values, 1, 'first');
for i = 1:length(values)
    indices(i) = find(x >= values(i), 1, 'first');
end