function normalizedData =  normalize(x)
% Normalize extracted sound
normVal = norm(x);
normalizedData = abs(x) / normVal;