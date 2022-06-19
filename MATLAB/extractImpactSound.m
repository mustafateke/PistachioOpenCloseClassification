function impactSound = extractImpactSound(data, samplingRate)
% Extract impact sound
startIndex = 0;
for i=1:length(data)
    if(data(i) > 0.085)
        startIndex = i;
        break;
    end
end

impactSound = [];
if startIndex > 0
    for i=1:ceil(samplingRate * 0.0015)
        impactSound(i) = data(startIndex + i - 1);
    end
end
