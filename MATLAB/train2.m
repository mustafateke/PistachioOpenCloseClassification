function [w0, wa, w1, wb, ukX, ukC, Xm, Cm] = train2(numberOfTrainingSamples, data)
%global w0, w1, wa, wb, ukX, ukC;

trainedSamples = 0;
X = [];
C = [];
samplingRate = 192000;
numOfChannels = 1;
recordDuration = 5;
while(trainedSamples < numberOfTrainingSamples)
    
%     data=wavrecord(samplingRate * recordDuration ,samplingRate, numOfChannels);
% 
%     impactSound = extractImpactSound(data, samplingRate);
    for i=1:288
    impactSound(1,i)= data(i, trainedSamples+1);
    end

    if (length(impactSound) > 0)
        
        
        % Calculate FFT
        inp = (impactSound' .* hann(length(impactSound)));
        hannWind=hann(length(impactSound));
        impactSoundTr=impactSound';
        input = impactSoundTr .*hannWind ;
        fftVal= fft(input,256);
        freq = abs(fftVal);
        freqs = samplingRate*(0:128)/256;
        magnitudes = freq(1:129);
        magnitudes = normalize(magnitudes);


        impactSound = normalize(impactSound);
        % Concatenate sound vectors
        X = [X impactSound'];

        % Compute mel-cepstrum
        impactCepstrum = melcepstrum(magnitudes, freqs);
        impactCepstrum = normalize(impactCepstrum);

        % Concatenate cepstrum vectors
        C = [C impactCepstrum'];
        
        trainedSamples = trainedSamples + 1
    end
    
    if(trainedSamples == numberOfTrainingSamples)
        % All samples are recorded
        % Calculate feature vectors and exit
        
       [w0, wa, w1, wb, ukX, ukC, Xm, Cm] = findFeatureVectors(X, C);
       save('trainingSet', 'X', 'C');
    end
end