%function classify(w0, w1, wa, wb, ukX, ukC, Xm, Cm)
% Author: Mustafa Teke
% mustafa.teke@gmail.com 
samplingRate = 192000;
numOfChannels = 1;
recordDuration = 5;
open = 0;
closed = 0;
total = 0;
for row=1:40  


    for i=1:288
    impactSound(1,i)= RecordedImpactSound(i, row);
    end

    if (length(impactSound) > 0)
        
        % Calculate FFT
        freq = abs(fft(impactSound' .* hann(length(impactSound)), 256));
        freqs = samplingRate*(0:128)/256;
        magnitudes = freq(1:129);
        magnitudes = normalize(magnitudes);


        % Compute mel-cepstrum
        impactCepstrum = melcepstrum(magnitudes, freqs);
        impactCepstrum = normalize(impactCepstrum);
        
        impactSound = normalize(impactSound);
        
        % Find projections
        wx = ukX' * (impactSound' - Xm);
        wc = ukC' * (impactCepstrum' - Cm);        
        
        distanceOpen = norm(wx- w0) + norm(wc - wa);
        distanceClosed = norm(wx- w1) + norm(wc - wb);            
        
        if(distanceOpen < distanceClosed)
            fprintf(1,'Open\n');
            open = open + 1;
        else
            fprintf(1,'Closed\n');
            closed = closed + 1;
        end
        
        total = total + 1;
        fprintf('Open : %d, Closed %d , Total : %d, \n', open, closed, total);
    end
end