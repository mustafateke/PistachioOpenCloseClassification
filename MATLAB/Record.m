function [signal, freqs, magnitudes] = Record(trialName)
samplingRate = 192000;
numOfChannels = 1;
recordDuration = 5;

data=wavrecord(samplingRate * recordDuration ,samplingRate, numOfChannels);

% Extract impact sound
startIndex = 0;
for i=1:length(data)
    if(data(i) > 0.085)
        startIndex = i;
        break;
    end
end

% Draw extracted signal in time-domain
dataFistik = [];
timeFistik = [];
if startIndex > 0
    for i=1:ceil(samplingRate * 0.0015)
        dataFistik(i) = data(startIndex + i - 1);
        timeFistik(i) = (i-1) * 1000 / samplingRate;
    end
end

if (length(dataFistik) > 0)
    signal = dataFistik;
    subplot(2,1,1);
    handle = plot(timeFistik, dataFistik);
    xlabel('Zaman (ms)')
    ylabel('Sinyal seviyesi (Volt)')
    grid on

    % Frequency spectrum
    freq = abs(fft(dataFistik, 256));
    f = samplingRate*(0:128)/256;
    freqs = f;
    Pfreq = freq.* conj(freq) / 256;
    %Graph the first 257 points (the other 255 points are redundant) on a meaningful frequency axis: f = 1000*(0:256)/512;
    subplot(2,1,2);
    magnitudes = freq(1:129);
    plot(f, freq(1:129));
    title('Frekans Bileþenleri')
    xlabel('Frekans (Hz)')
    ylabel('Büyüklük')
    grid on
    %pngName = [trialName, '.png'];
    %figName = [trialName, '.fig'];
    %saveas(handle, pngName);
    %saveas(handle, figName);
end