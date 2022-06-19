
%global w0, w1, wa, wb, ukX, ukC;

trainedSamples = 0;
X = [];
C = [];

        load('recordings.mat', 'X', 'C');
        data = X';
        save('data.mat', 'RecordedImpactSound', '-ascii');
        save('C.txt', 'C', '-ascii');

        
       [w0, wa, w1, wb, ukX, ukC, Xm, Cm] = findFeatureVectors(X, C);
       save('trainingSet', 'X', 'C');

