function [w0, wa, w1, wb, ukX, ukC, Xm, Cm] = findFeatureVectors(X, C)

[ukX,ValuesX, Xm, XX] = pc_evectors(X, 10);
[ukC,ValuesC, Cm, CC] = pc_evectors(C, 10);

% We have representative eigenvectors in hand
% Now calculate projections  w0, w1, wa, wb

%w0 = ukX(:,1:10)'*(XX);
%wa = ukC(:,1:10)'*(CC);

w0 = zeros(10, 1);
wa = zeros(10, 1);
w1 = zeros(10, 1);
wb = zeros(10, 1);

for j = 1 : 20 % L/2 = 20
        w0 = w0 + ukX' * XX(:, j);
        wa = wa + ukC' * CC(:, j);
end

w0 = w0 / 20;
wa = wa / 20;

for j = 21 : 40 % L/2 = 20
   w1 = w1 + ukX' * XX(:, j);
   wb = wb + ukC' * CC(:, j);
end

w1 = w1 / 20;
wb = wb / 20;