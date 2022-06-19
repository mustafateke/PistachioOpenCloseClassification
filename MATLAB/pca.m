function w = pca(x)
X = x-mean(x);
[u, lamba] = eig(X * X');
w = u * X;