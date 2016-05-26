clear;
close all;
clc;

rng('default');

% Maximum number of iterations
max_iters = 100;
k = 5;

% Data generation
N = 100;
m = [[10 0]' [-10 0]' [5, 5]', [0 0]' [0 0]'];
[features, classes] = size(m);
S = eye(features);
for i=2:classes
    S(:, :, i) = S(:, :, 1);
end
p = ones(classes, 1) / length(classes);
% X is our data set
[X, ~] = generate_gauss_classes(m, S, p, N);

centroids = rand(features, k);
oldcentroids = zeros(features, k);

iters = 0;
while ~isequal(centroids, oldcentroids) && (iters < max_iters)
    oldcentroids = centroids;

    % Find nearest mean for each elem
    clusters = zeros(1, length(X));
    for i=1:length(X)
        x = X(:,i);
        min_dist = inf;
        best_j = 0;
        for j=1:length(centroids)
            v = x - centroids(:,j);
            d = norm(v);
            if d < min_dist
               min_dist = d; 
               best_j = j;
            end
        end
        clusters(i) = best_j;
    end
    
    for i=1:length(centroids)
        vectors = [];
        for j=1:length(X)
            cluster = clusters(:, j);
            if cluster == i
                vectors(:, length(vectors) + 1) = X(:, j);
            end
        end
        centroids(:, i) = mean(vectors')';
    end

    iters = iters + 1;
end

% MATLAB kmeans
[idx, C] = kmeans(X', k);
figure;
plot(X(1,:), X(2,:), '.r', centroids(1,:), centroids(2,:), 'ob',...
     C(:, 1), C(:, 2), 'og');
legend('Data Set', 'My KMeans', 'MATLAB KMeans');
grid on;
title('KMeans Results');
