%{
Generates a two-class, two-dimensional data set using four normal 
distributions, with covariance matrices S(i)=s?I,i=1,...,4, where I is 
the 2x2 identity matrix.

The vectors that stem from the first two distributions belong to class 1, 
while the vectors originating from the other two distributions belong to 
class 1.

Arguments:
- 2x4 matrix, m, whose ith column is the mean vector of the ith 
  distribution
- Variance parameter s, mentioned before
- Number of the points, N, which will be generated from each distribution

Returns:
- Array, X, of dimensionality 2 x 4 ? N, whose first group of N vectors 
  stem from the first distribution, the second group from the second 
  distribution and so on
- 4 ? N dimensional row vector y with values 1 or 1, indicating the 
  classes to which the corresponding data vectors in X belong.
%}
function [x,y]=data_generator(m, s, N)
    S = s * eye(2);
    [~, c] = size(m);
    x = []; % Creating the training set
    for i = 1:c
        x = [x mvnrnd(m(:,i)',S,N)'];
    end
    y=[ones(1,N) ones(1,N) -ones(1,N) -ones(1,N)];