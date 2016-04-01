%{
Function that generates a data set of Nl-dimensional vectors that stem 
from c different Gaussian distributions N(mi,Si), with corresponding a 
priori probabilities Pi,i=1,...,c.

Args:
- m is an lxc matrix, the i-th column of which is the mean vector of the
  i-th class distribution.
- S is an lxlxc (three-dimensional) matrix, whose ith two-dimensional lxl
  component is the covariance of the distribution of the ith class. In 
  MATLAB S(:, :, i) denotes the i-th two-dimensional lxl matrix of S.
- P is the c dimensional vector that contains the a priori probabilities 
  of the classes. mi, Si, Pi,and c are provided as inputs.

Return:
- A matrix X with (approximately) N columns, each column of which is
  an l-dimensional data vector.
- A row vector y whose ith entry denotes the class from which the ith
  data vector stems.
%}
function [X, y] = generate_gauss_classes(m,S,P,N)
    [~, c] = size(m);
    X = [];
    y = [];
    for j=1:c
        % Generating the [p(j)*N)] vectors from each distribution
        t = mvnrnd(m(:,j), S(:,:,j), fix(P(j)*N));
        % The total number of points may be slightly less than N % due to the fix operator
        X = [X; t];
        y = [y ones(1,fix(P(j)*N))*j];
    end
    X = X';