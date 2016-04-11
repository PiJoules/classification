%{
LMS algorithm.

This will take as inputs
- Amatrix X containing N l-dimensional column vectors
- An N-dimensional row vector y whose ith component contains the class 
  (-1 or +1) where the corresponding vector is assigned
- An initial value vector w_ini for the parameter vector

returns:
- The estimated parameter vector
%}
function w=LMSalg(X, y, w_ini)
    [~, N] = size(X);
    rho = 0.1;  % Learning rate initialization
    w = w_ini;  % Initialization of the parameter vector
    for i=1:N
        w = w + rho * (y(i) - X(:,i)' * w) * X(:,i);
    end