%{
Function for the perceptron algorithm.

This will take as inputs:
- A matrix X containing N l-dimensional column vectors
- An N -dimensional row vector y, whose ith 
component contains the class (-1 or +1) where the corresponding vector 
belongs
- An initial value vector w_ini for the parameter vector

returns:
  The estimated parameter vector.
%}
function w=perce(X, y, w_ini)
    [l, N] = size(X);
    max_iter = 10000;  % Maximum allowable number of iterations
    rho = 0.05;
    w = w_ini;
    iter = 0;
    mis_clas = N;
    while (mis_clas > 0) && (iter < max_iter)
        % Learning rate
        % Initialization of the parameter vector % Iteration counter
        % Number of misclassified vectors
        iter = iter + 1;
        mis_clas = 0;
        gradi = zeros(l, 1);  % Computation of the "gradient"
        % term
        for i=1:N
          if ((X(:,i)' * w) * y(i)) < 0
            mis_clas = mis_clas + 1;
            gradi = gradi + (rho * (-y(i) * X(:,i)));
          end
        end
        w = w - (rho * gradi); % Updating the parameter vector
    end