%{
Sum of error squares classifier.

This will take as inputs:
- A matrix X containing N l-dimensional column vectors
- An N-dimensional row vector y whose ith component contains the class 
  (-1 or +1) where the corresponding vector belongs

returns:
- The estimated parameter vector
%}
function w=SSErr(X,y)
    w = (X*X')\(X*y');