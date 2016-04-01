%{
Args:
- Set of N vectors packed as columns of a matrix Z
- N dimensional vector containing the classes where each vector in Z 
  belongs
- The value for the parameter k of the classifier
- Set of N vectors packed as columns in the matrix X

Returns:
- An N-dimensional vector whose ith component contains the class where the 
  corresponding vector of X is assigned, according to the k-nearest 
  neighbor classifier.
%}
function z=k_nn_classifier(Z,v,k,X)
    [~,N1]=size(Z);
    [~,N]=size(X);
    c = max(v); % The number of classes
    % Computation of the (squared) Euclidean distance
    % of a point from each reference vector
    z = zeros(1, N);
    for i=1:N
        dist = sum((X(:,i)*ones(1,N1)-Z).^ 2);
        % Sorting the above distances in ascending order
        [~, nearest]=sort(dist);
        % Counting the class occurrences among the k-closest
        % reference vectors Z(:,i)
        refe=zeros(1,c); % Counting the reference vectors per class
        for q=1:k
            class=v(nearest(q));
            refe(class)=refe(class)+1;
        end
        [~, z(i)]=max(refe);
    end