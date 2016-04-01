function z=euclidean_classifier(m,X)
    [~,c] = size(m); % l=dimensionality, c=no. of classes
    [~,N] = size(X); % N=no. of vectors
    z = zeros(1, N);
    for i=1:N
        t = zeros(1, c);
        for j=1:c
            t(j)=sqrt((X(:,i)-m(:,j))'*(X(:,i)-m(:,j)));
        end
        % Determining the minimum quantity
        [~,z(i)]=min(t);
    end 