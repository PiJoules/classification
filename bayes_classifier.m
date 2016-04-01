function z=bayes_classifier(m,S,P,X)
    [~,c] = size(m); % l=dimensionality, c=no. of classes
    [~,N] = size(X); % N=no. of vectors
    z = zeros(1, N);
    for i=1:N
        t = zeros(1, c);
        for j=1:c
            t(j)=P(j)*comp_gauss_dens_val(m(:,j), S(:,:,j), X(:,i));
        end
        % Determining the maximum quantity
        [~,z(i)]=max(t);
    end 