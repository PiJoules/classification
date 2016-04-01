function clas_error=compute_error(y,y_est)
    [~,N]=size(y); % N=no.ofvectors
    clas_error=0; % Counting the misclassified vectors
    for i=1:N
        if y(i) ~= y_est(i)
            clas_error=clas_error+1;
        end
    end
    % Computing the classification error
    clas_error=clas_error/N;