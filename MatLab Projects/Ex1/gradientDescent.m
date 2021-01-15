function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

    % ====================== YOUR CODE HERE ======================
for iter = 1:num_iters
    for i=1:m
        h=theta'*X(i,:)';
        theta= theta-(alpha/m)*(h-y(i,:))*X(i,:)';  %theta ist Vektor mit theat0,1,2 ...j
                                                    %X ist Vektor mit X0,1,2 ...j
    end
 
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end 
    
end   

