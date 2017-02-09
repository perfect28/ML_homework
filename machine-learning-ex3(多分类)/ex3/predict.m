function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% for i = 1:m
%     z2 = Theta1*[1;X(i,:)'];%25*1
%     a2 = sigmoid(z2);
%     a2 = [1;a2]; 
%     z3 = Theta2*a2;%10*1
%     a3 = sigmoid(z3);
%     [Y,p(i)] = max(a3);
% end

X = [ones(m,1) X];%m*401

z2 = X*Theta1';%m*401+401*25=m*25,每一行都是一组数据的z2

a2 = [ones(m,1) sigmoid(z2)];%m*26,每一行都是一组数据的a2

z3 = a2*Theta2';%m*26+26*10=m*10,每一行都是一组数据的z3

a3 = sigmoid(z3);%m*10,每一行是一组数据取各个值的概率

[Y,p] = max(a3,[],2);%取行的最大值

% =========================================================================


end
