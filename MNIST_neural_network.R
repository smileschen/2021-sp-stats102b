## run the code in handwriting.R to create the functions load_mnist and show_digit
load_mnist()

# look at the values in the jth row of the training matrix
j <- 2
train$y[j] # the label
train$x[j, ] # the digit
# If you set your console so it displays 28 columns of values, 
# the display of the vector of data looks sort of like the digit itself
show_digit(train$x[j, ], asp = 1)

# vectorize the labels
# here, we take the digit labels (y values) in the dataset
# and turn them from digits between 0 and 9 into a vector that is
# of length 10, where the j+1th digit is a 1, and the rest are 0s
vectorize <- function(j) {
  k <- rep(0, 10)
  k[j + 1] <- 1
  k
}

y <- t( apply(matrix(train$y), 1, vectorize) )
head(y) # check results
train$y[1:6] # compare output

# plot the digit in the jth row of the training matrix
show_digit(train$x[j, ]) 

## define the sigmoid and sigmoid prime functions
sigmoid <- function(Z){
  1/(1 + exp(-Z))
}

sigmoidprime <- function(z){
  exp(-z)/((1+exp(-z))^2)
}

cost <- function(y,y_hat){
  0.5*sum((y - y_hat)^2)
}

# define the size our our neural network
input_layer_size <- 784
output_layer_size <- 10
hidden_layer_size <- 30 # arbitrarily chosen

set.seed(1)
# set some initial weights: initialized to be between -.5 and +.5
W_1 <- matrix(runif(input_layer_size * hidden_layer_size)-.5, nrow = input_layer_size, ncol = hidden_layer_size)
W_2 <- matrix(runif(hidden_layer_size * output_layer_size)-.5, nrow = hidden_layer_size, ncol = output_layer_size)

# stochastic batch gradient descent means that we will not use the entire training set
# every single time when we calculate the gradient. That would be too computationally
# expensive (60000 rows).
# so instead, we will sample a selection of rows and use those to calculate the gradient
# and train our neural network.
set.seed(1)
n <- dim(y)[1] # 60,000
batch_size <-  10 # arbitrarily chosen
j <- sample(1:n)  # shuffle the indices 1:60000
j_sub <- seq(1, n, by = batch_size)  # a sequence that counts: 1, 11, 21, 31, etc.
scalar <- 1 # gamma learning rate

for(i in j_sub){
  # i comes from j_sub: 1, 11, 21, etc. For i = 1, rows is 1:10, for i=11, rows is 11:20, etc
  rows <- j[i:(i+batch_size - 1)]  
  X <- train$x[rows, ]/255
  Y <- y[rows,]  # use our vectorized labels
  
  # Forward-Feed
  Z_2 <- X %*% W_1
  A_2 <- sigmoid(Z_2)
  Z_3 <- A_2 %*% W_2
  Y_hat <- sigmoid(Z_3)
  
  # calculate our gradients
  delta_3 <- ( -(Y - Y_hat) * sigmoidprime(Z_3 ) )
  djdw2 <- t(A_2) %*% delta_3
  delta_2 <- delta_3 %*% t(W_2) * sigmoidprime(Z_2)
  djdw1 <- t(X) %*% delta_2
  
  # gradient descent to update our weights
  W_1 <- W_1 - scalar * djdw1
  W_2 <- W_2 - scalar * djdw2
}



#####################################################################
#### test performance
## use test data
Xt <- test$x[1:10000,]/255
batch_size <- dim(Xt)[1]

Z_2 <- Xt %*% W_1
A_2 <- sigmoid(Z_2 )
Z_3 <- A_2 %*% W_2
Y_hat <- sigmoid(Z_3 )
guess <- max.col(Y_hat)-1

# results
guess[1]
show_digit(test$x[1,], asp = 1) # should be 7
guess[2]
show_digit(test$x[2,], asp = 1) # should be 2
guess[3]
show_digit(test$x[3,], asp = 1) # should be 1
guess[4]
show_digit(test$x[4,], asp = 1)


results <- data.frame( actual = test$y[1:batch_size], guess = guess)
results_table <- table(results)
results_table
sum(diag(results_table))/dim(Xt)[1]


#####################################################################


# mistakes
which(results[,1] != results[,2])

j = 9010
test$y[j] # the label
show_digit(test$x[j,], asp = 1)
guess[j]

j = 116
test$y[j] # the label
show_digit(test$x[j,], asp = 1)
guess[j]


