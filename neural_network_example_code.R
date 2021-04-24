# Neural Network Example code
# Code copyright Miles Chen. For personal use only. Do not distribute.
# The following R code is based on the Python code produced 
# by Stephen Welch and Welch Labs and the Youtube series:
# https://www.youtube.com/watch?v=bxe2T-V8XRs&list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU
# Follows the Welch Labs Videos
RNGkind(sample.kind = "Rejection")

# training data
x <- matrix(c(
    3, 5,
    5, 1,
    10, 2
), ncol = 2, byrow = TRUE)
y <- matrix(c(75, 82, 93))

# Scale the training data
# divide the x values by the column max, so it is on a scale from 0 to 1
colmax <- apply(x, 2, max)
X <- t(t(x) / colmax)
# y has a range of 100, so this puts it on a scale from 0 to 1 also
Y <- y / 100 # target values are now 0.75, 0.82, 0.93

# Video 2: building the neural network
# we will not be building an R class (but we could)
input_layer_size <- 2
output_layer_size <- 1
hidden_layer_size <- 3

# weight matrix
# one row for each variable in x; one column for each node in hidden layer
# (2 x 3) matrix
W_1 <- matrix(1, nrow = input_layer_size, ncol = hidden_layer_size)

# Z is the product of X and the weights.
# It tells us how to combine the x-variables for use in the 'activation'
# function. We will apply a function to Z_2.
Z_2 <- X %*% W_1

# We define a function called sigmoid. This will apply the sigmoid function
# to a vector or matrix element-wise.
sigmoid <- function(Z) {
    1 / (1 + exp(-Z))
}

# apply the sigmoid function to the matrix Z_2
A_2 <- sigmoid(Z_2)

# create our second weight matrix W_2, a (3x1) matrix
W_2 <- matrix(1, nrow = hidden_layer_size, ncol = output_layer_size)

# multiply the A matrix by the weight matrix
Z_3 <- A_2 %*% W_2

# applying the sigmoid function to the Z_3 matrix produces our Y-hats
Y_hat <- sigmoid(Z_3)

# define our cost function to quantify the error between Y and Y-hat
cost <- function(y, y_hat) {
    0.5 * sum((y - y_hat) ^ 2)
}
cost(Y, Y_hat)


# calculate the gradients (derivatives) with respect to the weights

# dw_2 gradient of the second weight matrix
# dw2 = -(y - y_hat)*[dy_hat/dz_3]*[dz_3/dw_2]

# derivative of the sigmoid function is exp(-z)/((1+exp(-z))^2)
# we define a function to calculate the derivative of the sigmoid function
sigmoidprime <- function(z) {
    exp(-z) / ((1 + exp(-z)) ^ 2)
}

# A plot for testing this function as seen in video
t <- seq(-6, 6, by = .01)
plot(t, sigmoid(t), type = "l", col = "blue")
points(t, sigmoidprime(t), type = "l", col = "green")

## use the function to calculate our gradients
# See the lecture notes for derivations of the gradients via back propagation
delta_3 <- (-(Y - Y_hat) * sigmoidprime(Z_3))
djdw2 <- t(A_2) %*% delta_3

delta_2 <- delta_3 %*% t(W_2) * sigmoidprime(Z_2)
djdw1 <- t(X) %*% delta_2

djdw1 # the resulting gradients
djdw2

# We update the weight matrices by subtracting the gradient
scalar <- 5
W_1 <- W_1 - scalar * djdw1
W_2 <- W_2 - scalar * djdw2

# we compute the cost function of the current Y and Y-hats
cost1 <- cost(Y, Y_hat)

# after calculating the gradient and applying the change to the weight
# matrices, we will recalculate Y-hats.
Z_2 <- X %*% W_1
A_2 <- sigmoid(Z_2)
Z_3 <- A_2 %*% W_2
new_Y_hat <- sigmoid(Z_3)

# Calculate the value of the cost function and compare costs
cost2 <- cost(Y, new_Y_hat)

# Indeed, the cost of the y-hats after subtracting the gradient is smaller,
# which is what we want to see.
cost1
cost2

################################################################
## Numerical Gradient Checking ##
# Before we perform gradient descent, we can numerically check the gradient.
# We perturb the value of the gradient by a small value.
# We check to see how much the function changes. The change in the function
# divided by the change is an approximate slope value. The change should match
# the value of the gradient

# starting values of 1 in the matrices of weights
W_1 <- matrix(1, nrow = input_layer_size, ncol = hidden_layer_size)
W_2 <- matrix(1, nrow = hidden_layer_size, ncol = output_layer_size)

Z_2 <- X %*% W_1
A_2 <- sigmoid(Z_2)
Z_3 <- A_2 %*% W_2
Y_hat <- sigmoid(Z_3) # current estimates based on our weight matrices
currentcost <- cost(Y, Y_hat)  # Current cost

e <- 1e-4  # size of perturbation

# place holder for our numeric gradients
# the gradient wrt to w_1 will be a 2 x 3 matrix.
numgrad_w_1 <- matrix(0, nrow = input_layer_size, ncol = hidden_layer_size)
elements <- input_layer_size * hidden_layer_size # 6

for (i in 1:elements) {
    # calculate the numeric gradient for each value in the W matrix
    W_1 <- matrix(1, nrow = input_layer_size, ncol = hidden_layer_size)
    W_2 <- matrix(1, nrow = hidden_layer_size, ncol = output_layer_size)
    W_1[i] <- W_1[i] + e # apply the perturbation
    Z_2 <- X %*% W_1
    A_2 <- sigmoid(Z_2)
    Z_3 <- A_2 %*% W_2
    Y_hat <- sigmoid(Z_3)
    # change in cost over perturbation = slope
    numgrad_w_1[i] <- (cost(Y, Y_hat) - currentcost) / e
}

# the gradient wrt to w_2 will be a 3 x 1 matrix.
numgrad_w_2 <- matrix(1, nrow = hidden_layer_size, ncol = output_layer_size)
for (i in 1:3) {
    W_1 <- matrix(1, nrow = input_layer_size, ncol = hidden_layer_size)
    W_2 <- matrix(1, nrow = hidden_layer_size, ncol = output_layer_size)
    W_2[i] <- W_2[i] + e
    Z_2 <- X %*% W_1
    A_2 <- sigmoid(Z_2)
    Z_3 <- A_2 %*% W_2
    Y_hat <- sigmoid(Z_3)
    numgrad_w_2[i] <- (cost(Y, Y_hat) - currentcost) / e
}


# values of the gradient using our gradient function
delta_3 <- (-(Y - Y_hat) * sigmoidprime(Z_3))
djdw2 <- t(A_2) %*% delta_3
delta_2 <- delta_3 %*% t(W_2) * sigmoidprime(Z_2)
djdw1 <- t(X) %*% delta_2
djdw1 # analytic gradient wrt to w_1
numgrad_w_1 # compare the analytic gradient to the numeric gradient
djdw2 # analytic gradient wrt to w_2
numgrad_w_2 # compare the analytic gradient to the numeric gradient
# The gradients seem to be very similar indicating that
# our calculation of the analytic gradient seems to be correct.

# check by dividing the norm of the difference by the norm of the sum
# compare difference between analytic and numeric gradients
norm(djdw1 - numgrad_w_1) /  norm(djdw1 + numgrad_w_1)
norm(djdw2 - numgrad_w_2) /  norm(djdw2 + numgrad_w_2)

## End numeric gradient check
#####################################################################

# application of gradient descent without BFGS
# This is just iteratively taking steps in the
# direction of the negative gradient

#initialize our weight matrices again with random weights
set.seed(1)
W_1 <- matrix(runif(6), nrow = input_layer_size, ncol = hidden_layer_size)
W_2 <- matrix(runif(3), nrow = hidden_layer_size, ncol = output_layer_size)

# for cost tracking
cost_hist <- rep(NA, 10000)

scalar <- 5
for (i in 1:10000) {
    # this takes the current weights and calculates y-hat
    Z_2 <- X %*% W_1
    A_2 <- sigmoid(Z_2)
    Z_3 <- A_2 %*% W_2
    Y_hat <- sigmoid(Z_3)
    cost_hist[i] <- cost(Y, Y_hat)
    # this part calculates the gradient at the current y-hat
    delta_3 <- (-(Y - Y_hat) * sigmoidprime(Z_3))
    djdw2 <- t(A_2) %*% delta_3
    delta_2 <- delta_3 %*% t(W_2) * sigmoidprime(Z_2)
    djdw1 <- t(X) %*% delta_2
    # this updates the weights based on the gradient
    W_1 <- W_1 - scalar * djdw1
    W_2 <- W_2 - scalar * djdw2
    # repeat
}

# the results
W_1
W_2
Y_hat
Y
cost(Y, Y_hat)
plot(cost_hist, type = "l") # plot the history of our cost function
plot(log(cost_hist), type = "l") # plotting the log of the cost emphasizes the change

# Recommended method: use optim(). It's better than our gradient algorithm.
# You still need to define the function and gradient.
# We have already defined the cost function, which we wish to optimize.
# It is the squared error.
# optim() requires that we express it in terms of the parameters
# we wish to adjust.
# In this case, we have 9 weight parameters to adjust.
# Earlier, we split these into two groups: 6 in W_1 and 3 in W_2
# For this function w[1:6] will be the weights in W_1
# and w[7:9] will be the weights in W_2

cost_optim <- function(w){
    # cost depends on our data and the weights.
    # our data is fixed
    y <- matrix(c(75, 82, 93))
    x <- matrix(c(
         3, 5,
         5, 1, 
        10, 2
    ), ncol = 2, byrow = TRUE)
    colmax <- apply(x, 2, max)
    X <- t(t(x) / colmax)
    Y <- y / 100
    
    # now we calculate the cost based on the provided weights
    W_1 <- matrix(w[1:6], nrow = 2, ncol = 3)
    W_2 <- matrix(w[7:9], nrow = 3, ncol = 1)
    Z_2 <- X %*% W_1
    A_2 <- sigmoid(Z_2)
    Z_3 <- A_2 %*% W_2
    Y_hat <- sigmoid(Z_3)
    0.5 * sum((Y - Y_hat) ^ 2)
}


set.seed(1)
res <- optim(par = runif(9), 
             fn = cost_optim, control = list(maxit = 1000))
print(res) # our squared error is very small ~4e-11
W_1 <- matrix(res$par[1:6], nrow = 2, ncol = 3)
W_2 <- matrix(res$par[7:9], nrow = 3, ncol = 1)
Z_2 <- X %*% W_1
A_2 <- sigmoid(Z_2)
Z_3 <- A_2 %*% W_2
Y_hat <- sigmoid(Z_3)
print(Y_hat)

# different starting locations can lead to different results
set.seed(10)
res <- optim(par = runif(9), fn = cost_optim, control = list(maxit = 1000))
print(res) # squared error is large ~0.0012

# using BFGS
set.seed(2)
res <- optim(par = runif(9), fn = cost_optim, method = "BFGS",
             control = list(maxit = 1000))
print(res) # squared error is very small 8.8e-15

# using BFGS with different starting seed
set.seed(3)
res <- optim(par = runif(9), fn = cost_optim, method = "BFGS",
             control = list(maxit = 1000))
print(res) # squared error is large ~0.0012

# trying different random seeds and looking at the resulting squared error
# we see many different results
for(i in 1:100){
    set.seed(i)
    res <- optim(par = runif(9), fn = cost_optim, method = "BFGS",
                 control = list(maxit = 1000))
    print(c(round(i), res$value))
}

