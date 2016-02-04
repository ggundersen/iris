# Logistic regression classifier to predict setosa and versicolor irises using
# two features, petal width and petal length.
#
# Gregory Gundersen
# 3 Feb 2016
# -----------------------------------------------------------------------------

sigmoid <- function(z) {
  return (1 / (1 + exp(-z)))
}

gradient_descent <- function(X, y, theta, alpha, num_iters) {
  m = length(y)
  for(iter in 1:num_iters) {
    hyp = sigmoid(X %*% theta)
    J = sum((-y * log(hyp)) - ((1-y) * log(1 - hyp))) / m
    grad = (t(X) %*% (hyp - y)) / m
    theta = theta - alpha * grad
  }
  return (theta)
}

predict <- function(theta, X) {
  p = sigmoid(X %*% theta) >= 0.5;
  return (p)
}

# ------------------------------- END FUNCTIONS -------------------------------

# This plot makes it pretty clear that we should be able to classify the irises
# by species based on their petal lengths and widths. To simplify the problem
# for myself, I'm going to remove the virginica species, which overlaps with
# the setosa species.
plot(iris$Petal.Width, iris$Petal.Length, col=iris$Species)

# Remove virginica species to avoid multi-class problem.
remove_idx <- which(iris$Species != "virginica")
iris <- iris[remove_idx,]

# Verify that this classification problem is straightforward.
plot(iris$Petal.Width, iris$Petal.Length, col=iris$Species)

# Randomize dataset before splitting it into training and testing sets.
rand_idx <- sample(nrow(iris))
iris <- iris[rand_idx,]

X <- data.matrix(iris[,3:4])
X_train <- X[1:80,]
X_test <- X[81:100,]
# Create binary matrix where 1 indicates the species is setosa, 0 indicates not
# setosa, i.e. is versicolor
y <- matrix((iris$Species == "setosa") * 1)
y_train <- y[1:80]
y_test <- y[81:100]
correct_labels <- iris[,5][81:100]

# Since we have two features, we are learning two parameters.
initial_theta <- matrix(c(0, 0))

# Learn theta. I'm not sure if alpha=0.5 and num_iters=400 are good defaults,
# but they seem to work on this dataset.
theta <- gradient_descent(X_train, y_train, initial_theta, 0.5, 400)
prob = predict(theta, X_test)

pct_correct <- sum(y_test == prob) / length(prob)
print(pct_correct)
