#' @title Nonlinear FPCA using neural networks
#'
#' @description Nonlinear functional principal component analysis using a transformed functional autoassociative neural network.
#'
#' @param X_ob A \code{matrix} denoting the observed data.
#' @param t_grid A \code{vector} denoting the observation time grids on \code{[0, 1]}.
#' @param t_grid_est A \code{vector} denoting the time grids that have to be predicted on \code{[0, 1]}.
#' @param L_smooth An \code{integer} denoting the number of B-spline basis functions that used to smooth the observed data for the computation of the loss function.
#' @param L An \code{integer} denoting the number of B-spline basis functions for the parameters in the network.
#' @param J An \code{integer} denoting the number of neurons in the first hidden layer.
#' @param K An \code{integer} denoting the number of principal components.
#' @param R An \code{integer} denoting the number of neurons in the third hidden layer.
#' @param lr A scalar denoting the learning rate. (default: 0.001)
#' @param batch_size An \code{integer} denoting the batch size.
#' @param n_epoch An \code{integer} denoting the number of epochs.
#'
#' @return A \code{list} containing the following components:
#' \item{model}{The resulting neural network trained by the observed data.}
#' \item{loss}{A \code{vector} denoting the averaged loss in each epoch.}
#' \item{Comp_time}{An object of class "difftime" denoting the computation time in seconds.}
#' @export
#'
#' @examples
#' \donttest{
#' n <- 2000
#' m <- 51
#' t_grid <- seq(0, 1, length.out = m)
#' m_est <- 101
#' t_grid_est <- seq(0, 1, length.out = m_est)
#' err_sd <- 0.1
#' Z_1a <- stats::rnorm(n, 0, 3)
#' Z_2a <- stats::rnorm(n, 0, 2)
#' Z_a <- cbind(Z_1a, Z_2a)
#' Phi <- cbind(sin(2 * pi * t_grid), cos(2 * pi * t_grid))
#' Phi_est <- cbind(sin(2 * pi * t_grid_est), cos(2 * pi * t_grid_est))
#' X <- Z_a %*% t(Phi)
#' X_to_est <- Z_a %*% t(Phi_est)
#' X_ob <- X + matrix(stats::rnorm(n * m, 0, err_sd), nr = n, nc = m)
#' L_smooth <- 10
#' L <- 10
#' J <- 20
#' K <- 2
#' R <- 20
#' nFunNN_res <- nFunNNmodel(X_ob, t_grid, t_grid_est, L_smooth,
#' L, J, K, R, lr = 0.001, n_epoch = 1500, batch_size = 100)}
nFunNNmodel <- function(X_ob, t_grid, t_grid_est, L_smooth, L, J, K, R, lr = 0.001, batch_size, n_epoch){
  n <- dim(X_ob)[1]
  basis <- fda::create.bspline.basis(c(0, 1), nbasis = L, norder = 4,
                                     breaks = seq(0, 1, length.out = L - 2))
  Xfd <- fda::smooth.basis(t_grid, t(X_ob), basis)$fd
  X_input <- fda::inprod(Xfd, basis)
  basis_smooth <- fda::create.bspline.basis(c(0, 1), nbasis = L_smooth, norder = 4,
                                            breaks = seq(0, 1, length.out = L_smooth - 2))
  Xfd_smooth <- fda::smooth.basis(t_grid, t(X_ob), basis_smooth)$fd
  response <- t(fda::eval.fd(t_grid_est, Xfd_smooth))

  BS_eval <- splines::bs(x = t_grid_est, degree = 3,
                         knots = seq(0, 1, length.out = L - 2)[-c(1, L - 2)], intercept = T)
  BS_eval <- torch::torch_tensor(BS_eval)$t()

  self <- NULL
  nFunNN_net <- torch::nn_module(
    initialize = function(input_size, hidden_size1, bottleneck_size,
                          hidden_size2, basis_eval){

      self$fc1 <- torch::nn_linear(in_features = input_size, out_features = hidden_size1,
                                   bias = TRUE)
      self$ReLU <- torch::nn_relu()
      self$fc2 <- torch::nn_linear(in_features = hidden_size1, out_features = bottleneck_size,
                                   bias = FALSE)
      self$fc3 <- torch::nn_linear(in_features = bottleneck_size,
                                   out_features = input_size * hidden_size2,
                                   bias = TRUE)
      self$basis_eval <- basis_eval
      self$input_size <- input_size
      self$hidden_size2 <- hidden_size2

      self$w <- torch::nn_parameter(torch::torch_randn(hidden_size2, 1))

    },
    forward = function(input){
      out1 <- self$ReLU(self$fc1(input))
      out2 <- self$fc2(out1)
      out3 <- self$fc3(out2)

      out4 <- torch::torch_matmul(out3$reshape(c(out3$shape[1], self$hidden_size2, self$input_size)),
                                  self$basis_eval)
      out4_act <- self$ReLU(out4)

      (out4_act * self$w)$sum(dim = 2)
    }
  )

  model <- nFunNN_net(input_size = L, hidden_size1 = J, bottleneck_size = K,
                      hidden_size2 = R, basis_eval = BS_eval)
  opt <- torch::optim_adam(model$parameters, lr = lr)

  losses <- list()
  ep <- 1
  start_time <- Sys.time()
  while(ep <= n_epoch){
    batch_loss <- list()
    i <- 1
    for(start in seq(1, n, by = batch_size)){
      ### -------- Forward pass --------
      end <- min(start + batch_size - 1, n)
      x_tilde <- torch::torch_tensor(X_input[start:end,])
      x_pre <- model(x_tilde)

      ### -------- Compute loss --------
      x <- torch::torch_tensor(response[start:end,])
      loss <- torch::nnf_mse_loss(x_pre, x)

      ### -------- Backpropagation --------
      opt$zero_grad()
      loss$backward()

      ### -------- Update weights --------
      opt$step()

      batch_loss[[i]] <- loss$item()
      i <- i + 1
    }
    losses[[ep]] <- mean(unlist(batch_loss))
    # if(ep%%100 == 1){
    #   print(paste('epoch', ep, sep = ' '))
    # }
    ep <- ep + 1
  }
  end_time <- Sys.time()

  res <- list()
  res$model <- model
  res$loss <- unlist(losses)
  res$Comp_time <- difftime(end_time, start_time, units = 'secs')

  return(res)

}
