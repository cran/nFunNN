#' @title Curve reconstruction
#'
#' @description Curve reconstruction by the trained transformed functional autoassociative neural network.
#'
#' @param model The trained transformed functional autoassociative neural network obtained from \code{\link[nFunNN]{nFunNNmodel}}.
#' @param X_ob A \code{matrix} denoting the observed data from subjects that we aim to predict.
#' @param L An \code{integer} denoting the number of B-spline basis functions for the parameters in the network.
#' @param t_grid A \code{vector} denoting the observation time grids on \code{[0, 1]}.
#'
#' @return A torch tensor denoting the predicted values.
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
#' L, J, K, R, lr = 0.001, n_epoch = 1500, batch_size = 100)
#' model <- nFunNN_res$model
#' X_pre <- nFunNN_CR(model, X_ob, L, t_grid)
#' sqrt(torch::nnf_mse_loss(X_pre, torch::torch_tensor(X_to_est))$item())}
nFunNN_CR <- function(model, X_ob, L, t_grid){

  basis <- fda::create.bspline.basis(c(0, 1), nbasis = L, norder = 4,
                                     breaks = seq(0, 1, length.out = L - 2))
  Xfd <- fda::smooth.basis(t_grid, t(X_ob), basis)$fd
  X_input <- fda::inprod(Xfd, basis)

  X_pre <- model(torch::torch_tensor(X_input))

  return(X_pre)

}
