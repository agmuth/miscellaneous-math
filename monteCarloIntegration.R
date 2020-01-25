#' Parallel implimentation of Monte Carlo integration.
#'
#' @param func funtion to compute the expected value of should take the output from proposal. 
#' Note this should be the target function divided by the density of the proposal distribution.
#' @param proposal function that returns random draws from some distribution. 
#' Function should have single argument corresponding to the number of draws to be returned.
#' @param cores number of cores to be used in parallel.
#' @param confidence confidence of corresponding confidence interval to be returned.
#' @param width width of confidence interval to be returned.
#' @param seed random seed.
#' @param max_batch_size maximum batch size to send to individual cores.
#'
#' @return estimate of mean and standard error of function of interest.
#' @export
#'
#' @examples
monteCarloIntegration <- function(func, proposal, cores=detectCores()-1, confidence=0.95, width=0.01, seed=NULL, max_batch_size=Inf){
  require(parallel)
  
  if(is.null(seed)){
    seed <- sample(1:9999, 1) #set random seed if not given
  }
  set.seed(seed, kind = "Mersenne-Twister")
  
  #get initial estimate of sd to compute approx sample size needed for desired CI width
  sd_estimate <- sd(func(proposal(10000)))
  z_coef <- qnorm(confidence + (1 - confidence) / 2)
  sample_size <- (2 * z_coef * sd_estimate / width)^2
  sample_size <- ceiling(sample_size)
  batch_size <- min(max_batch_size, ceiling(sample_size / cores))
  batches <- floor(sample_size / batch_size)
  
  cl <- makeCluster(cores)
  clusterExport(cl, c("func", "proposal"))
  clusterSetRNGStream(cl, iseed = seed)#, kind = "L'Ecuyer-CMRG")
  
  dat <- clusterApply(cl, rep(batch_size, batches), fun = function(n){
    x <- func(proposal(n))
    return(c(sum(x), sum(x^2), n)) #return sufficient statistics
  })
  stopCluster(cl)
  
  dat <- matrix(unlist(dat), ncol=3, byrow=TRUE)
  dat <- colSums(dat) #pool sufficient statistics
  
  #return estimate of mean and standard deviation of estimate of mean
  return(c(dat[1] / dat[3], 
           (dat[3]-1)^-1 * (dat[2] - 2*dat[3]^-1*dat[1]^2 + dat[3]^-2*dat[1]^2) / sqrt(dat[3])))
  
}


#TEST--------------------------------------
#Compute integral of sin(x)/x over positive real line. Should evaluate to pi/2.

#proposal <- function(n) rexp(n, 1)
#func <- function(x) (sin(x) / x) * (1*exp(-1*x))^-1 #target function divided by proposal distribution 

#monteCarloIntegration(func, proposal, seed=1234)

#print(paste0("exact value is: ", pi/2))
