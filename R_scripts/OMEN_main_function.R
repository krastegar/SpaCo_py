#' OMEN - Primary function and SpaC-test 
#'
#' @param mod1 first spots x feature data matrix; f_1 features as columns, n_1 loci as rows
#' @param mod1 second spots x feature data matrix; f_2 features as columns, n_2 loci as rows
#' @param coords1 coordinates of loci in mod1
#' @param coords2 coordinates of loci in mod2
#' @param A_11 n_1 x n_1 matrix showing neighborhood weight of loci in mod1
#' @param A_22 n_2 x n_2 matrix showing neighborhood weight of loci in mod2
#' @param A_12 n_1 x n_2 matrix showing neighborhood weight of loci between mod1 and mod2
#' @param compute_n_multi_Spacs Boolean if number of relevant spacs is to be computed. Increases run time significantly
#' @param CCA_regulizer Boolean if regulized indivdual spacs should be considered
#' @param reg_para weight parameter associated with CCA_regulization. 
#' @param PC_value Value to specify number of PCs or desired level of explained variance, see "PC_criterion"
#' 
#' @return
#' Returns a multi-modal SpaCoObject filled with the results of OMEN.
#' @export
#'
#' @import methods
#' @import rARPACK
#' @import Rcpp
#' @import RcppEigen


Run_SCT_multimodal <- function(mod1,
                               mod2, 
                               coords1,
                               coords2,
                               A_11 = NULL,
                               A_22 = NULL,
                               A_12,
                               CCA_regulizer = FALSE,
                               reg_para = 1,
                               compute_n_multi_spacs = TRUE, 
                               PC_value = 0.95){
  
  # Check required libraries: 
  
  requiredLibraries <- c("Rcpp", "RcppEigen", "rARPACK")
  lapply(requiredLibraries, require, character.only = TRUE)
  
  
  # Function to compute Graph Laplacian - I left this because it solves the issue with testing for 
  # sparse against non-sparse matrices - future versions of this function should check this at the beginning. 
  computeGraphLaplacian <- function(neighbourIndexMatrix) {
    if(inherits(neighbourIndexMatrix,what = "dgCMatrix"))
    {
      n <- neighbourIndexMatrix@Dim[1]
      graphLaplacian <- neighbourIndexMatrix + Matrix::Diagonal(n, 1 / n)
    }else
    {
      n <- nrow(neighbourIndexMatrix)
      
      graphLaplacian <- neighbourIndexMatrix + diag(1 / n, n)
    }
  }
  
  # Function for PCA whitening: 
  PCA_whitening <- function(data, value) {
    
    # Establish PCA conditions: 
    dataCentered <- scale(data, scale = TRUE, center = TRUE)
    n <- nrow(data)
    # Derive covariance matrix: 
    varMatrix <- (1 / (n - 1)) * eigenMapMatMult(t(dataCentered), dataCentered)
    
    # Eigendecomposition: 
    initialPCA <- eigen(varMatrix, symmetric = TRUE)
    
    # Dim. reduction by selection threshold: 
    nEigenVals <- min(which(cumsum(initialPCA$values) / sum(initialPCA$values) > value))
    
    # Whitening transformation
    D <- diag(1 / sqrt(initialPCA$values[1:nEigenVals]))
    whitenedData <- t(eigenMapMatMult(D, eigenMapMatMult(t(initialPCA$vectors[, 1:nEigenVals]), t(dataCentered))))
    
    # Return output
    return(list(dataReduced = whitenedData, nEigenVals = nEigenVals, initialPCA = initialPCA))
    
  }
  
  ######## OMEN formating #######
  # Define datastructures for the joint dataset: 
  joint_coords <- rbind(coords1, coords2)
  joint_coords <- joint_coords[c(rownames(mod1), rownames(mod2)),] # reorder 
  
  # Non-whitend Z matrix: 
  dim_mod_joint <- dim(mod1) + dim(mod2)
  data_joint <- matrix(0, nrow = dim_mod_joint[1], ncol = dim_mod_joint[2])
  data_joint[1:nrow(mod1),1:ncol(mod1)] <- as.matrix(mod1)
  data_joint[(nrow(mod1)+1):nrow(data_joint),(ncol(mod1)+1):ncol(data_joint)] <- as.matrix(mod2)
  
  # Assign rownames to data: 
  rownames(data_joint) <- rownames(joint_coords)
  colnames(data_joint) <- c(colnames(mod1), colnames(mod2))
  
  # PCA- Whitening (Also performes Z-scaling): 
  white_mod1 <- PCA_whitening(data = mod1, value = PC_value)
  white_mod2 <- PCA_whitening(data = mod2, value = PC_value)
  
  
  # Construct Whitend Z-matrix compressed structure: 
  X <- white_mod1$dataReduced 
  Y <- white_mod2$dataReduced
  
  # Generate Z matrix: 
  dim_mod_joint <- dim(X) + dim(Y)
  
  # Generate matrix of proper dimensions: 
  Z <- matrix(0, nrow = dim_mod_joint[1], ncol = dim_mod_joint[2])
  
  # Fill Z-matrix: 
  Z[1:nrow(X),1:ncol(X)] <- as.matrix(X)
  Z[(nrow(X)+1):nrow(Z),(ncol(X)+1):ncol(Z)] <- as.matrix(Y)
  
  # Reassign rownames: 
  rownames(Z) <- rownames(joint_coords)
  colnames(Z) <- c(paste0("PCA_Metafeature", 1:ncol(Z)))
  
  
  # Derive Neighboring matrix (A) between modalities:  
  
  L <- Matrix(0, nrow = nrow(joint_coords), ncol = nrow(joint_coords), sparse = TRUE)
  
  # Include Shared neighbormatrix: 
  L[1:nrow(coords1),(nrow(coords1)+1):ncol(L)] <- A_12 * (1/sum(A_12))
  L[(nrow(coords1)+1):nrow(L),1:nrow(coords1)] <- t(A_12) * (1/sum(A_12))
  
  # Include regulizer: 
  if (CCA_regulizer == TRUE){
    
    L[1:nrow(coords1),1:nrow(coords1)] <-   A_11 * (1/sum(A_11))
    L[(nrow(coords1)+1):nrow(L),(nrow(coords1)+1):ncol(L)] <-  A_22 * (1/sum(A_22))
    
  }
  
  # Renaming: 
  rownames(L) <- rownames(joint_coords)
  colnames(L) <- rownames(joint_coords)
  
  
  # Derive Graph laplacian. Note that one can input various version here for L  
  tmpTrainGL <- computeGraphLaplacian(L)
  
  # Introduce regulizer:
  # This was done for checking whether it alliates the problem of overestimating 
  # One modality - and it sort of does but I only included it in the discussion. 
  
  if (CCA_regulizer & reg_para != 1){
    
    # Construct regulizer matrix: 
    dim_11 <- dim(A_11)
    dim_22 <- dim(A_22)
    mat_11 <- matrix(reg_para, nrow = dim_11[1], ncol = dim_11[2])
    mat_22 <- matrix(reg_para, nrow = dim_22[1], ncol = dim_22[2])
    diag(mat_11) <- 1 # Dont perturb the diagonal! 
    diag(mat_22) <- 1 
    
    # Assign in matrix: 
    
    tmpTrainGL[1:nrow(coords1),1:nrow(coords1)] <- tmpTrainGL[1:nrow(coords1),1:nrow(coords1)] * mat_11
    tmpTrainGL[(nrow(coords1)+1):nrow(L),(nrow(coords1)+1):ncol(L)] <- tmpTrainGL[(nrow(coords1)+1):nrow(L),(nrow(coords1)+1):ncol(L)] * mat_22
    
  }
  
  
  
  # Compute test statistic matrix
  if(inherits(x = tmpTrainGL, what = "dgCMatrix")){
    Rx <- t(Z) %*% tmpTrainGL %*% Z
  }else{
    Rx <- eigenMapMatMult(t(Z), eigenMapMatMult(tmpTrainGL, Z))
  }
  
  # Eigendecomposition of Rx
  eigenRx <- eigen(Rx)
  PCsRx <- eigenRx$vectors  
  lambdas <- eigenRx$values 
  
  # Compute ONB and spacs (from whitened data!)
  Spac_matrix <- Z %*% PCsRx
  rownames(Spac_matrix) <- rownames(joint_coords)
  colnames(Spac_matrix) <- paste0("spac_", 1:ncol(Spac_matrix))
  
  # Create new SPACO object that holds all the data:
  SpaCoobject_multimodal = SpaCoObject( L,  # spots x spots
                                        data_joint,  # spots x features
                                        as.data.frame(joint_coords)) # x,y positions
  
  SpaCoobject_multimodal@spacs <- Spac_matrix
  SpaCoobject_multimodal@Lambdas <- lambdas
  SpaCoobject_multimodal@GraphLaplacian <- as(tmpTrainGL, "dgCMatrix")
  
  # Spacs from non-whitened data: 
  # Construct matrices to project original data on lower dim space and rotate on eigenvectors:  
  rot_X <- white_mod1$initialPCA$vectors[,1:white_mod1$nEigenVals]
  rot_Y <- white_mod2$initialPCA$vectors[,1:white_mod2$nEigenVals]
  
  tmp1 <- scale(mod1 %*% rot_X)
  tmp2 <- scale(mod2 %*% rot_Y)
  
  # Generate matrix of proper dimensions: 
  dim_tmp <- dim(tmp1) + dim(tmp2)
  tmp <- matrix(0, nrow = dim_tmp[1], ncol = dim_tmp[2])
  
  # Fill tmp-matrix: 
  tmp[1:nrow(tmp1),1:ncol(tmp1)] <- as.matrix(tmp1)
  tmp[(nrow(tmp1)+1):nrow(tmp),(ncol(tmp1)+1):ncol(tmp)] <- as.matrix(tmp2)
  
  # Generate spacs from tmp-matrix: 
  projection <- eigenMapMatMult(tmp, PCsRx)
  rownames(projection) <- rownames(data_joint)
  colnames(projection) <- paste0("spac_", 1:ncol(projection))
  SpaCoobject_multimodal@projection <- projection
  
  
  # Compute number of relevant OMEN-SpaCs if required
  
  computeRelevantSpacs <- function(X, Y, OpSim = 20, graphLaplacian, lambdas) {
    simSpacFunction <- function(i) {
      shuffleOrder1 <- sample(nrow(X), nrow(X))
      shuffleOrder2 <- sample(nrow(Y), nrow(Y))
      
      print("Things are happening (still!)")
      
      # Generate matrix of proper dimensions: 
      dim_mod_joint <- dim(X) + dim(Y)
      Z <- matrix(0, nrow = dim_mod_joint[1], ncol = dim_mod_joint[2])
      
      # Fill Z-matrix + permutuation: 
      Z[1:nrow(X),1:ncol(X)] <- as.matrix(X[shuffleOrder1,])
      Z[(nrow(X)+1):nrow(Z),(ncol(X)+1):ncol(Z)] <- as.matrix(Y[shuffleOrder2,])
      
      # Converting to sparse matrices is way faster
      Z <- Matrix(Z, sparse = TRUE)
      graphLaplacian <- Matrix(graphLaplacian, sparse = TRUE)
      
      if(inherits(x = L,what = "dgCMatrix")){
        RxShuffled <- t(Z) %*% graphLaplacian %*% Z
      }else{
        RxShuffled <- eigenMapMatMult(t(Z),eigenMapMatMult(graphLaplacian, Z))
      }
      
      eigs_sym(RxShuffled, 1, which = "LM")$values
    }
    
    graphLaplacian <- tmpTrainGL
    resultsAll <- replicate(20, simSpacFunction())
    
    # Derive confidence interval: 
    eigValSE <- sd(resultsAll) / sqrt(length(resultsAll))
    eigValCI <- mean(resultsAll) + qt(0.975, df = length(resultsAll) - 1) * eigValSE * c(-1, 1)
    lambdasInCI <- lambdas[lambdas > eigValCI[1] & lambdas < eigValCI[2]]
    
    # Derive maximum number of SpaCs 
    
    Op_sim = 1
    
    if(length(lambdasInCI) > 1)
    {
      for(i in 1:Op_sim)
      {
        print("Computing relevant SpaCs")
        batchResult <- replicate(20, simSpacFunction())
        resultsAll <- c(resultsAll, batchResult)
        eigValSE <- sd(resultsAll) / sqrt(length(resultsAll))
        eigValCI <- mean(resultsAll) + c(-1,1) *
          qt(0.975, df = length(resultsAll) - 1) * eigValSE
        lambdasInCI <- lambdas[which(lambdas > eigValCI[1] &
                                       lambdas < eigValCI[2])]
        if(length(lambdasInCI) < 2)
        {
          break
        }
      }
    }
    relSpacsIdx <- which(lambdas < max(eigValCI))
    nSpacs <- if (any(relSpacsIdx)) min(relSpacsIdx) else pcaResults$nEigenVals
    nSpacs
  }
  
  
  if (compute_n_multi_spacs) {
    nSpacs <- computeRelevantSpacs(X, Y, OpSim = 1, tmpTrainGL, lambdas)
    SpaCoobject_multimodal@nSpacs <- nSpacs
  }
  
  return(SpaCoobject_multimodal)
}


