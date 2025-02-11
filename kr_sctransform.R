# Load required libraries
library(Seurat)
library(sctransform)

#' Perform SCTransform normalization with optional scaling
#'
#' @param object Seurat object containing spatial transcriptomics data
#' @param vst.flavor Flavor of SCTransform normalization 
#' @param assay Name of the assay to use (default: "Spatial")
#' @param slot Slot containing raw counts (default: "counts")
#' @param n_genes Number of genes to use for SCTransform (default: 3000)
#' @param scale Logical, whether to scale the data (default: FALSE)
#' @param vars.to.regress Variables to regress out during scaling (default: NULL)
#' @return Seurat object with SCT-transformed assay
#' @export
normalize_and_scale <- function(object, vst.flavor ,assay = "Spatial", slot = "counts", 
                                n_genes = 3000, scale = FALSE,
                                vars.to.regress = NULL) {
  
  # Extract raw UMI counts
  umi <- GetAssayData(object = object, assay = assay, slot = slot)
  
  # Perform SCTransform normalization
  umi_normalized <- sctransform::vst(
    umi,
    n_genes = n_genes,
    return_gene_attr = TRUE,
    return_cell_attr = TRUE,
    vst.flavor = vst.flavor
  )
  
  # Extract transformed data
  sct.data <- umi_normalized$y
  
  # Scale the data if requested
  if (scale) {
    sct.data <- ScaleData(
      object = sct.data,
      features = rownames(x = sct.data),
      vars.to.regress = vars.to.regress,
      do.scale = TRUE,
      do.center = TRUE,
      verbose = FALSE
    )
  }
  
  # Create new assay and add it to Seurat object
  object[["SCT"]] <- CreateAssayObject(counts = sct.data)
  
  return(object)
}