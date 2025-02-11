library(Seurat)
library(SPACO)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)
library(sctransform)


# ----to install the data we need
#-----InstallData("stxKidney.SeuratData")
#kidney = data("stxKidney")

#-------------------------Helper functions ------------------------#


source("CustomSCTransform.R")
#-------------------------Helper functions ------------------------#

kidney <- stxKidney.SeuratData::stxKidney
kidney <- PercentageFeatureSet(kidney, pattern = "^mt-" ,col.name = "percent.mt")
kidney <- PercentageFeatureSet(kidney, pattern = "^Hbb-" ,col.name = "percent.hbb")


#------Using SCTransform  -------#
#source("CustomSCTransform.R")
umi <- GetAssayData(object = kidney, assay = "Spatial", slot = "counts")
umi_normalized <- sctransform::vst(
  umi,
  n_genes = 3000,
  return_gene_attr = TRUE,
  return_cell_attr = TRUE
)
#----- for scaling ---- #
sct.data <- umi_normalized$y

sct.data_scaled <- ScaleData(
  object = sct.data,
  features = rownames(x = sct.data),
  vars.to.regress = NULL,
  do.scale = TRUE,
  do.center = TRUE,
  verbose = FALSE
)
kidney[["SCT"]] <- CreateAssayObject(counts = sct.data_scaled)
SpaCoObject <- seurat_to_spaco(kidney, assay = "SCT", n_image= 1, slot = "scale.data")

