# This script aims to contain a workflow of a simple manual SPACO application 
# , specifically for data composed only of SF and its respective coords.
# Additionally, some rudimentary pre-processing steps are included. 

# Required data: 

# Spots x Feature matrix 
# Coordinates of corresponding sample 


# Packages: 

library(renv)
library(BiocManager)
library(devtools)
library(Seurat) 
library(SPACO)
library(ggplot2)
library(patchwork)
library(dplyr)
library(glmGamPoi)
library(dbscan)
library(Matrix)
library(reticulate)
library(hdf5r)
library(sceasy)


# Functions for this script: 
source("C:/Users/justu/Desktop/Bachlor Thesis/Programming part/Scripts/Functions/Essential_functions_Multimodal_integration.R")


################################################################################
############## Assign generalized names for the modality: ######################

# mod1_list <- MF_mod_list_processed
brain <- PercentageFeatureSet(brain, pattern = "^mt-" ,col.name = "percent.mt")
brain <- PercentageFeatureSet(brain, pattern = "^Hbb-" ,col.name = "percent.hbb")
brain <- SCTransform(brain, assay = "Spatial", variable.features.n = 3000)
spaco_object <- seurat_to_spaco(Seurat = brain, assay = "SCT", n_image= 1, slot = "scale.data")


################################################################################
####################### Data assignment for non brain data #####################

coords <- coords 
SF <- LD_aggrevated_filterSeurat

################################################################################
############ Generate Neigboring matrix from coordinates #######################

# Various neighbormatrices can be derived at this step: 

dist_matrix <- as.matrix(dist(coords))

neigbormatrix <- KNN_symmetric(dist_matrix = dist_matrix, k = 6, symmetric = TRUE)
neigbormatrix <- normalized_kernel_neighbormatrix(dist_matrix = dist_matrix)


################################################################################
############## 2. Generate SPACO Object and perform SPACO ######################


spaco_object = SpaCoObject(neigbormatrix,  # spots x spots
                         SF,  # spots x features
                         as.data.frame(coords)) # spots x 2 (x,y spatial dimensions)
  

spaco_object = RunSCA(spaco_object, compute_nSpacs = TRUE)


# Extract relevant SPACs
spacs <- spaco_object@projection[,1:spaco_object@nSpacs]
data <- spaco_object@data
coords <- spaco_object@coordinates
coords[1:3,]

# Visualisation of SPAC patterns: 

seurat_object_mod1 <- CreateSeuratObject(counts = t(data), assay = "SF_raw")

# Add metadata
seurat_object_mod1@images$image =  new(
  Class = 'SlideSeq',
  assay = "Mod1",
  key = "image_",
  coordinates = as.data.frame(coords)
)

seurat_object_mod1[["Spacs"]] <- CreateAssayObject(counts = t(spacs)) 

seurat_object_mod1@assays$

DefaultAssay(object = seurat_object_mod1) <- "Spacs" 
p1 <- SpatialFeaturePlot(seurat_object_mod1, features = "spac-1", pt.size.factor = 3.5) 
p2 <- SpatialFeaturePlot(seurat_object_mod1, features = "spac-2", pt.size.factor = 3.5) 
p3 <- SpatialFeaturePlot(seurat_object_mod1, features = "spac-3", pt.size.factor = 3.5) 

p1 + p2 + p3 

################################################################################
############## 3. Projection of individual patterns  ###########################

# Find SVFs: 

DE_genes<- SVGTest(spaco_object)
DE_genes_sort <- DE_genes[order(DE_genes$score, decreasing = TRUE),]
DE_genes_sort <- DE_genes_sort[DE_genes_sort$p.adjust<0.05,]

# Perform projection: 

neigbormatrix <- spaco_object@neighbours
Graph_Laplacian <- computeGraphLaplacian(neigbormatrix)
data <- spaco_object@data

Z <- Perform_projection(Matrix_to_project = data, spac_matrix = spacs, graphLaplacian = Graph_Laplacian)

seurat_object_mod1[["Projected_Data"]] <- CreateAssayObject(counts = t(Z)) 


############################## Visualisation ###################################

SVFs <- rownames(DE_genes_sort)

DefaultAssay(object = seurat_object_mod1) <- "SF_raw" 
p1a <- SpatialFeaturePlot(seurat_object_mod1, features = SVFs[1], pt.size.factor = 3) 
p1b <- SpatialFeaturePlot(seurat_object_mod1, features = SVFs[2], pt.size.factor = 3) 
p1c <- SpatialFeaturePlot(seurat_object_mod1, features = SVFs[3], pt.size.factor = 3) 

DefaultAssay(object = seurat_object_mod1) <- "Projected_Data" 
p2a <- SpatialFeaturePlot(seurat_object_mod1, features = SVFs[1], pt.size.factor = 3) 
p2b <- SpatialFeaturePlot(seurat_object_mod1, features = SVFs[2], pt.size.factor = 3) 
p2c <- SpatialFeaturePlot(seurat_object_mod1, features = SVFs[3], pt.size.factor = 3) 

p1a + p2a 
p1b + p1b
p1c + p2c 


################################################################################
################### Additional analysis (e.g. clustering etc.) #################

brain <- subset_non_neighbour_cells(spaco_object, brain)
brain <- spacs_to_seurat(spaco_object, brain)

cc <- SpatialFeaturePlot(brain,features = c("Spac_1","Spac_2","Spac_3","Spac_4"),combine = T)
brain <- RunUMAP(brain,reduction = "spaco",dims = 1:spaco_object@nSpacs,n.neighbors = 45, verbose = F)
brain <- FindNeighbors(brain,reduction = "spaco" , dims = 1:spaco_object@nSpacs, verbose = F)
brain <- FindClusters(brain,resolution = 0.24, verbose = F)

# Plotting: 

aa <- DimPlot(brain,group.by="seurat_clusters")+ggtitle("Unaltered - Ground truth")
a <- DimPlot(brain,reduction = "spaco")+ggtitle("Unaltered - Ground truth")
rr <- SpatialDimPlot(brain)+ggtitle("Unaltered - Ground truth")

aa
a
rr
cc

### PCA Comparison: ###

brain_2 <- RunPCA(brain, verbose = F)
dd <- SpatialFeaturePlot(brain_2,features = c("PC_1","PC_2","PC_3","PC_4"),combine = T)
brain_2 <- RunUMAP(brain_2,reduction = "pca",dims = 1:30, verbose = F)
brain_2 <- FindNeighbors(brain_2,reduction = "pca" , dims = 1:30, verbose = F)
brain_2 <- FindClusters(brain_2, verbose = F)

# Plotting 

bb <- DimPlot(brain_2,group.by="seurat_clusters")+ggtitle("Pca")
b <- DimPlot(brain_2,reduction = "pca")+ggtitle("Pca")
qq <- SpatialDimPlot(brain_2)+ggtitle("Pca")



################################################################################
########################### Storing the results ################################

# Removing all additional datastructres that were not used #

MF_seurat_object_results <- seurat_object_mod1
MF_SVFs_sorted <- DE_genes_sort
MF_spaco_object <- spaco_object


all_objects <- ls()
all_objects <- ls()

# Remove all objects except 'mod_list'
rm(list = setdiff(all_objects, c("MF_seurat_object_results","MF_SVFs_sorted","MF_spaco_object")))

