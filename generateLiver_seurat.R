# =============================================================================
# R script: create_seurat_spatial_noCLI_visiumV2.R
#
# This script reads a gene-count CSV and associated Visium spatial files,
# builds a Seurat object, constructs a VisiumV2 image slot manually (even if
# Read10X_Image() returns a VisiumV2), attaches it, and computes the spatial
# neighbor graph. After running this script in R, you'll have a Seurat object
# named `seurat_obj` with all slots populated similarly to Load10X_Spatial().
#
# =============================================================================

# -------------------------------
# 0. USER‐DEFINED PATHS
# -------------------------------
# Edit these three lines to point to your files/directories:
counts_csv  <- "/home/rstudio/data_dir/R_scripts/SPACO_paper_data/Spatial_metaData/young1_raw_counts_spatial.csv"
spatial_dir <- "/home/rstudio/data_dir/R_scripts/SPACO_paper_data/Spatial_metaData/spatial_1/"
output_rds  <- "/home/rstudio/data_dir/R_scripts/SPACO_paper_data/youngLiver1_seurat_spatial.rds"

# -------------------------------
# 1. Load required libraries
# -------------------------------
library(Seurat)
library(ggplot2)
library(patchwork)
library(dplyr)
library(png)
library(jsonlite)
devtools::load_all("/home/rstudio/data_dir/R_scripts/SpaCo_R_kia")

# -------------------------------
# 2. Load gene‐by‐spot counts
# -------------------------------
message("Loading counts matrix from CSV: ", counts_csv)
counts_mat <- read.csv(
  counts_csv,
  row.names = 1,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
message("  • Genes:    ", nrow(counts_mat))
message("  • Barcodes: ", ncol(counts_mat))

# -------------------------------
# 3. Create initial Seurat object
# -------------------------------
message("Creating Seurat object from counts matrix.")
seurat_obj <- CreateSeuratObject(counts = counts_mat)
rm(counts_mat)

# -------------------------------
# 4. Read tissue_positions_list.csv
# -------------------------------
coords_csv <- file.path(spatial_dir, "tissue_positions_list.csv")
if (!file.exists(coords_csv)) {
  stop("Cannot find tissue_positions_list.csv in ", spatial_dir)
}
message("Reading spatial coordinates from: ", coords_csv)
coords_df <- read.csv(
  coords_csv,
  header = FALSE,
  stringsAsFactors = FALSE
)

# Assign column names exactly as Seurat expects
colnames(coords_df) <- c(
  "barcode",
  "in_tissue",
  "array_row",
  "array_col",
  "pxl_row_in_fullres",
  "pxl_col_in_fullres"
)
rownames(coords_df) <- coords_df$barcode

# creating "row" and "col" for seurat_to_spaco wrapper
coords_df$row <- coords_df$array_row
coords_df$col <- coords_df$array_col

# Subset coordinates to barcodes present in Seurat object
seurat_bcs <- colnames(seurat_obj)
coords_bcs <- rownames(coords_df)
common_bcs <- intersect(coords_bcs, seurat_bcs)

# 4b. If nothing matches, try stripping a trailing “-1” from the Seurat barcodes
common_bcs <- intersect(rownames(coords_df), colnames(seurat_obj))
if (length(common_bcs) == 0) {
  #Replace ".1" at end of Seurat barcodes with "-1":
  seurat_bcs_clean <- sub("\\.1$", "-1", colnames(seurat_obj))
  colnames(seurat_obj) <- seurat_bcs_clean
  common_bcs <- intersect(coords_bcs, seurat_bcs_clean)
}
coords_df <- coords_df[common_bcs, , drop = FALSE]
message("  • Number of matching barcodes: ", length(common_bcs))
# -------------------------------
# 5. Load scale factors JSON
# -------------------------------
scalefactors_json <- file.path(spatial_dir, "scalefactors_json.json")
if (!file.exists(scalefactors_json)) {
  stop("Cannot find scalefactors_json.json in ", spatial_dir)
}
message("Reading scale factors JSON from: ", scalefactors_json)
raw_sf_list <- fromJSON(scalefactors_json)

# Inspect the names in raw_sf_list to confirm what’s available:
message("Names in raw scale.factors list (from JSON):")
print(names(raw_sf_list))

# Typical JSON from 10x Visium contains at least these keys:
#   • "spot_diameter_fullres"         (numeric)
#   • "fiducial_diameter_fullres"     (numeric)
#   • "tissue_hires_scalef"           (numeric)
#   • "tissue_lowres_scalef"          (numeric)
#
# Make sure those keys exist:
required_keys <- c(
  "spot_diameter_fullres",
  "fiducial_diameter_fullres",
  "tissue_hires_scalef",
  "tissue_lowres_scalef"
)
missing_keys <- setdiff(required_keys, names(raw_sf_list))
if (length(missing_keys) > 0) {
  stop(
    "The JSON is missing the following required keys: ",
    paste(missing_keys, collapse = ", ")
  )
}

# Now build a 'scalefactors' object (S4) from these four values:
message("Constructing a 'scalefactors' object …")
sf_obj <- scalefactors(
  spot    = raw_sf_list$spot_diameter_fullres,
  fiducial = raw_sf_list$fiducial_diameter_fullres,
  hires   = raw_sf_list$tissue_hires_scalef,
  lowres  = raw_sf_list$tissue_lowres_scalef
)
# Confirm class:
message("  • class(sf_obj) → ", class(sf_obj))

# -------------------------------
# 6. Load low‐res tissue image
# -------------------------------
# Read10X_Image() may already return a VisiumV2 object (depending on your Seurat version).
# We’ll extract the raw image array from whatever `Read10X_Image()` gives us,
# then re-package it into a fresh VisiumV2 with our custom coordinates + scale factors.
message("Loading tissue image (low‐res PNG) via Read10X_Image().")
img_raw <- Read10X_Image(
  image.dir  = spatial_dir,
  image.name = "tissue_lowres_image.png"
)

# Check class and grab the raw raster array:
if ("VisiumV2" %in% class(img_raw)) {
  message("  • Detected an existing VisiumV2 object. Extracting its @image slot.")
  img_array <- img_raw@image
} else if ("VisiumV1" %in% class(img_raw)) {
  message("  • Detected an existing VisiumV1 object. Extracting its @image slot.")
  img_array <- img_raw@image
} else {
  stop("Read10X_Image() did not return a Visium object. Check that tissue_lowres_image.png is present.")
}

# -------------------------------
# 7. Build a new VisiumV2 object with the correct 'scalefactors' slot
# -------------------------------
message("Loading tissue image (low-res PNG) via Read10X_Image().")
img_raw <- Read10X_Image(
  image.dir  = spatial_dir,
  image.name = "tissue_lowres_image.png"
)

if (!("VisiumV2" %in% class(img_raw) || "VisiumV1" %in% class(img_raw))) {
  stop("Read10X_Image() did not return a Visium object. Check tissue_lowres_image.png.")
}

# -------------------------------
# 8. Overwrite the image's scale.factors slot
# -------------------------------
message("Overwriting @scale.factors on img_raw …")
# In Seurat ≥ v5, img_raw@scale.factors must be a 'scalefactors' object:
img_raw@scale.factors <- sf_obj

# -------------------------------
# 9. Build spatial assay slot 
# -------------------------------
# If you haven’t already created a Spatial assay, do:
# (Replace “Spatial” with whatever your spatial assay is called)
if (!("Spatial" %in% names(seurat_obj@assays))) {
  seurat_obj[["Spatial"]] <- seurat_obj[["RNA"]]  # or however you want
  DefaultAssay(seurat_obj) <- "Spatial"
} else {
  DefaultAssay(seurat_obj) <- "Spatial"
}
        
# -------------------------------
# 10. Attach VisiumV2 image to Seurat object
# -------------------------------
message("Attaching Visium image to seurat_obj@images$image …")
seurat_obj[["image"]] <- img_raw

# -------------------------------
# 11. Copy spot‐level coordinates into seurat_obj@meta.data
# -------------------------------
message("Copying spatial coordinates into seurat_obj@meta.data …")
seurat_obj@meta.data[rownames(coords_df), "array_row"]          <- coords_df$array_row
seurat_obj@meta.data[rownames(coords_df), "array_col"]          <- coords_df$array_col
seurat_obj@meta.data[rownames(coords_df), "pxl_row_in_fullres"] <- coords_df$pxl_row_in_fullres
seurat_obj@meta.data[rownames(coords_df), "pxl_col_in_fullres"] <- coords_df$pxl_col_in_fullres
seurat_obj@meta.data[rownames(coords_df), "in_tissue"]          <- coords_df$in_tissue

# and if you also write:
seurat_obj@meta.data[rownames(coords_df), "row"] <- coords_df$row
seurat_obj@meta.data[rownames(coords_df), "col"] <- coords_df$col
#----------------------------------

# sanity check 
seurat_obj <- SCTransform(
  object = seurat_obj,
  assay  = "Spatial",
  verbose = FALSE
)

saveRDS(seurat_obj, '/home/rstudio/data_dir/R_scripts/SPACO_paper_data/young_liver_spac.RDS')

