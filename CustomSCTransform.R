CustomSCTransform <- function(
    object,
    assay = NULL,
    new.assay.name = "SCT",
    do.correct.umi = FALSE,
    ncells = 5000,
    n_genes = 2000,
    variable.features.n = NULL,
    variable.features.rv.th = 1.3,
    vars.to.regress = NULL,
    do.scale = FALSE,
    do.center = TRUE,
    clip.range = c(-sqrt(x = ncol(x = object)), sqrt(x = ncol(x = object))),
    conserve.memory = FALSE,
    return.only.var.genes = TRUE,
    seed.use = 1448145,
    verbose = TRUE,
    useNames = TRUE
) {
  requireNamespace("sctransform")
  assay <- assay %||% DefaultAssay(object = object)
  if (verbose) {
    message("Calculating cell attributes from input UMI matrix")
  }
  umi <- GetAssayData(object = object, assay = assay, slot = "counts")
  if (do.correct.umi) {
    corrected.umi <- sctransform::correct_counts(umi = umi, ...)
    umi <- corrected.umi
  }
  if (verbose) {
    message("Variance stabilizing transformation of count matrix of size ",
            nrow(x = umi), " by ", ncol(x = umi))
  }
  vst.out <- sctransform::vst(
    umi,
    n_genes = n_genes,
    return_gene_attr = TRUE,
    return_cell_attr = TRUE,
    verbosity = ifelse(test = verbose, yes = 1, no = 0)
  )
  if (verbose) {
    message("Model formula is: ", vst.out$model_str)
  }
  sct.data <- vst.out$y
  if (do.scale) {
    sct.data <- ScaleData(
      object = sct.data,
      features = rownames(x = sct.data),
      vars.to.regress = vars.to.regress,
      do.scale = do.scale,
      do.center = do.center,
      clip.range = clip.range,
      verbose = verbose
    )
  }
  object[[new.assay.name]] <- CreateAssayObject(counts = sct.data)
  if (return.only.var.genes) {
    object[[new.assay.name]] <- subset(
      x = object[[new.assay.name]],
      features = vst.out$gene_attr$gene_ids[1:variable.features.n]
    )
  }
  if (!is.null(x = variable.features.n)) {
    object <- FindVariableFeatures(
      object = object,
      assay = new.assay.name,
      selection.method = "vst",
      nfeatures = variable.features.n,
      verbose = verbose
    )
  } else if (!is.null(x = variable.features.rv.th)) {
    object <- FindVariableFeatures(
      object = object,
      assay = new.assay.name,
      selection.method = "vst",
      mean.function = ExpMean,
      dispersion.function = LogVMR,
      loess.span = 0.3,
      clip.max = "auto",
      nfeatures = NULL,
      mean.cutoff = c(0.1, 8),
      dispersion.cutoff = c(variable.features.rv.th, Inf),
      verbose = verbose
    )
  }
  return(object)
}
