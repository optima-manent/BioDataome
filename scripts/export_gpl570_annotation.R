#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(AnnotationDbi)
  library(hgu133plus2.db)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: export_gpl570_annotation.R <probes.txt> <mapping.tsv> <manifest.json>")
}

probe_path <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
output_path <- args[[2]]
manifest_path <- args[[3]]
probes <- readLines(probe_path, warn = FALSE)
probes <- unique(trimws(probes[nzchar(trimws(probes))]))

mapping <- AnnotationDbi::select(
  hgu133plus2.db,
  keys = probes,
  keytype = "PROBEID",
  columns = c("ENTREZID", "SYMBOL", "GENENAME")
)
mapping <- mapping[order(mapping$PROBEID, mapping$ENTREZID, na.last = TRUE), ]
write.table(mapping, output_path, sep = "\t", row.names = FALSE, quote = FALSE, na = "")

manifest <- list(
  schema = "gpl570-probe-annotation-v1",
  platform = "GPL570",
  feature_count = length(probes),
  mapped_probe_count = length(unique(mapping$PROBEID[!is.na(mapping$ENTREZID) & nzchar(mapping$ENTREZID)])),
  mapping_row_count = nrow(mapping),
  keytype = "PROBEID",
  columns = c("ENTREZID", "SYMBOL", "GENENAME"),
  r_version = R.version.string,
  bioconductor_version = as.character(BiocManager::version()),
  annotation_package = "hgu133plus2.db",
  annotation_package_version = as.character(packageVersion("hgu133plus2.db")),
  annotation_db_schema = AnnotationDbi::dbmeta(hgu133plus2_dbconn(), "DBSCHEMA")
)
write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE)
