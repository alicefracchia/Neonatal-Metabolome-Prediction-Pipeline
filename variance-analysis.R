###############################################################
# Neonatal Metabolome Association Analysis
# 
# This script:
# 1) Runs PERMANOVA to test multivariate association between
#    metabolomic profiles and clinical predictors.
# 2) Runs LASSO regression for each predictor with permutation
#    testing to assess significance of predictive association.
#
# Author: Alice
# Requirements: R >= 4.0
###############################################################

# -------------------------------
# 1. Load required libraries
# -------------------------------

library(dplyr)
library(vegan)      # PERMANOVA
library(glmnet)     # LASSO / Elastic Net
library(pROC)       # AUC
library(foreach)    # parallel loops
library(doParallel)

# Print package versions for reproducibility
pkgs <- c("dplyr","vegan","glmnet","pROC","foreach","doParallel")
sapply(pkgs, packageVersion)

# Set seed for reproducibility
set.seed(1234)

# -------------------------------
# 2. Load and preprocess data
# -------------------------------

file_path <- "/ngc/projects2/predict_r/research/projects/0053_genes_x_metabolites_birth/my_project/"
df <- read.csv(paste0(file_path, "met_final_res_by_sampday.csv"))

# Random dummy variable (example negative control)
df$random <- sample(c(0,1), nrow(df), replace = TRUE)

# Convert birth date to numeric days since earliest date
df$birth_date_clean <- as.Date(df$birth_date)
min_date <- min(df$birth_date_clean, na.rm = TRUE)
df$birth_days <- as.numeric(df$birth_date_clean - min_date)

# -------------------------------
# 3. Define metabolite block and predictors
# -------------------------------

start_col_met <- match("X1", colnames(df))
end_col_met   <- match("X1350", colnames(df))
outcomes_met  <- colnames(df)[start_col_met:end_col_met]

# Clinical / demographic predictors
outcomes_fac <- c(
  "gestational_days","delivery_mode","birth_weight","sex",
  "BMI","smoker_last_trimester","ever_smoker",
  "mother_ibd_status","father_ibd_status","parental_ibd",
  "ibd_status","age_at_diagnosis","pediatric","VEO",
  "UC_prs","CD_prs","random","sampling_days","birth_year"
)

# -------------------------------
# 4. PERMANOVA analysis
# -------------------------------

run_permanova <- function(outcomes, data_input, dataset_name) {
  
  results <- data.frame(
    Predictor = character(),
    F_Model   = numeric(),
    R2        = numeric(),
    p_value   = numeric(),
    Dataset   = character(),
    stringsAsFactors = FALSE
  )
  
  for (predictor in outcomes_fac) {
    
    # Keep complete cases for predictor + metabolites
    subset_df <- data_input[complete.cases(data_input[, c(predictor, outcomes)]), ]
    
    # Skip if predictor has no variance
    if (length(unique(subset_df[[predictor]])) < 2) next
    
    # Scale metabolomics and compute Euclidean distance
    Y_scaled <- scale(subset_df[, outcomes])
    distance_matrix <- dist(Y_scaled, method = "euclidean")
    
    cat("N =", nrow(subset_df), "for predictor:", predictor, "\n")
    
    # PERMANOVA
    permanova <- adonis2(distance_matrix ~ subset_df[[predictor]],
                          permutations = 1000)
    
    results <- rbind(results,
      data.frame(
        Predictor = predictor,
        F_Model   = permanova[1,"F"],
        R2        = permanova[1,"R2"],
        p_value   = permanova[1,"Pr(>F)"],
        Dataset  = dataset_name
      )
    )
  }
  
  return(results)
}

permanova_results <- run_permanova(outcomes_met, df, "df")
print(permanova_results)

# -------------------------------
# 5. Metric functions
# -------------------------------

# R² for continuous outcomes
r2_fn <- function(y, y_pred) {
  1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
}

# AUC for binary outcomes
auc_fn <- function(y, y_prob) {
  pROC::roc(y, y_prob, quiet = TRUE)$auc
}

# -------------------------------
# 6. Permutation test for LASSO
# -------------------------------

get_permutation_pvalue <- function(X, y, metric_fn,
                                   observed_score, family,
                                   n_perm = 500,
                                   n_cores = min(4, parallel::detectCores()-1)) {
  
  type <- ifelse(family == "binomial", "response", "link")
  
  cl <- makeCluster(n_cores)
  on.exit(stopCluster(cl), add = TRUE)
  registerDoParallel(cl)
  
  perm_scores <- foreach(i = 1:n_perm,
                          .combine = c,
                          .packages = c("glmnet","pROC")) %dopar% {
    
    y_perm <- sample(y)
    cvfit_perm <- cv.glmnet(X, y_perm, alpha = 1, family = family)
    y_pred <- predict(cvfit_perm, newx = X,
                      s = "lambda.min", type = type)
    metric_fn(y_perm, y_pred)
  }
  
  # One-sided permutation p-value
  mean(perm_scores >= observed_score)
}

# -------------------------------
# 7. LASSO association analysis
# -------------------------------

run_lasso <- function(data, outcomes_met, dataset_name, n_perm = 500) {
  
  results <- list()
  
  for (predictor in outcomes_fac) {
    
    subset_df <- data[complete.cases(data[, c(predictor, outcomes_met)]), ]
    if (length(unique(subset_df[[predictor]])) < 2) next
    
    X <- as.matrix(subset_df[, outcomes_met])
    y <- subset_df[[predictor]]
    
    # Determine outcome type
    if (is.numeric(y)) {
      family    <- "gaussian"
      metric    <- "R2"
      metric_fn <- r2_fn
      pred_type <- "link"
      
    } else if (length(unique(y)) == 2) {
      family    <- "binomial"
      metric    <- "AUC"
      metric_fn <- auc_fn
      y         <- factor(y)
      pred_type <- "response"
      
    } else {
      next
    }
    
    # Fit LASSO with cross-validated lambda
    cvfit <- cv.glmnet(X, y, alpha = 1, family = family)
    y_pred <- predict(cvfit, newx = X,
                      s = "lambda.min", type = pred_type)
    
    score <- metric_fn(y, y_pred)
    
    # Permutation p-value
    pval <- get_permutation_pvalue(X, y, metric_fn,
                                   score, family, n_perm)
    
    results[[predictor]] <- data.frame(
      Predictor = predictor,
      Metric    = metric,
      Score     = score,
      Pvalue    = pval,
      Dataset  = dataset_name,
      stringsAsFactors = FALSE
    )
    
    cat("Completed LASSO for:", predictor, "\n")
  }
  
  do.call(rbind, results)
}

lasso_results <- run_lasso(df, outcomes_met, "df", n_perm = 100)
print(lasso_results)
