#!/usr/bin/env Rscript
# Mixed models and figures, in the manuscript order.
# 1. Behavior and ratings read results/derived and raw_data/Rating.xlsx,
#    and write results/behavior/.
# 2. Choice models are estimated by fitting.py into results/fits/ and
#    results/model_comparison/. Recovery and simulation tables are read
#    from results/recovery/ and results/simulation/.
# 3. fitting.py joins Model 2 utilities and cue P3/LPP into
#    results/derived/merged_behavior_erp_utilities.csv.
# 4. Cooperation-task choice and decision time, including the quadratic
#    expected-value model, write results/model_behavior/.
# 5. Cue P3 and LPP write results/erp/.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(MASS)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(emmeans)
})

args <- commandArgs(trailingOnly = TRUE)
this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
if (length(args) >= 1) {
  root <- normalizePath(args[[1]])
} else if (length(this_file) == 1) {
  root <- normalizePath(file.path(dirname(this_file), ".."))
} else {
  stop("Pass the upload_package root as the first argument, or run with Rscript.")
}

derived_path <- file.path(root, "results", "derived", "merged_behavior_erp_utilities.csv")
behavior_dir <- file.path(root, "results", "behavior")
model_behavior_dir <- file.path(root, "results", "model_behavior")
erp_dir <- file.path(root, "results", "erp")
main_dir <- file.path(root, "figures", "main_figures")
supp_dir <- file.path(root, "figures", "supp_figures")
for (path in c(behavior_dir, model_behavior_dir, erp_dir, main_dir, supp_dir)) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}
legacy_dir <- file.path(root, "results", "downstream")
if (dir.exists(legacy_dir)) unlink(legacy_dir, recursive = TRUE)

task_colors <- c(Risk = "#265C9C", Cooperation = "#EB6B29")
util_colors <- c(Loss = "#265C9C", Betrayal = "#EB6B29")
prob_labels <- c("10%", "30%", "50%", "70%", "90%")
stake_labels <- c("0.2", "0.4", "0.6", "0.8", "1")

save_plot <- function(plot, path, width, height) {
  ggsave(path, plot, width = width, height = height, dpi = 1200, bg = "white")
  message("Wrote ", path)
}

theme_behavior <- function() {
  theme_bw(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(linewidth = 0.6, colour = "black"),
      legend.position = "top",
      legend.direction = "horizontal",
      plot.title = element_text(hjust = 0.5)
    )
}

theme_utility <- function() {
  theme_bw(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(colour = "black", linewidth = 0.8, fill = NA),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.title = element_blank(),
      axis.text = element_text(colour = "black"),
      plot.title = element_text(hjust = 0.5)
    )
}

theme_heat <- function() {
  theme_bw(base_size = 11) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 40, hjust = 1),
      plot.title = element_text(hjust = 0.5)
    )
}

label_task <- function(task) {
  factor(ifelse(task == "single", "Risk", "Cooperation"), levels = c("Risk", "Cooperation"))
}

subject_cells <- function(data, y, x) {
  out <- data %>%
    group_by(Subject, .data[[x]], task) %>%
    summarise(y = mean(.data[[y]], na.rm = TRUE), .groups = "drop")
  names(out)[names(out) == x] <- "x"
  out
}

# Cousineau–Morey interval on subject-cell means (summarySEwithin, one row per cell).
within_summary <- function(cells) {
  n_within <- dplyr::n_distinct(interaction(cells$x, cells$task))
  morey <- sqrt(n_within / (n_within - 1))
  subj <- cells %>% group_by(Subject) %>% summarise(subj_m = mean(y), .groups = "drop")
  grand <- mean(subj$subj_m)
  n_subj <- nrow(subj)
  crit <- qt(0.975, n_subj - 1)
  cells %>%
    left_join(subj, by = "Subject") %>%
    mutate(normed = y - subj_m + grand) %>%
    group_by(x, task) %>%
    summarise(
      y = mean(y),
      se = sd(normed) / sqrt(dplyr::n()) * morey,
      ci = sd(normed) / sqrt(dplyr::n()) * morey * crit,
      .groups = "drop"
    )
}

theme_ms <- function() {
  theme_bw(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(linewidth = 0.6, colour = "black"),
      axis.text = element_text(colour = "black"),
      legend.position = "top",
      legend.direction = "horizontal",
      plot.title = element_text(hjust = 0.5, size = 12),
      plot.tag = element_text(face = "bold", size = 14),
      plot.tag.position = "topleft"
    )
}

# Manuscript Figure 2 uses orange for risk and blue for cooperation.
manuscript_task_colors <- c(Risk = "#EB6B29", Cooperation = "#265C9C")

behavior_panel <- function(cells, summary_df, xlab, breaks, labels, ylim, ylab, vline = NULL, percent = FALSE) {
  points <- cells %>%
    mutate(
      task_label = label_task(task),
      x_point = x + ifelse(task == "single", -0.25, -0.10)
    )
  means <- summary_df %>%
    mutate(
      task_label = label_task(task),
      x_mean = x + 0.25
    )
  p <- ggplot()
  if (!is.null(vline)) {
    p <- p + geom_vline(xintercept = vline, linetype = "dashed", colour = "grey60", linewidth = 0.4)
  }
  p <- p +
    geom_point(
      data = points,
      aes(x_point, y, colour = task_label),
      position = position_jitter(width = 0.05, height = 0, seed = 1),
      size = 2.6, alpha = 0.35
    ) +
    geom_errorbar(
      data = means,
      aes(x = x_mean, ymin = y - ci, ymax = y + ci, colour = task_label),
      width = 0.22, linewidth = 0.7, position = position_dodge(width = 0.35)
    ) +
    geom_line(
      data = means,
      aes(x = x_mean, y = y, colour = task_label, group = task_label),
      linewidth = 0.7, position = position_dodge(width = 0.35)
    ) +
    geom_point(
      data = means,
      aes(x = x_mean, y = y, colour = task_label),
      shape = 15, size = 2.8, position = position_dodge(width = 0.35)
    ) +
    scale_colour_manual(values = manuscript_task_colors, name = "Task") +
    scale_x_continuous(breaks = breaks, labels = labels) +
    coord_cartesian(ylim = ylim) +
    labs(x = xlab, y = ylab) +
    theme_ms()
  if (percent) {
    p <- p + scale_y_continuous(
      labels = scales::label_percent(accuracy = 1),
      breaks = seq(0, 1, 0.2),
      name = ylab
    )
  }
  p
}

behavior_panels <- function(data, y, ylim, ylab, percent = FALSE, vline = NULL) {
  prob_cells <- subject_cells(data, y, "pro")
  stake_cells <- subject_cells(data, y, "montype")
  list(
    probability = behavior_panel(
      prob_cells, within_summary(prob_cells),
      "Probability", 1:5, prob_labels, ylim, ylab, vline, percent
    ),
    principal = behavior_panel(
      stake_cells, within_summary(stake_cells),
      "Principal", 1:5, paste0("\u00a5", stake_labels), ylim, ylab, NULL, percent
    )
  )
}

design_band <- function(model, xvar, fixed, xseq, binomial) {
  grid <- expand.grid(
    x = xseq,
    task = factor(c("single", "double"), levels = c("single", "double")),
    KEEP.OUT.ATTRS = FALSE
  )
  contrasts(grid$task) <- contr.sdif(2)
  grid[[xvar]] <- grid$x
  for (nm in names(fixed)) grid[[nm]] <- fixed[[nm]]
  pr <- predict(model, newdata = grid, re.form = NA, type = "link", se.fit = TRUE)
  fit <- as.numeric(pr$fit)
  se <- as.numeric(pr$se.fit)
  if (binomial) {
    se <- stats::dlogis(fit) * se
    fit <- stats::plogis(fit)
  }
  grid$fit <- fit
  grid$lo <- fit - 1.96 * se
  grid$hi <- fit + 1.96 * se
  grid$task_label <- label_task(grid$task)
  grid
}

design_panel <- function(model, xvar, fixed, xlab, ylab, binomial = FALSE, percent = FALSE, title = NULL) {
  grid <- design_band(model, xvar, fixed, seq(-1.2, 1.2, by = 0.02), binomial)
  p <- ggplot(grid, aes(x, fit, colour = task_label, fill = task_label)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.28, colour = NA) +
    geom_line(linewidth = 0.8) +
    scale_colour_manual(values = manuscript_task_colors, name = "Task") +
    scale_fill_manual(values = manuscript_task_colors, name = "Task") +
    scale_x_continuous(breaks = seq(-1, 1, 0.5)) +
    labs(x = xlab, y = ylab, title = title) +
    theme_ms()
  if (percent) {
    p <- p +
      scale_y_continuous(labels = scales::label_percent(accuracy = 1), breaks = seq(0, 1, 0.25), name = ylab) +
      coord_cartesian(xlim = c(-1, 1), ylim = c(0, 1))
  } else {
    p <- p + coord_cartesian(xlim = c(-1, 1))
  }
  p
}

fe_table <- function(model, name, engine, re_formula) {
  s <- coef(summary(model))
  cn <- colnames(s)
  pcol <- if ("Pr(>|z|)" %in% cn) "Pr(>|z|)" else if ("Pr(>|t|)" %in% cn) "Pr(>|t|)" else NA
  zcol <- if ("z value" %in% cn) "z value" else if ("t value" %in% cn) "t value" else NA
  data.frame(
    model = name,
    term = rownames(s),
    estimate = s[, "Estimate"],
    std_error = s[, "Std. Error"],
    ci_95_low = s[, "Estimate"] - 1.96 * s[, "Std. Error"],
    ci_95_high = s[, "Estimate"] + 1.96 * s[, "Std. Error"],
    z = if (!is.na(zcol)) s[, zcol] else NA_real_,
    p = if (!is.na(pcol)) s[, pcol] else NA_real_,
    engine = engine,
    re_formula = re_formula,
    loglik = as.numeric(logLik(model)),
    aic = AIC(model),
    nobs = nobs(model),
    stringsAsFactors = FALSE
  )
}

paper_contrast <- function(model, name, a = "zloss", b = "zdefect", use_abs = FALSE) {
  coefs <- fixef(model)
  se <- sqrt(diag(vcov(model)))
  va <- if (use_abs) abs(as.numeric(coefs[[a]])) else as.numeric(coefs[[a]])
  vb <- if (use_abs) abs(as.numeric(coefs[[b]])) else as.numeric(coefs[[b]])
  z_paper <- (va - vb) / sqrt(se[[a]]^2 + se[[b]]^2)
  cvec <- setNames(rep(0, length(coefs)), names(coefs))
  cvec[[a]] <- 1
  cvec[[b]] <- -1
  est <- as.numeric(sum(cvec * coefs))
  se_w <- sqrt(as.numeric(t(cvec) %*% vcov(model) %*% cvec))
  z_w <- est / se_w
  data.frame(
    model = name,
    contrast = "zloss_minus_zdefect",
    estimate = est,
    std_error_paper = sqrt(se[[a]]^2 + se[[b]]^2),
    z_paper = z_paper,
    p_paper_one_tailed = pnorm(-abs(z_paper)),
    p_paper_two_tailed = 2 * pnorm(-abs(z_paper)),
    std_error = se_w,
    z = z_w,
    p = 2 * pnorm(-abs(z_w)),
    use_abs = use_abs,
    defect_larger_signed = as.numeric(coefs[[b]]) > as.numeric(coefs[[a]]),
    stringsAsFactors = FALSE
  )
}

pop_curve <- function(model, data, varying, held, binomial) {
  x <- seq(
    round(min(data[[varying]], na.rm = TRUE), 1),
    round(max(data[[varying]], na.rm = TRUE), 1),
    by = 0.1
  )
  newdata <- setNames(data.frame(x, rep(0, length(x))), c(varying, held))
  pr <- predict(model, newdata = newdata, re.form = NA, type = "link", se.fit = TRUE)
  if (binomial) {
    fit <- plogis(pr$fit)
    se <- dlogis(pr$fit) * pr$se.fit
  } else {
    fit <- pr$fit
    se <- pr$se.fit
  }
  data.frame(x = x, fit = fit, se = se, stringsAsFactors = FALSE)
}

slope_table <- function(model, by, var, at, model_name, spec) {
  tab <- as.data.frame(summary(emtrends(model, by, var = var, at = at), infer = c(TRUE, TRUE)))
  tab$model <- model_name
  tab$specification <- spec
  tab
}

wald_combo <- function(model, weights, model_name, label) {
  coefs <- fixef(model)
  L <- setNames(rep(0, length(coefs)), names(coefs))
  for (nm in names(weights)) L[[nm]] <- weights[[nm]]
  est <- as.numeric(sum(L * coefs))
  se <- sqrt(as.numeric(t(L) %*% vcov(model) %*% L))
  z <- est / se
  data.frame(
    model = model_name,
    contrast = label,
    estimate = est,
    std_error = se,
    z = z,
    chi_square = z^2,
    p = 2 * pnorm(-abs(z)),
    stringsAsFactors = FALSE
  )
}

trend_pair <- function(model, by, var, model_name, label, at = NULL) {
  tr <- if (is.null(at)) {
    emtrends(model, by, var = var)
  } else {
    emtrends(model, by, var = var, at = at)
  }
  tab <- as.data.frame(pairs(tr))
  z <- if ("z.ratio" %in% names(tab)) tab$z.ratio else tab$t.ratio
  data.frame(
    model = model_name,
    contrast = paste(label, tab$contrast),
    estimate = tab$estimate,
    std_error = tab$SE,
    z = z,
    chi_square = z^2,
    p = tab$p.value,
    stringsAsFactors = FALSE
  )
}

read_coop_ratings <- function(path) {
  py <- Sys.getenv("PYTHON", unset = "")
  if (!nzchar(py)) py <- Sys.which("python3")
  if (!nzchar(py)) py <- Sys.which("python")
  if (!nzchar(py)) stop("Set PYTHON to an interpreter with openpyxl to read Rating.xlsx.")
  tmp <- tempfile(fileext = ".csv")
  pyfile <- tempfile(fileext = ".py")
  on.exit(unlink(c(tmp, pyfile)), add = TRUE)
  writeLines(
    sprintf(
      'import pandas as pd\npd.read_excel(r"%s", sheet_name="coop").to_csv(r"%s", index=False)\n',
      normalizePath(path, winslash = "/"),
      normalizePath(tmp, winslash = "/", mustWork = FALSE)
    ),
    pyfile
  )
  status <- system2(py, pyfile)
  if (!identical(status, 0L)) stop("Could not read ", path)
  read.csv(tmp, check.names = FALSE)
}

rating_anova_table <- function(model) {
  parts <- summary(model)
  bind_rows(lapply(names(parts), function(nm) {
    tab <- as.data.frame(parts[[nm]][[1]], stringsAsFactors = FALSE)
    tab$term <- trimws(rownames(tab))
    tab$stratum <- sub("^Error: ", "", nm)
    rownames(tab) <- NULL
    names(tab)[names(tab) == "Df"] <- "df"
    names(tab)[names(tab) == "Sum Sq"] <- "sum_sq"
    names(tab)[names(tab) == "Mean Sq"] <- "mean_sq"
    names(tab)[names(tab) == "F value"] <- "f"
    names(tab)[names(tab) == "Pr(>F)"] <- "p"
    tab
  }))
}

rating_figure <- function(cells) {
  fb_colors <- c(Bad = "#EB6B29", Good = "#265C9C")
  means <- within_summary(cells) %>%
    mutate(
      task_label = label_task(task),
      fb_label = factor(ifelse(x == "good", "Good", "Bad"), levels = c("Bad", "Good"))
    )
  ggplot(means, aes(task_label, y, fill = fb_label, group = fb_label)) +
    geom_line(
      colour = "grey70",
      position = position_dodge(width = 0.2),
      linewidth = 0.7
    ) +
    geom_errorbar(
      aes(ymin = y - se, ymax = y + se),
      width = 0.12, linewidth = 0.6, colour = "black",
      position = position_dodge(width = 0.2)
    ) +
    geom_point(
      shape = 22, size = 3.4, colour = "black",
      position = position_dodge(width = 0.2)
    ) +
    scale_fill_manual(values = fb_colors, name = "Feedback") +
    scale_y_continuous(breaks = seq(2.5, 8.5, 1)) +
    coord_cartesian(ylim = c(2.5, 8.5)) +
    labs(x = "Task", y = "Happy Rating") +
    theme_ms()
}

accept_count_figure <- function(data) {
  counts <- data %>%
    group_by(Subject, task) %>%
    summarise(N = sum(Choice == 1), .groups = "drop") %>%
    mutate(task_label = label_task(task))
  ggplot(counts, aes(N, fill = task_label, group = task_label)) +
    geom_density(alpha = 0.2, linewidth = 1.1, adjust = 1.5, colour = "black") +
    scale_fill_manual(values = c(Risk = "#206FB0", Cooperation = "#DD2624"), name = NULL) +
    scale_y_continuous(limits = c(0, 0.025), breaks = seq(0, 0.025, 0.005), expand = c(0, 0)) +
    scale_x_continuous(limits = c(40, 180), breaks = seq(40, 180, 25)) +
    labs(x = "Number accepted", y = "Density") +
    theme_ms()
}

utility_panel <- function(model, data, ylab, binomial = FALSE, percent = FALSE, xlim = c(-2, 0), ylim = NULL, open_frame = FALSE, legend = c("inside", "inside_left", "none")) {
  legend <- match.arg(legend)
  loss <- pop_curve(model, data, "zloss", "zdefect", binomial)
  defect <- pop_curve(model, data, "zdefect", "zloss", binomial)
  loss$predictor <- "Loss"
  defect$predictor <- "Betrayal"
  curve <- bind_rows(loss, defect)
  curve$predictor <- factor(curve$predictor, levels = c("Betrayal", "Loss"))
  p <- ggplot(curve, aes(x, fit, colour = predictor, fill = predictor)) +
    geom_ribbon(aes(ymin = fit - se, ymax = fit + se), alpha = 0.25, colour = NA) +
    geom_line(linewidth = 1.1) +
    geom_vline(xintercept = 0, linetype = "dotted", linewidth = 0.6) +
    scale_colour_manual(values = c(Betrayal = "#EB6B29", Loss = "#265C9C"), name = NULL) +
    scale_fill_manual(values = c(Betrayal = "#EB6B29", Loss = "#265C9C"), name = NULL) +
    labs(x = "Model (Normalized)", y = ylab) +
    theme_bw(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      panel.border = if (open_frame) element_blank() else element_rect(colour = "black", linewidth = 1.1, fill = NA),
      axis.line = element_blank(),
      axis.line.x.bottom = if (open_frame) element_line(linewidth = 0.6, colour = "black") else element_blank(),
      axis.line.y.left = if (open_frame) element_line(linewidth = 0.6, colour = "black") else element_blank(),
      legend.position = switch(legend, inside = c(0.78, 0.86), inside_left = c(0.03, 0.97), none = "none"),
      legend.justification = if (legend == "inside_left") c(0, 1) else c(0.5, 0.5),
      legend.background = element_rect(fill = "white", colour = NA),
      axis.text = element_text(colour = "black")
    )
  if (!is.null(xlim) || !is.null(ylim)) {
    p <- p + coord_cartesian(xlim = xlim, ylim = ylim, expand = FALSE)
  }
  if (percent) {
    p <- p + scale_y_continuous(labels = scales::label_percent(accuracy = 1), name = ylab)
  }
  p
}

read_matrix_csv <- function(path) {
  raw <- read.csv(path, check.names = FALSE)
  rn <- raw[[1]]
  mat <- as.matrix(raw[, -1, drop = FALSE])
  mode(mat) <- "numeric"
  rownames(mat) <- rn
  mat
}

manuscript_heat <- function(mat) {
  rename <- c("α_g" = "α_gain", "α_l" = "α_loss", "alpha_g" = "α_gain", "alpha_l" = "α_loss")
  relabel <- function(x) ifelse(x %in% names(rename), unname(rename[x]), x)
  rownames(mat) <- relabel(rownames(mat))
  colnames(mat) <- relabel(colnames(mat))
  order_lab <- c("α_gain", "α_loss", "γ", "λ", "β", "κ")
  order_lab <- order_lab[order_lab %in% rownames(mat)]
  mat <- mat[order_lab, order_lab, drop = FALSE]
  df <- as.data.frame(as.table(mat), stringsAsFactors = FALSE)
  names(df) <- c("Simulated", "Recovered", "r")
  df$Simulated <- factor(df$Simulated, levels = order_lab)
  df$Recovered <- factor(df$Recovered, levels = order_lab)
  ggplot(df, aes(Simulated, Recovered, fill = r)) +
    geom_tile(colour = "white", linewidth = 0.3) +
    geom_text(
      data = df[as.character(df$Simulated) == as.character(df$Recovered), ],
      aes(label = sprintf("%.2f", r)),
      colour = "black", size = 3.5, fontface = "bold"
    ) +
    scale_fill_gradient2(
      low = "#265C9C", mid = "white", high = "#EB6B29",
      midpoint = 0, limits = c(-1, 1), name = "Pearson's r"
    ) +
    coord_fixed() +
    labs(x = "Simulated", y = "Recovered") +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1, colour = "black"),
      axis.text.y = element_text(colour = "black")
    )
}

manuscript_ic <- function(path) {
  model_levels <- c("random_model", "prospect_model", "aversion_linear_model")
  model_colours <- c(M0 = "#A6A6A6", M1 = "#265C9C", M2 = "#EB6B29")
  ic_data <- read.csv(path) %>%
    mutate(Model = factor(model_name, levels = model_levels, labels = c("M0", "M1", "M2"))) %>%
    pivot_longer(c(aic, bic, looic), names_to = "Metric", values_to = "IC") %>%
    mutate(
      Metric = factor(Metric, levels = c("aic", "bic", "looic"), labels = c("AIC", "BIC", "CVIC")),
      x = as.numeric(Metric) + (as.numeric(Model) - 2) * 0.23
    ) %>%
    filter(is.finite(IC))
  ic_summary <- ic_data %>%
    group_by(Metric, Model, x) %>%
    summarise(Mean = mean(IC), SEM = sd(IC) / sqrt(dplyr::n()), .groups = "drop")
  ggplot() +
    geom_col(
      data = ic_summary, aes(x, Mean, colour = Model),
      fill = "white", width = 0.20, linewidth = 0.8
    ) +
    geom_line(
      data = ic_data, aes(x, IC, group = interaction(subject_id, Metric)),
      colour = "grey80", linewidth = 0.3
    ) +
    geom_point(data = ic_data, aes(x, IC, colour = Model), size = 1.4, alpha = 0.45) +
    geom_errorbar(
      data = ic_summary, aes(x, ymin = Mean - SEM, ymax = Mean + SEM),
      width = 0.07, linewidth = 0.7
    ) +
    scale_colour_manual(values = model_colours, name = NULL) +
    scale_x_continuous(breaks = 1:3, labels = c("AIC", "BIC", "CVIC"), limits = c(0.55, 3.45), expand = c(0, 0)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05)), limits = c(0, NA)) +
    labs(x = NULL, y = "Information criterion") +
    theme_classic(base_size = 12) +
    theme(
      legend.position = "top",
      axis.text = element_text(colour = "black"),
      axis.line = element_line(linewidth = 0.4)
    )
}

manuscript_density <- function(path) {
  parameter_data <- read.csv(path) %>%
    filter(parameter %in% c("lambda", "kappa"), is.finite(estimate)) %>%
    mutate(parameter = factor(parameter, levels = c("kappa", "lambda")))
  density_data <- parameter_data %>%
    group_by(parameter) %>%
    group_modify(~ {
      values <- .x$estimate
      bandwidth <- bw.nrd0(values) * 1.2
      curve <- density(
        c(-values, values), bw = bandwidth,
        from = 0, to = max(values) + 5 * bandwidth, n = 512
      )
      data.frame(x = curve$x, y = 2 * curve$y)
    }) %>%
    ungroup()
  ggplot(density_data, aes(x, y, fill = parameter)) +
    geom_area(alpha = 0.7, colour = NA, position = "identity") +
    geom_line(colour = "black", linewidth = 0.9) +
    facet_wrap(
      ~ parameter, ncol = 1, scales = "free",
      labeller = as_labeller(c(
        kappa = "The best betrayal parameter",
        lambda = "The best loss parameter"
      ))
    ) +
    scale_fill_manual(values = c(kappa = "#EB6B29", lambda = "#265C9C")) +
    scale_x_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.02))) +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
    labs(x = "Parameter value", y = "Density") +
    theme_bw(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(size = 12),
      axis.line = element_line(linewidth = 0.6, colour = "black"),
      legend.position = "none"
    )
}

simulation_panel <- function(summary_df, xlab, task_name, show_title) {
  sub <- summary_df[summary_df$task == task_name, ]
  colour <- unname(task_colors[[task_name]])
  x_labels <- if (xlab == "Instructed probability") paste0(sub$x * 100, "%") else sub$x
  on_left <- task_name == "Risk"
  p <- ggplot(sub, aes(x)) +
    geom_ribbon(aes(ymin = simulated_q025, ymax = simulated_q975), fill = colour, alpha = 0.40) +
    geom_line(aes(y = observed, colour = "Observed mean"), linewidth = 0.8) +
    geom_point(aes(y = observed, colour = "Observed mean"), size = 2) +
    geom_ribbon(
      aes(ymin = simulated_q025, ymax = simulated_q975, fill = "model simulation"),
      alpha = 0
    ) +
    scale_colour_manual(values = c("Observed mean" = "grey10"), name = NULL) +
    scale_fill_manual(values = c("model simulation" = "grey35"), name = NULL) +
    guides(
      fill = guide_legend(order = 1, override.aes = list(alpha = 0.75, colour = NA)),
      colour = guide_legend(order = 2, override.aes = list(linewidth = 0.8))
    ) +
    scale_x_continuous(breaks = sub$x, labels = x_labels) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = xlab, y = NULL, title = if (show_title) task_name else NULL) +
    theme_behavior()
  if (!on_left) {
    p <- p + theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank()
    )
  }
  p
}

simulation_figure <- function(prob_df, stake_df) {
  prep <- function(df, xcol) {
    df$task <- factor(df$task, levels = c("Risk", "Cooperation"))
    df$x <- df[[xcol]]
    df
  }
  prob_df <- prep(prob_df, "prob")
  stake_df <- prep(stake_df, "stake")
  top <- (
    simulation_panel(prob_df, "Instructed probability", "Risk", TRUE) |
      simulation_panel(prob_df, "Instructed probability", "Cooperation", TRUE)
  ) + plot_layout(axis_titles = "collect_x")
  bottom <- (
    simulation_panel(stake_df, "Principal", "Risk", FALSE) |
      simulation_panel(stake_df, "Principal", "Cooperation", FALSE)
  ) + plot_layout(axis_titles = "collect_x")
  body <- (top / bottom) + plot_layout(guides = "collect") & theme(legend.position = "top")
  y_title <- wrap_elements(grid::textGrob(
    "P(accept)", rot = 90, gp = grid::gpar(fontsize = 12)
  ))
  (y_title | body) + plot_layout(widths = c(0.045, 1))
}

param_labeller <- as_labeller(c(
  p = "p",
  alpha_g = "alpha[g]", alpha_l = "alpha[l]", gamma = "gamma",
  lambda = "lambda", beta = "beta", kappa = "kappa"
), label_parsed)

recovery_scatter_row <- function(points, parameters) {
  points$parameter <- factor(points$parameter, levels = parameters)
  ggplot(points, aes(true_value, estimated_value)) +
    geom_point(size = 1.1, alpha = 0.45) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.4) +
    facet_wrap(~ parameter, scales = "free", nrow = 1, labeller = param_labeller) +
    labs(x = "True", y = "Recovered") +
    theme_behavior() +
    theme(
      legend.position = "none",
      strip.background = element_rect(fill = NA, colour = NA),
      strip.text = element_text(size = 12),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 12)
    )
}

add_model_banners <- function(g, groups, labels) {
  strip_idx <- which(grepl("^strip-t-", g$layout$name))
  strip_idx <- strip_idx[order(g$layout$l[strip_idx])]
  if (length(strip_idx) == 0) stop("No strips found for model banners.")
  strip_row <- g$layout$t[strip_idx[1]]
  ref <- g$grobs[[strip_idx[1]]]$grobs[[1]]
  rect_gp <- ref$children[[1]]$gp
  rect_gp$fill <- "#D9D9D9"
  rect_gp$col <- NA
  text_gp <- ref$children[[2]]$children[[1]]$gp
  text_gp$fontsize <- 12
  g <- gtable::gtable_add_rows(g, g$heights[strip_row], pos = strip_row - 1)
  strip_idx <- which(grepl("^strip-t-", g$layout$name))
  strip_idx <- strip_idx[order(g$layout$l[strip_idx])]
  for (i in seq_along(groups)) {
    cells <- strip_idx[groups[[i]]]
    grob <- grid::grobTree(
      grid::rectGrob(gp = rect_gp),
      grid::textGrob(labels[[i]], gp = text_gp)
    )
    g <- gtable::gtable_add_grob(
      g, grob,
      t = strip_row, b = strip_row,
      l = min(g$layout$l[cells]), r = max(g$layout$r[cells]),
      clip = "off",
      name = paste0("model-banner-", labels[[i]])
    )
  }
  g
}

recovery_scatter_figure <- function(rows) {
  plots <- lapply(rows, function(row) recovery_scatter_row(row$points, row$parameters))
  aligned <- patchwork:::align_patches(plots)
  grobs <- lapply(seq_along(rows), function(i) {
    add_model_banners(ggplotGrob(aligned[[i]]), rows[[i]]$groups, rows[[i]]$labels)
  })
  Reduce(`/`, lapply(grobs, function(g) wrap_elements(full = g)))
}

message("Reading ", derived_path)
invalid <- c(5, 6, 51)
dat <- read.csv(derived_path)
dat <- dat[!dat$Subject %in% invalid, ]
dat <- dat[dat$Choice %in% c(0, 1), ]
dat$Subject <- factor(dat$Subject)
dat$Choice <- as.numeric(dat$Choice)
dat$task <- factor(dat$task, levels = c("single", "double"))
contrasts(dat$task) <- contr.sdif(2)

# pro is stored as codes 1-5. Map them to 0.1, 0.3, 0.5, 0.7, 0.9.
stopifnot(all(dat$pro %in% 1:5))
dat <- dat %>%
  mutate(
    p = 0.2 * pro - 0.1,
    ev_safe = mon,
    ev_risky = p * (2 * mon) + (1 - p) * (0.5 * mon),
    ev_diff = ev_risky - ev_safe,
    abs_ev_diff = abs(ev_diff),
    z_ev_diff = as.numeric(scale(ev_diff)),
    z_abs_ev_diff = as.numeric(scale(abs_ev_diff))
  )

dou <- dat[dat$task == "double", ]
dou$zloss <- as.numeric(scale(dou$loss, center = TRUE, scale = TRUE))
dou$zdefect <- as.numeric(scale(dou$defect, center = FALSE, scale = TRUE))

ctrl_glmer <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
ctrl_lmer <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

message("Fitting design-level choice glmer...")
m_choice_des <- glmer(
  Choice ~ zpro * zmon * task + (zpro + task | Subject),
  data = dat, family = binomial, control = ctrl_glmer
)
message("Fitting design-level RT lmer...")
m_rt_des <- lmer(RT ~ zpro * zmon * task + (task | Subject), data = dat, REML = FALSE)
message("Fitting choice glmer (M2 utilities)...")
m_choice <- glmer(
  Choice ~ zloss + zdefect + (zloss + zdefect | Subject),
  data = dou, family = binomial, control = ctrl_glmer
)
message("Fitting RT lmer (M2 utilities)...")
m_rt <- lmer(RT ~ zloss + zdefect + (zdefect | Subject), data = dou, REML = FALSE)
message("Fitting P3 lmer (M2 utilities)...")
m_p3 <- lmer(
  MonP3 ~ zloss + zdefect + (zloss | Subject),
  data = dou, REML = FALSE, control = ctrl_lmer
)
message("Fitting LPP lmer (M2 utilities)...")
m_lpp <- lmer(
  MonLPP ~ zloss + zdefect + (zloss | Subject),
  data = dou, REML = FALSE, control = ctrl_lmer
)
message("Fitting expected-value decision-time lmer...")
m_rt_ev <- lmer(
  RT ~ (z_ev_diff + I(z_ev_diff^2)) * task + (task | Subject),
  data = dat, REML = FALSE
)

fe_behavior <- rbind(
  fe_table(m_choice_des, "choice_design", "lme4::glmer", "(zpro + task | Subject)"),
  fe_table(m_rt_des, "rt_design", "lme4::lmer", "(task | Subject)")
)
fe_model <- rbind(
  fe_table(m_choice, "choice_util", "lme4::glmer", "(zloss + zdefect | Subject)"),
  fe_table(m_rt, "rt_util", "lme4::lmer", "(zdefect | Subject)"),
  fe_table(m_rt_ev, "rt_ev", "lme4::lmer", "(task | Subject)")
)
fe_erp <- rbind(
  fe_table(m_p3, "p3_util", "lme4::lmer", "(zloss | Subject)"),
  fe_table(m_lpp, "lpp_util", "lme4::lmer", "(zloss | Subject)")
)
fe <- rbind(fe_behavior, fe_model, fe_erp)
contrasts_behavior <- rbind(
  paper_contrast(m_choice, "choice_util"),
  paper_contrast(m_rt, "rt_util")
)
contrasts_erp <- rbind(
  paper_contrast(m_p3, "p3_util"),
  paper_contrast(m_lpp, "lpp_util")
)
contrasts_out <- rbind(contrasts_behavior, contrasts_erp)
write.csv(fe_behavior, file.path(behavior_dir, "fixed_effects.csv"), row.names = FALSE)
write.csv(fe_model, file.path(model_behavior_dir, "fixed_effects.csv"), row.names = FALSE)
write.csv(fe_erp, file.path(erp_dir, "fixed_effects.csv"), row.names = FALSE)
write.csv(contrasts_behavior, file.path(model_behavior_dir, "loss_vs_defect_contrasts.csv"), row.names = FALSE)
write.csv(contrasts_erp, file.path(erp_dir, "loss_vs_defect_contrasts.csv"), row.names = FALSE)

desc <- aggregate(cbind(Choice, RT) ~ task, data = dat, FUN = mean, na.rm = TRUE)
names(desc) <- c("task", "accept_rate", "mean_rt")
write.csv(desc, file.path(behavior_dir, "task_descriptives.csv"), row.names = FALSE)

at_list <- list(zpro = c(-1, 1), zmon = c(-1, 1), task = c("single", "double"))
slopes_choice <- bind_rows(
  slope_table(m_choice_des, ~ task, "zpro", at_list, "choice_design", "task"),
  slope_table(m_choice_des, ~ task, "zmon", at_list, "choice_design", "task"),
  slope_table(m_choice_des, ~ zmon, "zpro", at_list, "choice_design", "zmon"),
  slope_table(m_choice_des, ~ zmon * task, "zpro", at_list, "choice_design", "zmon_by_task")
)
slopes_rt <- bind_rows(
  slope_table(m_rt_des, ~ task, "zpro", at_list, "rt_design", "task"),
  slope_table(m_rt_des, ~ task, "zmon", at_list, "rt_design", "task"),
  slope_table(m_rt_des, ~ zmon, "zpro", at_list, "rt_design", "zmon")
)
write.csv(slopes_choice, file.path(behavior_dir, "choice_probability_slopes.csv"), row.names = FALSE)
write.csv(slopes_rt, file.path(behavior_dir, "rt_probability_slopes.csv"), row.names = FALSE)
slope_tests <- bind_rows(
  trend_pair(m_rt_des, ~ zmon, "zpro", "rt_design", "probability slope, low vs high principal", list(zmon = c(-1, 1))),
  trend_pair(m_rt_des, ~ task, "zpro", "rt_design", "probability slope by task"),
  trend_pair(m_rt_des, ~ task, "zmon", "rt_design", "principal slope by task"),
  trend_pair(m_choice_des, ~ zmon, "zpro", "choice_design", "probability slope, low vs high principal", list(zmon = c(-1, 1))),
  trend_pair(m_choice_des, ~ task, "zmon", "choice_design", "principal slope by task"),
  trend_pair(m_choice_des, ~ task, "zpro", "choice_design", "probability slope by task"),
  wald_combo(m_choice_des, c("zpro:task2-1" = 1, "zpro:zmon:task2-1" = 1), "choice_design", "task difference in probability slope at high principal"),
  wald_combo(m_choice_des, c("zpro:task2-1" = 1, "zpro:zmon:task2-1" = -1), "choice_design", "task difference in probability slope at low principal")
)
write.csv(slope_tests, file.path(behavior_dir, "slope_comparisons.csv"), row.names = FALSE)

sink(file.path(behavior_dir, "lme4_model_summaries.txt"))
cat("=== choice_design ===\n"); print(summary(m_choice_des))
cat("\n=== rt_design ===\n"); print(summary(m_rt_des))
sink()
sink(file.path(model_behavior_dir, "lme4_model_summaries.txt"))
cat("=== choice_util ===\n"); print(summary(m_choice))
cat("\n=== rt_util ===\n"); print(summary(m_rt))
cat("\n=== rt_ev ===\n"); print(summary(m_rt_ev))
sink()
sink(file.path(erp_dir, "lme4_model_summaries.txt"))
cat("=== p3_util ===\n"); print(summary(m_p3))
cat("\n=== lpp_util ===\n"); print(summary(m_lpp))
sink()

util_fe <- fe[fe$model %in% c("choice_util", "rt_util", "p3_util", "lpp_util"),
              c("model", "term", "estimate", "std_error", "z", "p")]
print(util_fe)
print(contrasts_out[, c("model", "estimate", "z", "p")])

term_text <- function(model_name, term_name) {
  row <- fe[fe$model == model_name & fe$term == term_name, ]
  if (nrow(row) != 1) return("NA")
  sprintf("%.3f (p=%.3g)", row$estimate, row$p)
}

contrast_p <- function(model_name) {
  row <- contrasts_out[contrasts_out$model == model_name, ]
  if (nrow(row) < 1) return(NA_real_)
  row$p[[1]]
}

note <- c(
  "# Manuscript analyses",
  sprintf("- N subjects: %d", nlevels(dat$Subject)),
  sprintf("- Rows: %d", nrow(dat)),
  "",
  "## Task descriptives"
)
for (i in seq_len(nrow(desc))) {
  note <- c(note, sprintf("- %s: accept=%.3f, RT=%.1f ms", desc$task[i], desc$accept_rate[i], desc$mean_rt[i]))
}
note <- c(note, "", "## Cooperation-task utility models (lme4)")
for (item in list(
  c("choice_util", "Choice (lme4::glmer)"),
  c("rt_util", "RT (lme4::lmer)"),
  c("p3_util", "P3 (lme4::lmer)"),
  c("lpp_util", "LPP (lme4::lmer)")
)) {
  note <- c(note, sprintf(
    "- %s: loss=%s; defect=%s; contrast p=%.4g",
    item[[2]], term_text(item[[1]], "zloss"), term_text(item[[1]], "zdefect"), contrast_p(item[[1]])
  ))
}
message("Fitting affective-rating ANOVA...")
ratings <- read_coop_ratings(file.path(root, "raw_data", "Rating.xlsx"))
ratings <- ratings[!ratings$Subject %in% invalid, ]
ratings <- ratings %>%
  pivot_longer(c(s2, s0, d2, d0), names_to = "type", values_to = "Rating") %>%
  mutate(
    FB = factor(ifelse(type %in% c("s2", "d2"), "good", "bad"), levels = c("bad", "good")),
    task = factor(ifelse(type %in% c("s2", "s0"), "single", "double"), levels = c("single", "double")),
    Subject = factor(Subject)
  )
m_rating <- aov(Rating ~ FB * task + Error(Subject / (FB * task)), data = ratings)
rating_anova <- rating_anova_table(m_rating)
rating_anova$partial_eta_sq <- NA_real_
for (i in which(!is.na(rating_anova$f))) {
  err <- rating_anova$stratum == rating_anova$stratum[i] & rating_anova$term == "Residuals"
  if (any(err)) {
    rating_anova$partial_eta_sq[i] <- rating_anova$sum_sq[i] / (rating_anova$sum_sq[i] + rating_anova$sum_sq[which(err)[1]])
  }
}
rating_means <- as.data.frame(summary(emmeans(m_rating, ~ FB | task), infer = c(TRUE, TRUE)))
rating_pairs <- as.data.frame(pairs(emmeans(m_rating, ~ task | FB)))
rating_d <- ratings %>%
  group_by(Subject, FB, task) %>%
  summarise(Rating = mean(Rating), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = task, values_from = Rating) %>%
  group_by(FB) %>%
  summarise(
    mean_diff_cooperation_minus_risk = mean(double - single),
    sd_diff = stats::sd(double - single),
    cohens_dz = mean_diff_cooperation_minus_risk / sd_diff,
    .groups = "drop"
  )
write.csv(rating_anova, file.path(behavior_dir, "affective_rating_anova.csv"), row.names = FALSE)
write.csv(rating_means, file.path(behavior_dir, "affective_rating_means.csv"), row.names = FALSE)
write.csv(rating_pairs, file.path(behavior_dir, "affective_rating_pairs.csv"), row.names = FALSE)
write.csv(rating_d, file.path(behavior_dir, "affective_rating_effect_sizes.csv"), row.names = FALSE)
rating_cells <- ratings %>%
  transmute(Subject, x = as.character(FB), task = as.character(task), y = Rating)

fb_row <- rating_anova[rating_anova$term == "FB" & !is.na(rating_anova$p), ]
task_row <- rating_anova[rating_anova$term == "task" & !is.na(rating_anova$p), ]
int_row <- rating_anova[rating_anova$term == "FB:task" & !is.na(rating_anova$p), ]
note <- c(
  note,
  "",
  "## Affective ratings",
  sprintf("- N subjects: %d", nlevels(ratings$Subject)),
  sprintf("- Outcome F=%.3f, p=%.4g, partial eta^2=%.3f", fb_row$f, fb_row$p, fb_row$partial_eta_sq),
  sprintf("- Task F=%.3f, p=%.4g, partial eta^2=%.3f", task_row$f, task_row$p, task_row$partial_eta_sq),
  sprintf("- Outcome x task F=%.3f, p=%.4g, partial eta^2=%.3f", int_row$f, int_row$p, int_row$partial_eta_sq)
)
writeLines(note, file.path(root, "results", "consistency_check.md"))
cat(paste(note, collapse = "\n"), "\n")

message("Drawing behavior and utility figures...")
choice_data <- dat
indiff_x <- 1 + (1 / 3 - 0.1) / 0.2
rt_panels <- behavior_panels(choice_data, "RT", c(400, 1600), "Decision time (ms)", vline = indiff_x)
choice_panels <- behavior_panels(choice_data, "Choice", c(0, 1), "P(Choice)", percent = TRUE)
rt_prob <- design_panel(m_rt_des, "zpro", list(zmon = 0), "Probability (Normalized)", "Decision time (ms)") +
  labs(tag = "C")
rt_stake <- design_panel(m_rt_des, "zmon", list(zpro = 0), "Principal (Normalized)", "Decision time (ms)")
choice_low <- design_panel(
  m_choice_des, "zpro", list(zmon = -1), "Probability (Normalized)", "P(Choice)",
  binomial = TRUE, percent = TRUE, title = "-1 SD principal"
) + labs(tag = "F")
choice_high <- design_panel(
  m_choice_des, "zpro", list(zmon = 1), "Probability (Normalized)", "P(Choice)",
  binomial = TRUE, percent = TRUE, title = "+1 SD principal"
)
save_plot(
  (rt_panels$probability + labs(tag = "A") | rt_panels$principal + labs(tag = "B") | (rt_prob | rt_stake)) +
    plot_layout(widths = c(1.05, 1.05, 1.45), guides = "collect") &
    theme(legend.position = "top"),
  file.path(main_dir, "fig_behavior_rt.png"), 13.4, 4.5
)
save_plot(
  (choice_panels$probability + labs(tag = "D") | choice_panels$principal + labs(tag = "E") | (choice_low | choice_high)) +
    plot_layout(widths = c(1.05, 1.05, 1.45), guides = "collect") &
    theme(legend.position = "top"),
  file.path(main_dir, "fig_behavior_choice.png"), 13.4, 4.5
)
save_plot(
  rating_figure(rating_cells),
  file.path(main_dir, "fig_affective_rating.png"), 5.2, 4.2
)
save_plot(
  accept_count_figure(choice_data),
  file.path(main_dir, "fig_n_accepted.png"), 5.4, 4.2
)

save_plot(
  utility_panel(m_choice, dou, "P(Choice)", binomial = TRUE, percent = TRUE, xlim = c(-2, 0), ylim = c(0, 1), open_frame = TRUE, legend = "inside_left") |
    utility_panel(m_rt, dou, "Decision time (ms)", xlim = c(-2, 0), ylim = c(700, 1050), open_frame = TRUE, legend = "none"),
  file.path(main_dir, "fig_utility_effects.png"), 9.2, 4.3
)
save_plot(
  utility_panel(m_p3, dou, "P3 Amplitude", xlim = NULL) |
    utility_panel(m_lpp, dou, "LPP Amplitude", xlim = NULL),
  file.path(main_dir, "fig_erp_effects.png"), 9.2, 4.3
)

message("Drawing parameter, comparison, recovery, and simulation figures...")
for (old_path in c(
  file.path(supp_dir, "fig_kappa.png"),
  file.path(supp_dir, "fig_m2_params.png"),
  file.path(supp_dir, "fig_ev_rt.png"),
  file.path(supp_dir, "fig_parameter_recovery_heatmap.png"),
  file.path(main_dir, "fig_parameter_recovery_scatter_M2.png")
)) {
  if (file.exists(old_path)) file.remove(old_path)
}
save_plot(
  manuscript_density(file.path(root, "results", "fits", "aversion_linear_model", "subject_params.csv")),
  file.path(main_dir, "fig_m2_params.png"), 5.2, 5.6
)
save_plot(
  manuscript_ic(file.path(root, "results", "model_comparison", "subject_ic.csv")),
  file.path(main_dir, "fig_model_ic.png"), 6.2, 4.4
)

heat_m2 <- read_matrix_csv(file.path(root, "results", "recovery", "parameter", "parameter_recovery_corr_heatmap_M2.csv"))
save_plot(manuscript_heat(heat_m2), file.path(main_dir, "fig_parameter_recovery_heatmap_M2.png"), 6.2, 5.2)

scatter_params <- list(
  M0 = "p",
  M1 = c("alpha_g", "alpha_l", "gamma", "lambda", "beta"),
  M2 = c("alpha_g", "alpha_l", "gamma", "lambda", "beta", "kappa")
)
read_recovery_points <- function(model_name) {
  points_path <- file.path(root, "results", "recovery", "parameter", paste0("parameter_recovery_points_", model_name, ".csv"))
  if (!file.exists(points_path)) stop("Missing ", points_path)
  read.csv(points_path)
}
points_m0 <- read_recovery_points("M0")
points_m1 <- read_recovery_points("M1")
points_m2 <- read_recovery_points("M2")
for (old_name in c(
  "fig_parameter_recovery_scatter_M0.png",
  "fig_parameter_recovery_scatter_M1.png",
  "fig_parameter_recovery_scatter_M2.png"
)) {
  old_path <- file.path(supp_dir, old_name)
  if (file.exists(old_path)) file.remove(old_path)
}
save_plot(
  recovery_scatter_figure(list(
    list(
      points = bind_rows(points_m0, points_m1),
      parameters = c(scatter_params$M0, scatter_params$M1),
      groups = list(1, 2:6),
      labels = c("M0", "M1")
    ),
    list(
      points = points_m2,
      parameters = scatter_params$M2,
      groups = list(1:6),
      labels = "M2"
    )
  )),
  file.path(supp_dir, "fig_parameter_recovery_scatter.png"),
  14.6, 6.8
)

conf <- read.csv(file.path(root, "results", "recovery", "model", "model_recovery_confusion.csv"), check.names = FALSE)
conf_long <- conf %>%
  pivot_longer(-true_short, names_to = "selected", values_to = "n") %>%
  group_by(true_short) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ungroup()
conf_long$true_short <- factor(conf_long$true_short, levels = c("M0", "M1", "M2"))
conf_long$selected <- factor(conf_long$selected, levels = c("M0", "M1", "M2"))
p_conf <- ggplot(conf_long, aes(selected, true_short, fill = pct)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", pct)), size = 4) +
  scale_fill_gradient(low = "#F7FBFF", high = "#08306B", limits = c(0, 100), name = "%") +
  labs(x = "Selected model", y = "True model", title = "Model recovery") +
  coord_fixed() +
  theme_heat()
save_plot(p_conf, file.path(supp_dir, "fig_model_recovery_confusion.png"), 5.6, 4.8)

sim_prob_path <- file.path(root, "results", "simulation", "simulation_by_probability.csv")
sim_stake_path <- file.path(root, "results", "simulation", "simulation_by_stake.csv")
if (file.exists(sim_prob_path) && file.exists(sim_stake_path)) {
  sim_prob <- read.csv(sim_prob_path)
  sim_stake <- read.csv(sim_stake_path)
  for (old_name in c(
    "fig_simulation_by_probability.png",
    "fig_simulation_by_stake.png",
    "fig_simulation_panels.png"
  )) {
    old_path <- file.path(supp_dir, old_name)
    if (file.exists(old_path)) file.remove(old_path)
  }
  save_plot(
    simulation_figure(sim_prob, sim_stake),
    file.path(supp_dir, "fig_simulation.png"), 8.4, 7.4
  )
}

message("Figures written to ", main_dir, " and ", supp_dir)
