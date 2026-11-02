#!/usr/bin/env Rscript
# Statistical analysis and figures.
# Mixed models follow scripts_renew.R. Figures show subject means,
# within-subject intervals, and population prediction curves.
# Trial utilities are read from results/derived, written by fitting.py.
# Model-comparison, recovery, and simulation tables are read from results/.

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
out_dir <- file.path(root, "results", "downstream")
main_dir <- file.path(root, "figures", "main_figures")
supp_dir <- file.path(root, "figures", "supp_figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(main_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(supp_dir, recursive = TRUE, showWarnings = FALSE)

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
      ci = sd(normed) / sqrt(dplyr::n()) * morey * crit,
      .groups = "drop"
    )
}

behavior_panel <- function(cells, summary_df, xlab, breaks, labels, ylim, ylab) {
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
  ggplot() +
    geom_point(
      data = points,
      aes(x_point, y, colour = task_label),
      position = position_jitter(width = 0.05, height = 0, seed = 1),
      size = 2.2, alpha = 0.35
    ) +
    geom_errorbar(
      data = means,
      aes(x = x_mean, ymin = y - ci, ymax = y + ci, colour = task_label),
      width = 0.28, linewidth = 0.6, position = position_dodge(width = 0.35)
    ) +
    geom_line(
      data = means,
      aes(x = x_mean, y = y, colour = task_label, group = task_label),
      linewidth = 0.7, position = position_dodge(width = 0.35)
    ) +
    geom_point(
      data = means,
      aes(x = x_mean, y = y, colour = task_label),
      shape = 15, size = 2.4, position = position_dodge(width = 0.35)
    ) +
    scale_colour_manual(values = task_colors, name = NULL) +
    scale_x_continuous(breaks = breaks, labels = labels) +
    coord_cartesian(ylim = ylim) +
    labs(x = xlab, y = ylab) +
    theme_behavior()
}

behavior_figure <- function(data, y, ylim, ylab) {
  prob_cells <- subject_cells(data, y, "pro")
  stake_cells <- subject_cells(data, y, "montype")
  p_prob <- behavior_panel(
    prob_cells, within_summary(prob_cells),
    "Instructed probability", 1:5, prob_labels, ylim, ylab
  )
  p_stake <- behavior_panel(
    stake_cells, within_summary(stake_cells),
    "Principal", 1:5, stake_labels, ylim, ylab
  )
  (p_prob | p_stake) + plot_layout(guides = "collect") & theme(legend.position = "top")
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
    p = 2 * pnorm(-abs(z)),
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
  ggplot(means, aes(task_label, y, colour = fb_label, fill = fb_label, group = fb_label)) +
    geom_errorbar(
      aes(ymin = y - ci, ymax = y + ci),
      width = 0.12, linewidth = 0.6, colour = "black",
      position = position_dodge(width = 0.25)
    ) +
    geom_line(position = position_dodge(width = 0.25), linewidth = 0.7) +
    geom_point(shape = 22, size = 3.2, colour = "black", position = position_dodge(width = 0.25)) +
    scale_colour_manual(values = fb_colors, name = NULL) +
    scale_fill_manual(values = fb_colors, name = NULL) +
    scale_y_continuous(breaks = seq(2.5, 8.5, 1)) +
    coord_cartesian(ylim = c(2.5, 8.5)) +
    labs(x = "Task", y = "Valence rating") +
    theme_behavior()
}

ev_rt_figure <- function(model) {
  grid <- expand.grid(
    z_ev_diff = seq(-2.5, 2.5, by = 0.05),
    task = factor(c("single", "double"), levels = c("single", "double"))
  )
  pr <- predict(model, newdata = grid, re.form = NA, se.fit = TRUE)
  grid$fit <- as.numeric(pr$fit)
  grid$se <- as.numeric(pr$se.fit)
  grid$task_label <- label_task(grid$task)
  ggplot(grid, aes(z_ev_diff, fit, colour = task_label, fill = task_label)) +
    geom_ribbon(aes(ymin = fit - se, ymax = fit + se), alpha = 0.20, colour = NA) +
    geom_line(linewidth = 1) +
    scale_colour_manual(values = task_colors, name = NULL) +
    scale_fill_manual(values = task_colors, name = NULL) +
    scale_x_continuous(breaks = seq(-2, 2, 1)) +
    labs(x = "\u0394EV (normalized)", y = "Decision time (ms)") +
    theme_behavior()
}

utility_panel <- function(model, data, ylab, title, binomial = FALSE, percent = FALSE) {
  loss <- pop_curve(model, data, "zloss", "zdefect", binomial)
  defect <- pop_curve(model, data, "zdefect", "zloss", binomial)
  loss$predictor <- "Loss"
  defect$predictor <- "Betrayal"
  curve <- bind_rows(loss, defect)
  curve$predictor <- factor(curve$predictor, levels = c("Loss", "Betrayal"))
  p <- ggplot(curve, aes(x, fit, colour = predictor, fill = predictor)) +
    geom_ribbon(aes(ymin = fit - se, ymax = fit + se), alpha = 0.20, colour = NA) +
    geom_line(linewidth = 1) +
    geom_vline(xintercept = 0, linetype = "dotted", linewidth = 0.5) +
    scale_colour_manual(values = util_colors) +
    scale_fill_manual(values = util_colors) +
    labs(x = "Scaled model value", y = ylab, title = title) +
    theme_utility()
  if (percent) {
    p <- p + scale_y_continuous(labels = scales::percent, name = ylab)
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

heat_plot <- function(mat, title) {
  df <- as.data.frame(as.table(mat), stringsAsFactors = FALSE)
  names(df) <- c("true_param", "recovered", "r")
  df$true_param <- factor(df$true_param, levels = rownames(mat))
  df$recovered <- factor(df$recovered, levels = colnames(mat))
  ggplot(df, aes(recovered, true_param, fill = r)) +
    geom_tile(colour = "white", linewidth = 0.4) +
    geom_text(aes(label = sprintf("%.2f", r)), size = 3) +
    scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0, limits = c(-0.25, 1), name = "r") +
    labs(x = "Recovered", y = "Simulated (true)", title = title) +
    coord_fixed() +
    theme_heat() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
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
message("Fitting expected-value RT lmer...")
m_rt_ev <- lmer(
  RT ~ (z_ev_diff + I(z_ev_diff^2)) * task + (task | Subject),
  data = dat, REML = FALSE
)
message("Fitting expected-value P3 and LPP lmers...")
m_p3_ev <- lmer(
  MonP3 ~ (z_ev_diff + I(z_ev_diff^2)) * task + (task | Subject),
  data = dat, REML = FALSE, control = ctrl_lmer
)
m_lpp_ev <- lmer(
  MonLPP ~ (z_ev_diff + I(z_ev_diff^2)) * task + (task | Subject),
  data = dat, REML = FALSE, control = ctrl_lmer
)
message("Fitting ERP interaction and design lmers...")
m_p3_int <- lmer(
  MonP3 ~ zloss * zdefect + (zloss | Subject),
  data = dou, REML = FALSE, control = ctrl_lmer
)
m_lpp_int <- lmer(
  MonLPP ~ zloss * zdefect + (zloss | Subject),
  data = dou, REML = FALSE, control = ctrl_lmer
)
m_p3_des <- lmer(MonP3 ~ zpro * zmon * task + (task | Subject), data = dat, REML = FALSE)
m_lpp_des <- lmer(MonLPP ~ zpro * zmon * task + (task | Subject), data = dat, REML = FALSE)
message("Fitting joint P3/LPP lmer...")
erp_long <- dou %>%
  select(Subject, MonP3, MonLPP, zdefect, zloss) %>%
  pivot_longer(c(MonP3, MonLPP), names_to = "Component", values_to = "Amplitude") %>%
  mutate(Component = factor(Component, levels = c("MonP3", "MonLPP")))
contrasts(erp_long$Component) <- contr.sdif(2)
m_erp <- lmer(
  Amplitude ~ Component * (zdefect + zloss) + (zloss | Subject),
  data = erp_long, REML = FALSE
)

fe <- rbind(
  fe_table(m_choice_des, "choice_design", "lme4::glmer", "(zpro + task | Subject)"),
  fe_table(m_rt_des, "rt_design", "lme4::lmer", "(task | Subject)"),
  fe_table(m_choice, "choice_util", "lme4::glmer", "(zloss + zdefect | Subject)"),
  fe_table(m_rt, "rt_util", "lme4::lmer", "(zdefect | Subject)"),
  fe_table(m_p3, "p3_util", "lme4::lmer", "(zloss | Subject)"),
  fe_table(m_lpp, "lpp_util", "lme4::lmer", "(zloss | Subject)"),
  fe_table(m_rt_ev, "rt_ev", "lme4::lmer", "(task | Subject)"),
  fe_table(m_p3_ev, "p3_ev", "lme4::lmer", "(task | Subject)"),
  fe_table(m_lpp_ev, "lpp_ev", "lme4::lmer", "(task | Subject)"),
  fe_table(m_p3_int, "p3_interaction", "lme4::lmer", "(zloss | Subject)"),
  fe_table(m_lpp_int, "lpp_interaction", "lme4::lmer", "(zloss | Subject)"),
  fe_table(m_p3_des, "p3_design", "lme4::lmer", "(task | Subject)"),
  fe_table(m_lpp_des, "lpp_design", "lme4::lmer", "(task | Subject)"),
  fe_table(m_erp, "erp_joint", "lme4::lmer", "(zloss | Subject)")
)
contrasts_out <- rbind(
  paper_contrast(m_choice, "choice_util"),
  paper_contrast(m_rt, "rt_util"),
  paper_contrast(m_p3, "p3_util"),
  paper_contrast(m_lpp, "lpp_util"),
  paper_contrast(m_lpp, "lpp_util_abs", use_abs = TRUE)
)
write.csv(fe, file.path(out_dir, "fixed_effects.csv"), row.names = FALSE)
write.csv(contrasts_out, file.path(out_dir, "loss_vs_defect_contrasts.csv"), row.names = FALSE)

desc <- aggregate(cbind(Choice, RT) ~ task, data = dat, FUN = mean, na.rm = TRUE)
names(desc) <- c("task", "accept_rate", "mean_rt")
write.csv(desc, file.path(out_dir, "task_descriptives.csv"), row.names = FALSE)

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
write.csv(slopes_choice, file.path(out_dir, "choice_probability_slopes.csv"), row.names = FALSE)
write.csv(slopes_rt, file.path(out_dir, "rt_probability_slopes.csv"), row.names = FALSE)

task_prob <- rbind(
  wald_combo(m_choice_des, c("zpro:task2-1" = 1, "zpro:zmon:task2-1" = 1), "choice_design", "task_difference_at_zmon_1"),
  wald_combo(m_choice_des, c("zpro:task2-1" = 1, "zpro:zmon:task2-1" = -1), "choice_design", "task_difference_at_zmon_-1")
)
write.csv(task_prob, file.path(out_dir, "choice_task_probability_contrasts.csv"), row.names = FALSE)

ev_trends <- as.data.frame(summary(
  emtrends(m_rt_ev, revpairwise ~ task, var = "z_ev_diff"),
  infer = c(TRUE, TRUE)
))
write.csv(ev_trends, file.path(out_dir, "rt_ev_trends.csv"), row.names = FALSE)
write.csv(
  as.data.frame(summary(emtrends(m_lpp_int, revpairwise ~ zloss, var = "zdefect"), infer = c(TRUE, TRUE))),
  file.path(out_dir, "lpp_interaction_revpairwise.csv"), row.names = FALSE
)
write.csv(
  slope_table(m_lpp_int, ~ zdefect, "zloss", list(zdefect = c(-1, 1)), "lpp_interaction", "at_zdefect"),
  file.path(out_dir, "lpp_interaction_at_zdefect.csv"), row.names = FALSE
)
write.csv(
  slope_table(m_lpp_int, ~ zloss, "zdefect", list(zloss = c(-1, 1)), "lpp_interaction", "at_zloss"),
  file.path(out_dir, "lpp_interaction_at_zloss.csv"), row.names = FALSE
)
erp_trends <- as.data.frame(summary(
  emtrends(m_erp, revpairwise ~ Component, var = "zdefect"),
  infer = c(TRUE, TRUE)
))
write.csv(erp_trends, file.path(out_dir, "erp_joint_trends.csv"), row.names = FALSE)
ic_ev <- data.frame(
  model = c("rt_design", "rt_ev"),
  aic = AIC(m_rt_des, m_rt_ev)$AIC,
  bic = BIC(m_rt_des, m_rt_ev)$BIC
)
write.csv(ic_ev, file.path(out_dir, "rt_ev_comparison.csv"), row.names = FALSE)

sink(file.path(out_dir, "lme4_model_summaries.txt"))
cat("=== choice_design ===\n"); print(summary(m_choice_des))
cat("\n=== rt_design ===\n"); print(summary(m_rt_des))
cat("\n=== choice_util ===\n"); print(summary(m_choice))
cat("\n=== rt_util ===\n"); print(summary(m_rt))
cat("\n=== p3_util ===\n"); print(summary(m_p3))
cat("\n=== lpp_util ===\n"); print(summary(m_lpp))
cat("\n=== rt_ev ===\n"); print(summary(m_rt_ev))
cat("\n=== p3_ev ===\n"); print(summary(m_p3_ev))
cat("\n=== lpp_ev ===\n"); print(summary(m_lpp_ev))
cat("\n=== p3_interaction ===\n"); print(summary(m_p3_int))
cat("\n=== lpp_interaction ===\n"); print(summary(m_lpp_int))
cat("\n=== p3_design ===\n"); print(summary(m_p3_des))
cat("\n=== lpp_design ===\n"); print(summary(m_lpp_des))
cat("\n=== erp_joint ===\n"); print(summary(m_erp))
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
  "# Downstream mixed models following scripts_renew.R",
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
rating_means <- as.data.frame(summary(emmeans(m_rating, ~ FB | task), infer = c(TRUE, TRUE)))
write.csv(rating_anova, file.path(out_dir, "affective_rating_anova.csv"), row.names = FALSE)
write.csv(rating_means, file.path(out_dir, "affective_rating_means.csv"), row.names = FALSE)
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
  sprintf("- Outcome F=%.3f, p=%.4g", fb_row$f, fb_row$p),
  sprintf("- Task F=%.3f, p=%.4g", task_row$f, task_row$p),
  sprintf("- Outcome x task F=%.3f, p=%.4g", int_row$f, int_row$p)
)
writeLines(note, file.path(out_dir, "consistency_check.md"))
cat(paste(note, collapse = "\n"), "\n")

message("Drawing behavior and utility figures...")
choice_data <- dat
save_plot(
  behavior_figure(choice_data, "Choice", c(0, 1), "P(accept)"),
  file.path(main_dir, "fig_behavior_choice.png"), 10, 4.4
)
save_plot(
  behavior_figure(choice_data, "RT", c(400, 1600), "Decision time (ms)"),
  file.path(main_dir, "fig_behavior_rt.png"), 10, 4.4
)
save_plot(
  rating_figure(rating_cells),
  file.path(main_dir, "fig_affective_rating.png"), 5.2, 4.2
)

util_fig <- (
  utility_panel(m_choice, dou, "P(accept)", "Choice", binomial = TRUE, percent = TRUE) |
    utility_panel(m_rt, dou, "Decision time (ms)", "Decision time")
) / (
  utility_panel(m_p3, dou, "P3 amplitude", "P3") |
    utility_panel(m_lpp, dou, "LPP amplitude", "LPP")
) + plot_layout(guides = "collect") & theme(legend.position = "top")
save_plot(util_fig, file.path(main_dir, "fig_utility_effects.png"), 9.2, 7.4)
save_plot(ev_rt_figure(m_rt_ev), file.path(supp_dir, "fig_ev_rt.png"), 6.4, 4.2)

message("Drawing parameter, comparison, recovery, and simulation figures...")
params <- read.csv(file.path(root, "results", "fits", "aversion_linear_model", "subject_params_wide.csv"))
param_long <- params %>%
  select(subject_id, param_alpha_g, param_alpha_l, param_gamma, param_lambda, param_beta, param_kappa) %>%
  pivot_longer(-subject_id, names_to = "parameter", values_to = "value") %>%
  mutate(parameter = sub("^param_", "", parameter))
param_long$parameter <- factor(param_long$parameter, levels = c("alpha_g", "alpha_l", "gamma", "lambda", "beta", "kappa"))
p_params <- ggplot(param_long, aes(value)) +
  geom_histogram(aes(y = after_stat(density)), bins = 15, fill = "#4C78A8", colour = "white") +
  geom_density(linewidth = 0.6) +
  facet_wrap(~ parameter, scales = "free", ncol = 3, labeller = param_labeller) +
  labs(x = NULL, y = "Density", title = "M2 parameter distributions") +
  theme_behavior() +
  theme(legend.position = "none", strip.text = element_text(size = 11))
save_plot(p_params, file.path(supp_dir, "fig_m2_params.png"), 9, 5.6)

p_kappa <- ggplot(params, aes(param_kappa)) +
  geom_histogram(bins = 15, fill = "#EB6B29", colour = "white") +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.6) +
  labs(x = "kappa", y = "Count", title = "Betrayal cost") +
  theme_behavior() + theme(legend.position = "none")
save_plot(p_kappa, file.path(supp_dir, "fig_kappa.png"), 5, 3.6)

comp <- read.csv(file.path(root, "results", "model_comparison", "model_comparison.csv"))
comp$short_name <- factor(comp$short_name, levels = c("M0", "M1", "M2"))
ic_long <- comp %>%
  select(short_name, mean_aic, mean_bic, mean_looic) %>%
  pivot_longer(-short_name, names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric, mean_aic = "AIC", mean_bic = "BIC", mean_looic = "CVIC"))
win_long <- comp %>%
  select(short_name, n_best_aic, n_best_bic, n_best_looic) %>%
  pivot_longer(-short_name, names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric, n_best_aic = "AIC", n_best_bic = "BIC", n_best_looic = "CVIC"))
ic_long$metric <- factor(ic_long$metric, levels = c("AIC", "BIC", "CVIC"))
win_long$metric <- factor(win_long$metric, levels = c("AIC", "BIC", "CVIC"))
p_ic <- ggplot(ic_long, aes(short_name, value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  scale_fill_manual(values = c(AIC = "#4C78A8", BIC = "#F58518", CVIC = "#54A24B"), name = NULL) +
  labs(x = NULL, y = "Mean information criterion", title = "Model comparison") +
  theme_behavior()
p_win <- ggplot(win_long, aes(short_name, value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  scale_fill_manual(values = c(AIC = "#4C78A8", BIC = "#F58518", CVIC = "#54A24B"), name = NULL) +
  labs(x = NULL, y = "N subjects best", title = "Subject-level selection") +
  theme_behavior()
save_plot(
  (p_ic | p_win) + plot_layout(guides = "collect") & theme(legend.position = "top"),
  file.path(main_dir, "fig_model_ic.png"), 9.5, 4.2
)

heat_m2 <- read_matrix_csv(file.path(root, "results", "recovery", "parameter", "parameter_recovery_corr_heatmap_M2.csv"))
save_plot(heat_plot(heat_m2, "Parameter recovery (M2)"), file.path(main_dir, "fig_parameter_recovery_heatmap_M2.png"), 6.2, 5.2)

heat_m1 <- read_matrix_csv(file.path(root, "results", "recovery", "parameter", "parameter_recovery_corr_heatmap_M1.csv"))
save_plot(
  heat_plot(heat_m1, "M1") | heat_plot(heat_m2, "M2"),
  file.path(supp_dir, "fig_parameter_recovery_heatmap.png"), 11, 5.2
)

scatter_params <- list(
  M0 = "p",
  M1 = c("alpha_g", "alpha_l", "gamma", "lambda", "beta"),
  M2 = c("alpha_g", "alpha_l", "gamma", "lambda", "beta", "kappa")
)
points_m2 <- read.csv(file.path(root, "results", "recovery", "parameter", "parameter_recovery_points_M2.csv"))
points_m2$parameter <- factor(points_m2$parameter, levels = scatter_params$M2)
p_scatter <- ggplot(points_m2, aes(true_value, estimated_value)) +
  geom_point(size = 1.1, alpha = 0.45) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.4) +
  facet_wrap(~ parameter, scales = "free", ncol = 3, labeller = param_labeller) +
  labs(x = "True", y = "Recovered", title = "Parameter recovery (M2)") +
  theme_behavior() + theme(legend.position = "none", strip.text = element_text(size = 11))
save_plot(p_scatter, file.path(main_dir, "fig_parameter_recovery_scatter_M2.png"), 9, 5.8)

read_recovery_points <- function(model_name) {
  points_path <- file.path(root, "results", "recovery", "parameter", paste0("parameter_recovery_points_", model_name, ".csv"))
  if (!file.exists(points_path)) stop("Missing ", points_path)
  read.csv(points_path)
}
points_m0 <- read_recovery_points("M0")
points_m1 <- read_recovery_points("M1")
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
