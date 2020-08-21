library(feather)
library(tidyverse)
# library(data.table)
library(brms)
# library(bayesplot)
# library(tidybayes)

source("../../models/stan_utility.r")


args <- commandArgs(trailingOnly = TRUE)
load_dir <- args[1]
print(load_dir)
# Call like `Rscript smod.r ../../data/cage/suite_data`
setwd(load_dir)

# setwd("/Users/wbeard/repos/perf/data/interim/")


thin <- function(df, n=4) {
  df[seq(1, nrow(df), n), ]
}

get_pis <- function(mod, all.days) {
  pi1 <- predictive_interval(mod, newdata=all.days, prob = .9)
  pi2 <- predictive_interval(mod, newdata=all.days, prob = .7)
  pi3 <- predictive_interval(mod, newdata=all.days, prob = .5)
  pi4 <- predictive_interval(mod, newdata=all.days, prob = .68)
  pi5 <- predictive_interval(mod, newdata=all.days, prob = .01)
  cbind(pi1, pi2, pi3, pi4, pi5)
}

models


mod.mod <- function(models) {
  print(exists('a', models))
  print(exists('b', models))
  models[['a']] <<- 1
  1
}
mod.mod(models)
models

models <- list()

get.formula <- function(df) {
  get.knots <- function(df) {
    ceiling(max(df$dayi) / 8)
  }
  
  k <- get.knots(df)
  print(paste0('pulling model with ', k, ' knots'))
  key <- paste0('k', k)
  if (!is_null(models[[key]])){
    return(models[[key]])
  }
  
  print(k)
  print('going...')
  form_s <- glue("y ~ s(dayi, k={k}, bs='cr')")
  f <- bf(form_s)
  
  models[[key]] <<- brm(f, df, chains = 0)
  models[[key]]
}

mod <- get.formula(df0)

library(glue)
k = 42
form_s <- glue("y ~ s(dayi, k={k}, bs='cr')")
bf(form_s)

get.model2 <- function(df, k.i = 42) {
  # f <- with(list(k=42), bf(y ~ s(dayi, k=k, bs='cr')))
  f <-bf(y ~ s(dayi, k=k.i, bs='cr'))
  brm(f, df, chains = 0)
}
get.model2(df0)

read.df <- function(pth) {
  df = read_feather(pth) %>% as.data.frame()
  df = df[!df$out,]
   # %>% subset(select = c("dayi"))
  df
}

# f = bf(y ~ s(dayi, k=28, bs='cr'))
# , sigma ~ rstd
f = bf(y ~ s(dayi, k=28, bs='cr'))

fn <- function(df0) {
  k = 28
  f = bf(y ~ s(dayi, k=k, bs='cr'))
  fit_empty <- brm(f, df0, chains = 0)
}
k = 28
f = bf(y ~ s(dayi, k=k, bs='cr'))

# files <- Sys.glob("suite_data/ts*")
files <- Sys.glob("br/ts*")
df0 <- read.df(files[1])

fit_empty <- brm(f, df0, chains = 0)

for (filename in files) {
  print(filename)
  df <- read.df(filename)
  print(max(df$dayi))
  # k <- get.knots(df)
  # all.days <- data.frame(dayi = seq(min(df$dayi), to = max(df$dayi)))
  # 
  # 
  # {
  #   sink("/dev/null");
  #   model = update(fit_empty, newdata=df, recompile = FALSE, cores = 4);
  #   sink();
  # }
  # 
  # check_all_diagnostics(model$fit)
  # pi <- get_pis(model, all.days)
  # draws <- brms::posterior_predict(model, newdata = all.days)
  # draws <- thin(draws)
  # 
  # fn2 = paste0('brpi/', basename(filename))
  # fn.draws = paste0('br_draws/', basename(filename))
  # print("=>")
  # print(fn2)
  # write_feather(data.frame(pi), fn2)
  # write_feather(data.frame(draws), fn.draws)

}




# fit1 = update(fit_empty, newdata=df, recompile = FALSE, cores = 4)
# fit2 = update(fit_empty, newdata=df, recompile = FALSE, cores = 4)

# marginal_effects(fit1)

# Sys.sleep(30)
