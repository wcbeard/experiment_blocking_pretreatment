library(feather)
library(tidyverse)
library(data.table)
library(partykit)

airq <- airquality
airq <- subset(airquality, !is.na(Ozone))
airct <- ctree(Ozone ~ ., data = airq,
               control = ctree_control(maxsurrogate = 3))
airct

df = read_feather('/Users/wbeard/repos/crash/data/interim/soc_msg.fth') %>% as.data.frame()
df = read_feather('/Users/wbeard/repos/crash/data/interim/soc_msg.fth') %>% as.data.frame()
ct <- ctree(missing ~ ., data = df, control=ctree_control(maxdepth=4))
pl(ct)
ct
plot(ct)

pl <- function(ct) {
  pdf(file = "/Users/wbeard/repos/crash/reports/figures/soc_cr_ctree.pdf",
      width = 18, # The width of the plot in inches
      # height = 4
      )
  plot(ct)
  dev.off()
}

df <- df[, !c("btime", "date", "build_id", )]

head(df)
typeof(df)
sapply(df,class)
df
