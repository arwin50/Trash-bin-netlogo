library(tidyverse)
library(FSA)      # Dunn post-hoc test
library(rstatix)  # effect sizes
library(coin)     # Kruskal-Wallis with exact p-values

df <- read_csv("fuente_dataset.csv")
df <- df %>% filter(`arrival-rate` > 0, ticks == 7500)

# Kruskal-Wallis for each factor
kruskal.test(total-litter-events ~ factor(`bin-placement`), data=df)

# Post-hoc Dunn test — tells you which bin levels differ
dunnTest(total-litter-events ~ factor(`bin-placement`), 
         data=df, method="bonferroni")

# Effect size
df %>% kruskal_effsize(total-litter-events ~ `bin-placement`)


