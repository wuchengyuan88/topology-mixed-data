# loading R package TDA
library(package="TDA")
df <- read.csv('processed.cleveland.dataclean_standardized.csv')
totaln <- nrow(df)
print(totaln)
#297

temp1 <- read.csv(paste0('./pointclouds/pc',296,'.csv'),header = FALSE)
mat1 <- as.matrix(temp1) 
Diag1 <- ripsDiag(X=mat1, maxdimension=1, maxscale=40,
                  dist="euclidean")
plot(Diag1[["diagram"]])