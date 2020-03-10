# loading R package TDA
library(package="TDA")
library(tictoc) #measuring time
tic("TDA_distmatrix")
df <- read.csv('processed.cleveland.dataclean_standardized.csv')
totaln <- nrow(df)
print(totaln)
#297

#60% train, 20% validation, 20% validation
trainnum <- 179
validnum <- 59
testnum <- 59
print(trainnum+validnum+testnum)

#distance matrices for dimensions 0 and 1
distmatrix0 <- matrix(0,nrow = totaln, ncol = totaln)
#distmatrix1 <- matrix(0,nrow = totaln, ncol = totaln)

startindex<-trainnum+1

#temp <- read.csv(paste0('./pointclouds/pc',0,'.csv'),header = FALSE)
#temp1 <- as.matrix(temp) 


for (i in 1:totaln){
  #full range: for (j in 1:trainnum){
  for (j in 1:i){
    print(i)
    #print(j)
    temp1 <- read.csv(paste0('./pointclouds/pc',i-1,'.csv'),header = FALSE)
    mat1 <- as.matrix(temp1) 
    Diag1 <- ripsDiag(X=mat1, maxdimension=1, maxscale=100,
                      dist="euclidean")

    temp2 <- read.csv(paste0('./pointclouds/pc',j-1,'.csv'),header = FALSE)
    mat2 <- as.matrix(temp2)
    Diag2 <- ripsDiag(X=mat2, maxdimension=1, maxscale=100,
                      dist="euclidean")

dist0 <- wasserstein(Diag1=Diag1[["diagram"]],Diag2=Diag2[["diagram"]],
                   dimension=0)

#dist1 <- wasserstein(Diag1=Diag1[["diagram"]],Diag2=Diag2[["diagram"]],dimension=1)

distmatrix0[i,j] <- dist0
#distmatrix1[i,j] <- dist1
}
}

#distmatrixtotal = distmatrix0+distmatrix1

write.csv(distmatrix0,"WassersteinDistMatrix0.csv")
#write.csv(distmatrix1,"WassersteinDistMatrix1.csv")
#write.csv(distmatrixtotal,"WassersteinDistMatrixTotal.csv")

###End of code
toc()
#TDA_distmatrix: 266.728 sec elapsed