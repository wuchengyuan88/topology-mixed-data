Order of running code:
1) dataCleaning_v2.py

Performs cleaning of data, including standardization and symmetry breaking.

2) pointCloud_v2.py

Generates point clouds (as .csv files) in a folder named “pointclouds”.

3) TDA_distmatrix_v2.R

Computes the distance matrix (Wasserstein distance) between point clouds, and stores it in a .csv file.

4) knn_v2.py

Calculates the validation and test accuracies and other metrics.


Optional:

- 10FoldCV_v2.py

Calculates the 10-fold cross-validation accuracy and other metrics.

- “misc” folder contains other optional code for example, to draw persistence diagrams.

