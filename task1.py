from kmeans import kmeans
import pandas as pd

# Read in the data set
X = pd.read_csv('data.csv', header=None)
Y = pd.read_csv('label.csv', header=None)


alg = kmeans()

#alg.run(X, Y, dist='cosine')
alg.run(X, Y, dist='jaccard')
#alg.run(X, Y, dist='euclidean')
