from scipy.spatial.distance import cosine, euclidean
from itertools import permutations
import random, sys
import numpy as np
import pandas as pd

def jaccards(X, Y):
    top, bot = sum([min(x,y) for x,y in zip(X,Y)]),\
                sum([max(x,y) for x,y in zip(X,Y)])

    if top == 0 and bot == 0:
        return 1
    elif top != 0 and bot == 0:
        return 0
    else:
        return 1 - top/bot

class kmeans():

    def run(self, df, true_y, k=None, iters=100, dist='euclidean', \
                                stop_if_SSE_inc=True):

        # Save k
        k = k if k is not None else len(true_y[true_y.columns[0]].unique())
        self.k = k

        # Get the labels as a simple list
        true_y = list(true_y[0])
        self.true_y = true_y

        # Remove any columns with only 1 unique value
        df = df[[col for col in df.columns if df[col].unique()[0] > 1]]
        self.df = df

        # Find the minmax per columns
        minmax = {col:(df[col].min(), df[col].max()) for col in df.columns}

        # Initialize with random means btwn min and max
        self.centroids = [[random.uniform(minmax[col][0]+1, minmax[col][1]-1) \
                            for col in df.columns] for cluster in range(k)]

        # Mem management, delete minmax (no longer needed)
        del minmax

        # Get the correct distance method
        if dist == 'euclidean':
            self.dist = euclidean
        elif dist == 'cosine':
            self.dist = cosine
        elif dist == 'jaccard':
            self.dist = jaccards
        elif callable(dist):
            self.dist = dist
        else:
            raise ValueError('Expected a string for dist of valid name')

        SSE = sys.float_info.max

        A, B = None, None
        for run in range(iters):

            # Classify each point
            Y = df.apply(self.classify_point, axis=1, result_type='expand')

            # Calculate the square error
            last_SSE = SSE                          # Saves the old SSE
            SSE = (Y[Y.columns[1]] ** 2).sum()      # Calculates new SSE

            if stop_if_SSE_inc and SSE > last_SSE:
                print('Terminated early as SSE increased')
                return

            # Figure out the best label per cluster
            lbl_tbl = {k2:[0]*k for k2 in range(k)}

            # Figure out matches
            for t_y, init_y in zip(true_y, Y[Y.columns[0]]):
                lbl_tbl[init_y][t_y] += 1

            # Returns accuracy of predicted labels to clusters
            def score_acc(perm):
                return sum([lbl_tbl[init_y][t_y] \
                                        for init_y, t_y in enumerate(perm)])

            possible_lbls = list(permutations(range(k)))

            # Find the best permutation of labels that leads to the highest starting
            #   accuracy, so now cluster number leads to a label
            lbls = max(possible_lbls, key=score_acc)
            del possible_lbls

            # Calculate the accuracy
            acc = score_acc(lbls)/len(Y)

            # Update the centroids
            new_centroids = self.update_means(Y)

            same = True
            for new_c, old_c in zip(new_centroids, self.centroids):
                if len(new_c) != len(old_c):
                    raise ValueError('Centroid changed length in update')
                for new, old in zip(new_c, old_c):
                    if new != old:
                        same = False
                        break

            if same:
                print('Terminated early as clusters are the same')
                return

            self.centroids = new_centroids

            print(f'{run}: ACC: {round(acc,4)*100}% | SSE: {SSE}')


    # Returns new means of clusters
    def update_means(self, Y):

        # Get the data frame copy
        df = self.df.copy()


        # New means
        means = [[0 for col in df.columns] for cluster in range(self.k)]

        # Add labels
        df['Y'] = Y[Y.columns[0]]

        for name, group in df.groupby('Y'):
            group.pop('Y')
            means[int(name)] = list(group.mean())

        df.pop('Y')

        return means

    # Classify a point
    def classify_point(self, pt_data):
        # Get the distance method
        dist = self.dist

        # Create variables to track best cluster/distance
        best_cluster, best_dist = None, None

        # Go through each cluster and means and then see which closest to
        for cluster, centroid in enumerate(self.centroids):
            # Calculate cur dist
            cur_dist = dist(pt_data, centroid)

            if best_dist is None: # If nothing, save as best
                best_cluster, best_dist = cluster, cur_dist
            elif cur_dist < best_dist: # If better, save as best
                best_cluster, best_dist = cluster, cur_dist

        # Return best
        return (best_cluster, best_dist)
