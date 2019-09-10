import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import time


data = pd.read_csv("creditcard.csv")
X = data.drop(['Class'], axis=1)
y = data['Class']

clusters = np.linspace(2, 25, 25, endpoint=True)
clustersMSE = []
clustersTime = []
for cluster in clusters:
    before = time.time()
    clusterer = KMeans(n_clusters=int(cluster)).fit(X)
    print("cluster centers ", clusterer.cluster_centers_)
    print("inertia ", clusterer.inertia_)
    clustersMSE.append(clusterer.inertia_)
    clustersTime.append(time.time() - before)

print(clustersMSE)


from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(clusters, clustersMSE, 'b')
plt.ylabel('Mean Squared Error')
plt.xlabel('Number of Clusters')
plt.title("Number of Clusters vs. Mean Squared Error")
plt.draw()
plt.savefig("kMeansCC.png")

plt.figure()
line1, = plt.plot(clusters, clustersTime)
plt.ylabel('Time (microseconds)')
plt.xlabel('Number of Clusters')
plt.title("Number of Clusters vs. Time (microseconds)")
plt.draw()
plt.savefig("kMeansTimeCC.png")

components = np.linspace(1, 30, 30, endpoint=True)
componentsLL = []
componentsTime = []
from sklearn.mixture import GaussianMixture
for component in components:
    before = time.time()
    gmm = GaussianMixture(n_components=int(component)).fit(X)
    componentsTime.append(time.time() - before)
    componentsLL.append(gmm.score(X))
    labels = gmm.predict(X)
    print "Components: ", component
    print "Converged? ", gmm.converged_
    print "Log_Likelihood", gmm.score(X)
    print len(gmm.predict_proba(X)[1])
    print gmm.predict_proba(X)[0]
    print "\n\n"

plt.figure()
line1, = plt.plot(components, componentsLL)
plt.ylabel('Log Likelihood')
plt.xlabel('Number of Components')
plt.title("Number of Components vs. Log Likelihood")
plt.draw()
plt.savefig("GMMLogLikelihoodCC.png")

plt.figure()
line1, = plt.plot(components, componentsTime)
plt.ylabel('Time (microseconds)')
plt.xlabel('Number of Components')
plt.title("Number of Components vs. Time (microseconds)")
plt.draw()
plt.savefig("GMMTimeCC.png")


from sklearn.decomposition import PCA
pca = PCA(n_components=20)
pca.fit(X)
PCAComponents = pca.components_ #[[]]
PCARatio = pca.explained_variance_ratio_ #[]
print "\n"

numComponents = np.linspace(1, 20, 20, endpoint=True)
plt.figure()
line1, = plt.plot(numComponents, pca.explained_variance_ratio_)
plt.ylabel('Ratio of Explained Variance by Component')
plt.xlabel('Number of Components')
plt.title("Number of Components vs. Explained Variance Ratio")
plt.draw()
plt.savefig("PCACC.png")

pca = PCA(n_components=1) #taking the first component
PCA_X = pca.fit(X).transform(X)

print PCA_X

clusters = np.linspace(2, 25, 25, endpoint=True)
clustersMSE = []
clustersTime = []
for cluster in clusters:
    before = time.time()
    clusterer = KMeans(n_clusters=int(cluster)).fit(PCA_X)
    print("cluster centers ", clusterer.cluster_centers_)
    print("inertia ", clusterer.inertia_)
    clustersMSE.append(clusterer.inertia_)
    clustersTime.append(time.time() - before)

print(clustersMSE)

plt.figure()
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(clusters, clustersMSE, 'b')
plt.ylabel('Mean Squared Error')
plt.xlabel('Number of Clusters')
plt.title("Number of Clusters vs. Mean Squared Error")
plt.draw()
plt.savefig("PCAkMeansCC.png")

plt.figure()
line1, = plt.plot(clusters, clustersTime)
plt.ylabel('Time (microseconds)')
plt.xlabel('Number of Clusters')
plt.title("Number of Clusters vs. Time (microseconds)")
plt.draw()
plt.savefig("PCAkMeansTimeCC.png")

components = np.linspace(1, 30, 30, endpoint=True)
componentsLL = []
componentsTime = []
from sklearn.mixture import GaussianMixture
for component in components:
    before = time.time()
    gmm = GaussianMixture(n_components=int(component)).fit(PCA_X)
    componentsTime.append(time.time() - before)
    componentsLL.append(gmm.score(PCA_X))
    labels = gmm.predict(PCA_X)
    print "Components: ", component
    print "Converged? ", gmm.converged_
    print "Log_Likelihood", gmm.score(PCA_X)
    print len(gmm.predict_proba(PCA_X)[1])
    print gmm.predict_proba(PCA_X)[0]
    print "\n\n"

plt.figure()
line1, = plt.plot(components, componentsLL)
plt.ylabel('Log Likelihood')
plt.xlabel('Number of Components')
plt.title("Number of Components vs. Log Likelihood")
plt.draw()
plt.savefig("PCAGMMLogLikelihoodCC.png")

plt.figure()
line1, = plt.plot(components, componentsTime)
plt.ylabel('Time (microseconds)')
plt.xlabel('Number of Components')
plt.title("Number of Components vs. Time (microseconds)")
plt.draw()
plt.savefig("PCAGMMTimeCC.png")
