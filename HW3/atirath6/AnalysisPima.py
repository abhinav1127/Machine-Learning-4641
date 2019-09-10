import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import time
from sklearn.model_selection import cross_val_score,learning_curve
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, auc, recall_score
from sklearn.neural_network import MLPClassifier
from matplotlib.legend_handler import HandlerLine2D
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
from numpy import array



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    # plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



data = pd.read_csv("pimaDiabetes.csv")
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

line1, = plt.plot(clusters, clustersMSE, 'b')
plt.ylabel('Mean Squared Error')
plt.xlabel('Number of Clusters')
plt.title("Number of Clusters vs. Mean Squared Error")
plt.draw()
plt.savefig("kMeansPima.png")

plt.figure()
line1, = plt.plot(clusters, clustersTime)
plt.ylabel('Time (microseconds)')
plt.xlabel('Number of Clusters')
plt.title("Number of Clusters vs. Time (microseconds)")
plt.draw()
plt.savefig("kMeansTimePima.png")




components = np.linspace(1, 30, 30, endpoint=True)
componentsLL = []
componentsTime = []
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
plt.savefig("GMMLogLikelihoodPima.png")

plt.figure()
line1, = plt.plot(components, componentsTime)
plt.ylabel('Time (microseconds)')
plt.xlabel('Number of Components')
plt.title("Number of Components vs. Time (microseconds)")
plt.draw()
plt.savefig("GMMTimePima.png")


pca = PCA(n_components=8)
pca.fit(X)
PCAComponents = pca.components_ #[[]]
PCARatio = pca.explained_variance_ratio_ #[]
print PCARatio
print "\n"

numComponents = np.linspace(1, 8, 8, endpoint=True)
plt.figure()
line1, = plt.plot(numComponents, PCARatio)
plt.ylabel('Ratio of Explained Variance by Component')
plt.xlabel('Number of Components')
plt.title("Number of Components vs. Explained Variance Ratio")
plt.draw()
plt.savefig("PCAPima.png")

pca = PCA(n_components=3) #taking the first 3 components
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
line1, = plt.plot(clusters, clustersMSE, 'b')
plt.ylabel('Mean Squared Error')
plt.xlabel('Number of Clusters')
plt.title("Number of Clusters vs. Mean Squared Error")
plt.draw()
plt.savefig("PCAkMeansPima.png")

plt.figure()
line1, = plt.plot(clusters, clustersTime)
plt.ylabel('Time (microseconds)')
plt.xlabel('Number of Clusters')
plt.title("Number of Clusters vs. Time (microseconds)")
plt.draw()
plt.savefig("PCAkMeansTimePima.png")

components = np.linspace(1, 30, 30, endpoint=True)
componentsLL = []
componentsTime = []
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
plt.savefig("PCAGMMLogLikelihoodPima.png")

plt.figure()
line1, = plt.plot(components, componentsTime)
plt.ylabel('Time (microseconds)')
plt.xlabel('Number of Components')
plt.title("Number of Components vs. Time (microseconds)")
plt.draw()
plt.savefig("PCAGMMTimePima.png")


# Step 4

train = pd.read_csv("pimaDiabetes_train.csv")
test = pd.read_csv("pimaDiabetes_test.csv")
X_train = train.drop(['Class'], axis=1)
y_train = train['Class']
X_test = test.drop(['Class'], axis=1)
y_test = test['Class']

pca = PCA(n_components=8)
pca.fit(X_train)
PCAComponents = pca.components_ #[[]]
PCARatio = pca.explained_variance_ratio_ #[]
print PCARatio
print "\n"

numComponents = np.linspace(1, 8, 8, endpoint=True)
plt.figure()
line1, = plt.plot(numComponents, PCARatio)
plt.ylabel('Ratio of Explained Variance by Component')
plt.xlabel('Number of Components')
plt.title("Number of Components vs. Explained Variance Ratio")
plt.draw()
plt.savefig("NNTrainingPCAPima.png")

pca = PCA(n_components=3) #taking the first 3 components AGAIN
X_train = pca.fit(X_train).transform(X_train)
X_test = pca.transform(X_test)



idealLayers = 1

nodesinLayers = np.linspace(1, 25, 25, endpoint=True)
train_results = []
CV_results = []
recall_train_results = []
recall_CV_results = []
layer = []

for node in nodesinLayers:
    tuple  = (int(node),) * int(idealLayers)
    print(tuple)
    clf = MLPClassifier(hidden_layer_sizes=tuple, max_iter=500, solver='adam', activation='tanh')
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='recall')
    recall_CV_results.append(sum(scores) / float(len(scores)))
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
    CV_results.append(sum(scores) / float(len(scores)))
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    recall_train_results.append(recall_score(y_train, train_pred))

print recall_train_results
print recall_CV_results
print train_results
print CV_results
plt.figure()
line1, = plt.plot(nodesinLayers, train_results, 'b', label='Train AUC')
line2, = plt.plot(nodesinLayers, CV_results, 'r', label='CV AUC')
line3, = plt.plot(nodesinLayers, recall_train_results, 'b', label='Train recall', color='green')
line4, = plt.plot(nodesinLayers, recall_CV_results, 'r', label='CV recall', color='yellow')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC/ recall')
plt.xlabel('Number of Nodes in each Layer')
plt.title('Nodes per Layer')
plt.savefig("NNPCANodesinLayersPima.png")


numOfNodes = CV_results.index(max(CV_results))
numOfNodes += 1

tuple = (numOfNodes,) * idealLayers
mlp = MLPClassifier(hidden_layer_sizes=(tuple),max_iter=500, solver= 'adam', activation= 'tanh')
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

plt.figure()
cnf_matrix = confusion_matrix(y_test, predictions)
plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                      title='Confusion matrix')
plt.savefig("NNPCACFMatrixNPD.png")

plt.figure()
plot_learning_curve(mlp, "Learning Curve", X_train, y_train, cv=3)
plt.savefig("NNPCALearningCurvePima.png")

# # Step 5

train = pd.read_csv("pimaDiabetes_train.csv")
test = pd.read_csv("pimaDiabetes_test.csv")
X_train = train.drop(['Class'], axis=1)
y_train = train['Class']
X_test = test.drop(['Class'], axis=1)
y_test = test['Class']

# clusters = np.linspace(2, 25, 25, endpoint=True)
# clustersMSE = []
# for cluster in clusters:
#     clusterer = KMeans(n_clusters=int(cluster)).fit(X_train)
#     clustersMSE.append(clusterer.inertia_)
#
# line1, = plt.plot(clusters, clustersMSE, 'b')
# plt.ylabel('Mean Squared Error')
# plt.xlabel('Number of Clusters')
# plt.title("Number of Clusters vs. Mean Squared Error")
# plt.draw()
# plt.savefig("NNTrainkMeansPima.png")


clusterer = KMeans(n_clusters=8).fit(X_train)
trainingPredictions = clusterer.predict(X_train)
testPredictions = clusterer.predict(X_test)

X_train = pd.DataFrame(data={'predictedGroup': trainingPredictions})
X_test = pd.DataFrame(data={'predictedGroup': testPredictions})


idealLayers = 2

nodesinLayers = np.linspace(1, 25, 25, endpoint=True)
train_results = []
CV_results = []
recall_train_results = []
recall_CV_results = []
layer = []

for node in nodesinLayers:
    tuple  = (int(node),) * int(idealLayers)
    print(tuple)
    clf = MLPClassifier(hidden_layer_sizes=tuple, max_iter=500, solver='adam', activation='tanh')
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='recall')
    recall_CV_results.append(sum(scores) / float(len(scores)))
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
    CV_results.append(sum(scores) / float(len(scores)))
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    recall_train_results.append(recall_score(y_train, train_pred))

print recall_train_results
print recall_CV_results
print train_results
print CV_results
plt.figure()
line1, = plt.plot(nodesinLayers, train_results, 'b', label='Train AUC')
line2, = plt.plot(nodesinLayers, CV_results, 'r', label='CV AUC')
line3, = plt.plot(nodesinLayers, recall_train_results, 'b', label='Train recall', color='green')
line4, = plt.plot(nodesinLayers, recall_CV_results, 'r', label='CV recall', color='yellow')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC/ recall')
plt.xlabel('Number of Nodes in each Layer')
plt.title('Nodes per Layer')
plt.savefig("NNKmeansNodesinLayersPima.png")


numOfNodes = CV_results.index(max(CV_results))
numOfNodes += 1

tuple = (numOfNodes,) * idealLayers
mlp = MLPClassifier(hidden_layer_sizes=(tuple),max_iter=500, solver= 'adam', activation= 'tanh')
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

plt.figure()
cnf_matrix = confusion_matrix(y_test, predictions)
plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                      title='Confusion matrix')
plt.savefig("NNKmeansCFMatrix_Pima.png")

plt.figure()
plot_learning_curve(mlp, "Learning Curve", X_train, y_train, cv=3)
plt.savefig("NNKmeansLearningCurvePima.png")


train = pd.read_csv("pimaDiabetes_train.csv")
test = pd.read_csv("pimaDiabetes_test.csv")
X_train = train.drop(['Class'], axis=1)
y_train = train['Class']
X_test = test.drop(['Class'], axis=1)
y_test = test['Class']

# components = np.linspace(1, 30, 30, endpoint=True)
# componentsLL = []
# for component in components:
#     gmm = GaussianMixture(n_components=int(component)).fit(X_train)
#     componentsLL.append(gmm.score(X_train))
#     labels = gmm.predict(X_train)
#
# plt.figure()
# line1, = plt.plot(components, componentsLL)
# plt.ylabel('Log Likelihood')
# plt.xlabel('Number of Components')
# plt.title("Number of Components vs. Log Likelihood")
# plt.draw()
# plt.savefig("NNTrainGMMLogLikelihoodPima.png")

gmm = GaussianMixture(n_components=11).fit(X_train)
probaTrain = gmm.predict_proba(X_train)
probaTest = gmm.predict_proba(X_test)

X_train = array(probaTrain)
X_test = array(probaTest)

print X_train
print X_test

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

idealLayers = 2

nodesinLayers = np.linspace(1, 25, 25, endpoint=True)
train_results = []
CV_results = []
recall_train_results = []
recall_CV_results = []
layer = []

for node in nodesinLayers:
    tuple  = (int(node),) * int(idealLayers)
    print(tuple)
    clf = MLPClassifier(hidden_layer_sizes=tuple, max_iter=500, solver='adam', activation='tanh')
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='recall')
    recall_CV_results.append(sum(scores) / float(len(scores)))
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
    CV_results.append(sum(scores) / float(len(scores)))
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    recall_train_results.append(recall_score(y_train, train_pred))

print recall_train_results
print recall_CV_results
print train_results
print CV_results
plt.figure()
line1, = plt.plot(nodesinLayers, train_results, 'b', label='Train AUC')
line2, = plt.plot(nodesinLayers, CV_results, 'r', label='CV AUC')
line3, = plt.plot(nodesinLayers, recall_train_results, 'b', label='Train recall', color='green')
line4, = plt.plot(nodesinLayers, recall_CV_results, 'r', label='CV recall', color='yellow')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC/ recall')
plt.xlabel('Number of Nodes in each Layer')
plt.title('Nodes per Layer')
plt.savefig("NNGMMNodesinLayersPima.png")


numOfNodes = CV_results.index(max(CV_results))
numOfNodes += 1

tuple = (numOfNodes,) * idealLayers
mlp = MLPClassifier(hidden_layer_sizes=(tuple),max_iter=500, solver= 'adam', activation= 'tanh')
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

plt.figure()
cnf_matrix = confusion_matrix(y_test, predictions)
plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                      title='Confusion matrix')
plt.savefig("NNGMMCFMatrixPima.png")

plt.figure()
plot_learning_curve(mlp, "Learning Curve", X_train, y_train, cv=3)
plt.savefig("NNGMMLearningCurvePima.png")
