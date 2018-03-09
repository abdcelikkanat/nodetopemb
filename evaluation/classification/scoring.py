#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""

__author__      = "Bryan Perozzi"

import numpy
import sys

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer

from scipy.sparse import csr_matrix

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels

def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in iteritems(G)}

def main():
  datasetname = "blogcatalog"
  base = "../../twe1/temp_files/"
  folder_directory = "output/"+datasetname+"/"

  file_path = "blogcatalog_pathlen250_numofpaths10_combined.embedding"
  #file_path = "citeseer_degreeBasedWalk_Pow2_combined.embedding"
  mat_file = datasetname+".mat"
  output_text_file = "../results/blogcatalog_pathlen250_numofpaths10_combined.result"
  #output_text_file = "../results/citeseer_degreeBasedWalk_Pow2.result"


  embeddings_file = base + folder_directory + file_path
  matfile = "../../mat_files/" + mat_file
  adj_matrix_name = "network"
  label_matrix_name = "group"
  num_shuffles = 25
  all = False

  # 1. Load Embeddings
  model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
  
  # 2. Load labels
  mat = loadmat(matfile)
  A = mat[adj_matrix_name]
  graph = sparse2graph(A)
  labels_matrix = mat[label_matrix_name]
  labels_count = labels_matrix.shape[1]
  mlb = MultiLabelBinarizer(range(labels_count))
  
  # Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
  features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])

  # 2. Shuffle, to create train/test groups
  shuffles = []
  for x in range(num_shuffles):
    shuffles.append(skshuffle(features_matrix, labels_matrix))
  
  # 3. to score each train/test group
  all_results = defaultdict(list)
  
  if all:
    training_percents = numpy.asarray(range(1, 10)) * .1
  else:
    training_percents = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  for train_percent in training_percents:
    for shuf in shuffles:
      X, y = shuf

      training_size = int(train_percent * X.shape[0])

      X_train = csr_matrix(X[:training_size, :])
      y_train_ = csr_matrix(y[:training_size])

      y_train = [[] for x in range(y_train_.shape[0])]

      cy = y_train_.tocoo()
      for i, j in zip(cy.row, cy.col):
          y_train[i].append(j)

      assert sum(len(l) for l in y_train) == y_train_.nnz

      X_test = csr_matrix(X[training_size:, :])
      y_test_ = csr_matrix(y[training_size:])

      y_test = [[] for _ in range(y_test_.shape[0])]
  
      cy =  y_test_.tocoo()
      for i, j in zip(cy.row, cy.col):
          y_test[i].append(j)
  
      clf = TopKRanker(LogisticRegression())

      clf.fit(X_train, y_train_)
  
      # find out how many labels should be predicted
      top_k_list = [len(l) for l in y_test]
      preds = clf.predict(X_test, top_k_list)
  
      results = {}
      averages = ["micro", "macro"]
      for average in averages:
          results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
  
      all_results[train_percent].append(results)

  output_text = ""
  output_text += "Results, using embeddings of dimensionality" + str(X.shape[1]) + "\n"
  output_text += "-------------------"  + "\n"
  for train_percent in sorted(all_results.keys()):
    output_text += "Train percent:" + str(train_percent)
    for index, result in enumerate(all_results[train_percent]):
      output_text += "Shuffle #" + str(index + 1) + " " + str(result) + "\n"
    avg_score = defaultdict(float)
    for score_dict in all_results[train_percent]:
      for metric, score in iteritems(score_dict):
        avg_score[metric] += score
    for metric in avg_score:
      avg_score[metric] /= len(all_results[train_percent])
    output_text += "Average score:" + str(dict(avg_score))  + "\n"
    output_text += "-------------------"  + "\n"

  with open(output_text_file, 'w') as f:
      f.write(output_text)

main()