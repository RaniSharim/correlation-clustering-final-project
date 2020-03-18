from scipy.io import arff
from scipy.spatial.distance import jaccard
from scipy.optimize import nnls
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

# load the data and get the class labels 
def loadData():
    data, _ = arff.loadarff('water-quality-nom.arff')
    df = pd.DataFrame(data)
    label_graph = df[df.columns[-14:]].astype('int32').astype('bool').values
    return label_graph

def calculate_similarity_matrix(label_graph):
    similarity_matrix = 1 - pairwise_distances(label_graph, metric='jaccard')
    return similarity_matrix

def C_occ(label_graph , lables, similarity_matrix):
    label_similarity = 1 - pairwise_distances(lables, metric='jaccard')
    diff = np.absolute(label_similarity - similarity_matrix)
    return 0.5 * np.sum(diff)

def C_vp(label_graph , lables, similarity_matrix, v_idx):
    label_similarity = 1 - pairwise_distances(lables, metric='jaccard')
    diff_v = np.absolute(label_similarity[v_idx] - similarity_matrix[v_idx])
    return np.sum(diff_v)

def find_new_label_set(label_graph, labels, similarity_matrix, max_labels, v_idx):
    # current_labels = labels[v_idx]
    z = similarity_matrix[v_idx]
    A = np.zeros([labels.shape[0], labels.shape[1] + 1])
   
    # setup b = z*|S_j|
    s_sizes = np.sum(label_graph, axis=1)
    b = z * s_sizes

    # setup the constrains for nnls
    z_t = z.reshape([z.shape[0] ,1])
    # seems to be a mixup in the article about the signs here...
    # setup -zt + [sum((1+z)x_i) for i in s_j]
    A[:,0] = -z
    A[:,1:] = (1 + z_t) * label_graph
    # get top indices
    (X, _) = nnls(A, b)
    # we don't want t, only the x_i
    X = X[1:]
    top_indices = np.flip(np.argsort(X))

    min_cost = C_vp(label_graph, labels, similarity_matrix, v_idx)
    v_labels = np.copy(labels[v_idx])

    # test all possible configurations
    for p in range(1, max_labels+1):
        # set top p X's to true, otherwise false
        selected_indices = top_indices[:p]
        new_labels = np.zeros(labels.shape[1], dtype=bool)
        np.put(new_labels, selected_indices, True)
        labels[v_idx] = new_labels

        # check which gives the best cost
        cost = C_vp(label_graph, labels, similarity_matrix, v_idx)
        if cost < min_cost:
            # print(cost)
            min_cost = cost
            v_labels = new_labels

    return v_labels

def local_search_jaccard_triangulation(label_graph, similarity_matrix, max_labels):
    number_of_vertices = label_graph.shape[0]
    # init random labels
    labels = np.zeros(label_graph.shape, dtype=bool)
    labels[:, :max_labels] = True
    for i in range(0, number_of_vertices):
        np.random.shuffle(labels[i])

    total_cost = C_occ(label_graph, labels, similarity_matrix)
    print(total_cost)
    old_cost = total_cost + 1
    iteration = 0
    while old_cost > total_cost:
        iteration = iteration + 1
        old_cost = total_cost
        # local optimization step for each node
        for i in range(0, number_of_vertices):
            if (i % 100 == 0):
                print(i, end = ' ', flush=True)
            labels[i] = find_new_label_set(label_graph, labels, similarity_matrix, max_labels, i)

        total_cost = C_occ(label_graph, labels, similarity_matrix)
        print()
        print(total_cost)
        np.save('labels_{0}_{1}.npy'.format(max_labels, iteration), labels)
    
    np.save('final_labels_{0}.npy'.format(max_labels), labels)


def precision_and_recall(labels, label_graph):
    # find the size of the intersection
    intersection_sizes = np.sum(np.logical_and(labels, label_graph), axis = 1)
    # find the size of the prediction and truth per node
    prediction_sizes = np.sum(labels, axis = 1, dtype = float)
    truth_sizes = np.sum(label_graph, axis = 1, dtype = float)

    # print(truth_sizes)
    # print(intersection_sizes)
    # print(prediction_sizes)

    # calculate precision and recall per node
    precision = np.divide(intersection_sizes, prediction_sizes, where=prediction_sizes!=0)
    recall = np.divide(intersection_sizes, truth_sizes, where=truth_sizes!=0)

    # return avarage
    return (np.average(precision), np.average(recall))

def greedy_label_assignment(label_graph, similarity_matrix, max_labels):
    # calculate cardinalities of labels and sort high to low
    sorted_label_cardinalities = np.flip(np.argsort(np.sum(label_graph, axis = 0)))
    labels = np.zeros_like(label_graph, dtype = bool)
    for vertex_idx in range(0, label_graph.shape[0]):
        numer_of_assigned_labels = 0
        for label_idx in sorted_label_cardinalities:
            # if the vertex has the given label add it to the vertex
            if (label_graph[vertex_idx, label_idx]):
                labels[vertex_idx, label_idx] = True
                numer_of_assigned_labels = numer_of_assigned_labels + 1

            if numer_of_assigned_labels == max_labels:
                break

    return labels



    


label_graph = loadData()
similarity_matrix = calculate_similarity_matrix(label_graph)
local_search_jaccard_triangulation(label_graph, similarity_matrix, 5)

for max_labels in [3,4,5,6,7,8]:
    print("max labels: {0}".format(max_labels))
    local_search_jaccard_triangulation(label_graph, similarity_matrix, max_labels)

for max_labels in [3,4,5,6,7,8]:
    labels = np.load('final_labels_{0}.npy'.format(max_labels))
    (precision, recall) = precision_and_recall(labels, label_graph)
    print("max labels = {0} agv_precision = {1:.3f} avg_recall = {2:.3f}".format(max_labels, precision, recall))

for max_labels in [3,4,5,6,7,8]:
    labels = greedy_label_assignment(label_graph, similarity_matrix, max_labels)
    total_cost = C_occ(label_graph, labels, similarity_matrix)
    (precision, recall) = precision_and_recall(labels, label_graph)
    print("max labels = {0} cost = {1} agv_precision = {2:.3f} avg_recall = {3:.3f}".format(max_labels, total_cost, precision, recall))