from __future__ import print_function

import csv

from StochasticGraphGenerator import StochasticGraphGenerator
import igraph
import numpy
import random
from collections import Counter

from collections import Counter


def calc_result_and_print(graph, origin_cluster):
    fg_cluster = graph.community_fastgreedy().as_clustering()
    im_cluster = graph.community_infomap()
    lp_cluster = graph.community_label_propagation()
    ml_cluster = graph.community_multilevel()
    wt_cluster = graph.community_walktrap().as_clustering()
    fg_result = igraph.compare_communities(origin_cluster, fg_cluster, method='adjusted_rand')
    im_result = igraph.compare_communities(origin_cluster, im_cluster, method='adjusted_rand')
    lp_result = igraph.compare_communities(origin_cluster, lp_cluster, method='adjusted_rand')
    ml_result = igraph.compare_communities(origin_cluster, ml_cluster, method='adjusted_rand')
    wt_result = igraph.compare_communities(origin_cluster, wt_cluster, method='adjusted_rand')
    # print("FastGready result {0}".format(fg_result))
    # print("InfoMap result {0}".format(im_result))
    # print("LabelPropagation result {0}".format(lp_result))
    # print("MultiLevel result {0}".format(ml_result))
    # print("WalkTrap result {0}".format(wt_result))
    return [fg_result, im_result, lp_result, ml_result, wt_result]


def anlyis_graph_edges(graph, alpa_edges, beta_edges):
    f_beta = open('beta.csv', 'wb')
    f_beta.write('edge,similarity_jaccard\n')
    local_clustering_coefficient = graph.transitivity_local_undirected()
    for i, e in enumerate(beta_edges):
        if i % 100 == 0:
            print('\redges {0}/{1}'.format(i, len(beta_edges)), end='')
        graph.delete_edges([e])
        similarity_jaccard = graph.similarity_jaccard([e[0], e[1]])[0][1]
        f_beta.write('({0};{1}), {2}\n'.format(e[0], e[1], similarity_jaccard))
        graph.add_edges([e])
    pass
    f_beta.flush()
    f_beta.close()
    print()
    f_alpha = open('alpha.csv', 'wb')
    f_alpha.write('edge,similarity_jaccard\n')
    for i, e in enumerate(alpa_edges):
        if i % 100 == 0:
            print('\redges {0}/{1}'.format(i, len(alpa_edges)), end='')
        graph.delete_edges([e])
        similarity_jaccard = graph.similarity_jaccard([e[0], e[1]])[0][1]
        f_alpha.write('({0};{1}), {2}\n'.format(e[0], e[1], similarity_jaccard))
        graph.add_edges([e])
    pass
    f_alpha.flush()
    f_alpha.close()


def write_generated_graph_data_to_csv(graph_sizes, communities, alphas, betas):
    f = open('output_expirement_results.csv', 'wb')
    csv_f = csv.writer(f)
    csv_f.writerow(['index', 'graph_size', 'alpha', 'beta', 'communities', 'density', 'beta_edges_median'])
    rows = []
    i = 1
    total_graphs_generated = numpy.prod(map(len, [communities, alphas, betas, graph_sizes])) * 10
    for graph_size in graph_sizes:
        for community in communities:
            for alpha in alphas:
                for beta in betas:
                    for j in xrange(10):
                        print('graph generating {0}/{1}'.format(i, total_graphs_generated))

                        origin_membership = StochasticGraphGenerator.generate_equal_size_membership(graph_size,
                                                                                                    community)

                        graph, beta_edges, alpha_edges = StochasticGraphGenerator().generate(graph_size,
                                                                                             origin_membership, alpha,
                                                                                             beta)
                        beta_median = numpy.median(graph.similarity_jaccard(pairs=beta_edges))
                        density = graph.density()
                        rows.append([i, graph_size, alpha, beta, community, density, beta_median])
                        i += 1
    csv_f.writerows(rows)


def run_expirement(graph, threshold):
    graph_size = len(graph.vs)
    nodes = xrange(graph_size)
    iterations = 100000
    edge_deleted = []
    for edge in graph.get_edgelist():
        similarity_jaccard = graph.similarity_jaccard(pairs=[edge])[0]
        if similarity_jaccard < threshold:
            edge_deleted.append(edge)
    graph.delete_edges(edge_deleted)
    print("edges deleted {0}".format(len(edge_deleted)))
    return edge_deleted


betas = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.04, 0.05, 0.08, 0.1]
alphas = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
communities = [10, 7, 5, 2]
graph_sizes = [1000, 500]

# write_generated_graph_data_to_csv(graph_sizes, communities, alphas, betas)
f = open('output_delete_edges_results_2.csv', 'wb')
csv_f = csv.writer(f)
csv_f.writerow(['index', 'mod', 'graph_size', 'alpha', 'beta', 'communities', 'density', 'threshold', 'FastGready', 'InfoMap',
                'LabelPropagation', 'MultiLevel', 'WalkTrap', 'edge_deleted'])
rows = []
i = 0
total_graphs_generated = numpy.prod(map(len, [communities, alphas, betas, graph_sizes])) * 1
for graph_size in graph_sizes:
    for community in communities:
        for alpha in alphas:
            for beta in betas:
                print('graph generating {0}/{1}'.format(i, total_graphs_generated))
                densities = []
                origin_results = []
                new_results = []
                edges_deleted = []
                for j in xrange(5):
                    origin_membership = StochasticGraphGenerator.generate_equal_size_membership(graph_size, community)

                    graph, beta_edges, alpha_edges = StochasticGraphGenerator().generate(graph_size, origin_membership,
                                                                                         alpha, beta)
                    origin_cluster = igraph.Clustering(origin_membership)
                    origin_results.append(calc_result_and_print(graph, origin_cluster))
                    density = graph.density()
                    densities.append(density)
                    threshold = 0.0144 * (density ** -0.431)
                    edges_deleted.append(run_expirement(graph, threshold))
                    new_results.append(calc_result_and_print(graph, origin_cluster))

                density = numpy.mean(densities)
                threshold = 0.0144 * (density ** -0.431)
                fg, im, lp, ml, wt = map(numpy.mean, zip(*origin_results))
                rows.append([i, 'origin', graph_size, alpha, beta, community, density, threshold, fg, im, lp, ml, wt, 0])
                fg, im, lp, ml, wt = map(numpy.mean, zip(*new_results))
                avg_deleted_edges = int(numpy.mean(edges_deleted))
                rows.append([i, 'delete_edges', graph_size, alpha, beta, community, density, threshold, fg, im, lp, ml, wt, avg_deleted_edges])
                i += 1
csv_f.writerows(rows)
f.close()
# beta_median = numpy.median(graph.similarity_jaccard(pairs=beta_edges))
# print()
# print()
# # anlyis_graph_edges(graph, alpha_edges, beta_edges)
# density = graph.density()
# threshold = 0.0144 * (density ** -0.431)
# print('beta median {0}'.format(beta_median))
# print('threshold {0}'.format(threshold))
# print("graph density {0}".format(density))
#
# run_expirement(graph, threshold)
# print()
# calc_result_and_print(graph, origin_cluster)
pass
