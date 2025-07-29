import networkx as nx
import numpy as np

def global_efficiency_weighted(G, weight="weight"):

    """

    compute the global efficiency of a weighted graph.
 
    Parameters

    ----------

    G : networkx.Graph

        graph where edges may have weights representing distances or costs.

    weight : str, optional

        name of the edge attribute to use as weight (default is "weight").
 
    Returns

    -------

    g_eff : float

        global efficiency of the graph, defined as the average of the 

        inverse shortest path lengths between all pairs of nodes.

    """
 
    # adapted from networkx to take weights into account
 
    N = len(G)

    n_paths = N * (N - 1)  # total number of ordered node pairs
 
    if N > 5000:

        print("too large:", G)

        return np.nan
 
    if n_paths == 0 :

        return 0
 
    node_pos = {n : np.array(G.nodes[n]["pts"]) for n in G.nodes}
 
    g_eff = 0

    g_eff_euclid = 0
 
    lengths = nx.all_pairs_dijkstra_path_length(G, weight=weight, backend="parallel")  # compute shortest paths

    for source, targets in lengths:

        for target, length_sp in targets.items():

            if length_sp > 0:  # avoid self-loops

                g_eff += 1 / length_sp
 
                length_euclid = np.linalg.norm(node_pos[target] - node_pos[source])

                g_eff_euclid += 1 / length_euclid
 
    g_eff /= n_paths

    g_eff_euclid /= n_paths
 
    return g_eff if g_eff==0 else g_eff / g_eff_euclid
 
def local_efficiency_weighted(G):

    """Returns the average local efficiency of the graph.
 
    The *efficiency* of a pair of nodes in a graph is the multiplicative

    inverse of the shortest path distance between the nodes. The *local

    efficiency* of a node in the graph is the average global efficiency of the

    subgraph induced by the neighbors of the node. The *average local

    efficiency* is the average of the local efficiencies of each node [1]_.
 
    Parameters

    ----------

    G : :class:`networkx.Graph`

        An undirected graph for which to compute the average local efficiency.
 
    Returns

    -------

    float

        The average local efficiency of the graph.
 
    Examples

    --------
>>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
>>> nx.local_efficiency(G)

    0.9166666666666667
 
    Notes

    -----

    Edge weights are ignored when computing the shortest path distances.
 
    See also

    --------

    global_efficiency
 
    References

    ----------

    .. [1] Latora, Vito, and Massimo Marchiori.

           "Efficient behavior of small-world networks."

           *Physical Review Letters* 87.19 (2001): 198701.
<https://doi.org/10.1103/PhysRevLett.87.198701>
 
    """

    N = len(G)

    efficiency_list = (global_efficiency_weighted(G.subgraph(G[v])) for v in G)

    return sum(efficiency_list) / N
 
def robustness_curve(G, root_node=0, n_steps=20, mode="uniform", seed=None, n_reps=10, max_fraction_rm=1):

    """

    Compute robustness curve: percentage of total network length

    connected to root node as a function of fraction of edge length removed.

    Averaged over multiple repetitions.
 
    Parameters:

        G : networkx.Graph

            The input graph with edge attribute 'weight'.

        root_node : node label

            The node considered as the root.

        n_steps : int

            Number of removal steps between 0 and 1.

        mode : str

            'uniform' (default) or 'length_weighted' for removal probability.

        seed : int or None

            Random seed for reproducibility.

        n_reps : int

            Number of repetitions to average over.
 
    Returns:

        fractions_removed : np.ndarray

        fractions_connected_mean : np.ndarray

    """
 
    rng = np.random.seed(seed)
 
    edges = list(G.edges(data=True))

    weights = np.array([attr['weight'] for (_, _, attr) in edges])

    total_length = weights.sum()
 
    fractions_removed = np.linspace(0, max_fraction_rm, n_steps)

    all_fractions_connected = []
 
    for _ in range(n_reps):

        fractions_connected = []
 
        for fraction in fractions_removed:

            G_copy = G.copy()
 
            n_edges_to_remove = int(fraction * len(edges))
 
            if n_edges_to_remove > 0:

                if mode == "uniform":

                    selected_indices = np.random.choice(len(edges), size=n_edges_to_remove, replace=False)

                elif mode == "length_weighted":

                    probabilities = weights / weights.sum()

                    selected_indices =  np.random.choice(len(edges), size=n_edges_to_remove, replace=False, p=probabilities)

                else:

                    raise ValueError("Unknown mode: choose 'uniform' or 'length_weighted'")
 
                edges_to_remove = [edges[i][:2] for i in selected_indices]

                G_copy.remove_edges_from(edges_to_remove)
 
            # Get connected component containing root node

            if root_node in G_copy:

                components = nx.node_connected_component(G_copy, root_node)

                subgraph = G_copy.subgraph(components)

                connected_length = sum(nx.get_edge_attributes(subgraph, 'weight').values())

            else:

                connected_length = 0.0
 
            fractions_connected.append(connected_length / total_length)
 
        all_fractions_connected.append(fractions_connected)
 
    fractions_connected_mean = np.mean(all_fractions_connected, axis=0)
 
    return fractions_removed, fractions_connected_mean
 
 
def robustness_score(fractions_removed, fractions_connected, mode="auc", target_fraction=0.5):

    """

    Summarize a robustness curve into a single number.
 
    Parameters:

        fractions_removed : np.ndarray

        fractions_connected : np.ndarray

        mode : str

            'auc' (default) to compute area under the curve (normalized),

            or 'target_fraction' to find the fraction removed when

            connected component drops below target_fraction.

        target_fraction : float

            Target fraction for 'target_fraction' mode (default 0.5).
 
    Returns:

        summary_value : float

    """

    if mode == "auc":

        auc = np.trapz(fractions_connected, fractions_removed)

        max_auc = 1.0 * 1.0  # maximum area = 1*1

        return auc / max_auc
 
    elif mode == "target_fraction":

        below_target = fractions_connected <= target_fraction

        if np.any(below_target):

            idx = np.argmax(below_target)

            return fractions_removed[idx]

        else:

            return 1.0  # never dropped below target, very robust
 
    else:

        raise ValueError("Unknown mode: choose 'auc' or 'target_fraction'")
 