# Estimate source-edge adaptation factors from source/target embedding-distance bins.
# For bin b, use w_b = (target edges / target pairs) / (source edges / source pairs).
# No class labels or pseudo-labels are consumed by this implementation.
import numpy as np
import torch

def calculate_distance_matrix(node_features):
    """
    Calculate the Euclidean distances between all pairs of nodes.
    :param node_features: Node feature matrix H^(k), shape [num_nodes, feature_dim]
    :return: Distance matrix, shape [num_nodes, num_nodes]
    """
    num_nodes = node_features.shape[0]
    # Materialize all ordered pairs, including diagonal self-pairs: O(N^2) storage.
    distance_matrix = torch.cdist(node_features, node_features, p=2) 
    return distance_matrix


def calculate_reweight_return(src_graph, tgt_graph, src_embed, tgt_embed, num_bins, device):
    """
    Reweight the edge weights of the source domain graph based on node embedding distances.
    :param src_graph: Source domain graph, with edge_index attribute [2, num_edges_src]
    :param tgt_graph: Target domain graph, with edge_index attribute [2, num_edges_tgt]
    :param src_embed: Node embeddings of the source domain [num_nodes_src, feature_dim]
    :param tgt_embed: Node embeddings of the target domain [num_nodes_tgt, feature_dim]
    :param num_bins: Number of distance intervals (bins)
    :param device: Computation device (CPU or GPU)
    """
    # Returns [E_src] factors in source edge_index order; neither graph is modified in place.
    # Embeddings, edge indices, and bin boundaries must reside on compatible devices.
    num_nodes_src = src_graph.num_nodes
    src_edge_index = src_graph.edge_index  # [2, num_edges_src]
    num_edges_src = src_edge_index.size(1)
    num_nodes_tgt = tgt_graph.num_nodes
    tgt_edge_index = tgt_graph.edge_index  # [2, num_edges_tgt]
    num_edges_tgt = tgt_edge_index.size(1)

    # Step 1: Compute the distance matrix
    src_distance_matrix = calculate_distance_matrix(src_embed)  # [N_src, N_src]
    tgt_distance_matrix = calculate_distance_matrix(tgt_embed)  # [N_tgt, N_tgt]

    # Step 2: Calculate the global minimum and maximum distances
    # Share one distance range across domains so their bin statistics can be compared.
    combined_max_distance = max(src_distance_matrix.max().item(), tgt_distance_matrix.max().item())
    combined_min_distance = min(src_distance_matrix.min().item(), tgt_distance_matrix.min().item())
    bins = torch.linspace(combined_min_distance, combined_max_distance, num_bins + 1, device=device)

    # Step 3: Calculate the distance for nodes of each edge
    src_edge_distances = src_distance_matrix[src_edge_index[0], src_edge_index[1]]  # [num_edges_src]
    tgt_edge_distances = tgt_distance_matrix[tgt_edge_index[0], tgt_edge_index[1]]  # [num_edges_tgt]

    # Step 4: Assign edge distances to their corresponding bins
    # Clamp endpoint cases into valid bins, including zero-distance edges and the maximum.
    src_bin_indices = torch.bucketize(src_edge_distances, bins) - 1  # [num_edges_src]
    src_bin_indices = src_bin_indices.clamp(min=0, max=num_bins - 1)  # Ensure indices are within the valid range.

    tgt_bin_indices = torch.bucketize(tgt_edge_distances, bins) - 1  # [num_edges_tgt]
    tgt_bin_indices = tgt_bin_indices.clamp(min=0, max=num_bins - 1)  # Ensure indices are within the valid range.

    # Step 5: Count the number of edges in each bin
    src_edge_count = torch.bincount(src_bin_indices, minlength=num_bins).float()  # [num_bins]
    tgt_edge_count = torch.bincount(tgt_bin_indices, minlength=num_bins).float()  # [num_bins]

    # Step 6: Count the total number of node pairs (pairwsie) in each bin
    # for source domain:
    # Denominators include all ordered node pairs, whereas numerators count listed edges.
    src_total_count = torch.histc(src_distance_matrix.view(-1), bins=num_bins, min=combined_min_distance,
                                  max=combined_max_distance).to(device)  # [num_bins]

    # for target domain:
    tgt_total_count = torch.histc(tgt_distance_matrix.view(-1), bins=num_bins, min=combined_min_distance,
                                  max=combined_max_distance).to(device)  # [num_bins]

    # Replace empty-bin denominators with one; zero observed edges still yield zero probability.
    src_total_count = torch.where(src_total_count == 0, torch.ones_like(src_total_count), src_total_count)
    tgt_total_count = torch.where(tgt_total_count == 0, torch.ones_like(tgt_total_count), tgt_total_count)

    # Step 7: Calculate the conditional probabilities P^S and P^T
    src_edge_prob = src_edge_count / src_total_count  # [num_bins]
    tgt_edge_prob = tgt_edge_count / tgt_total_count  # [num_bins]

    src_edge_prob = torch.where(torch.isnan(src_edge_prob), torch.zeros_like(src_edge_prob), src_edge_prob)
    tgt_edge_prob = torch.where(torch.isnan(tgt_edge_prob), torch.zeros_like(tgt_edge_prob), tgt_edge_prob)

    # Step 8: Calculate the reweighting factor
    # Undefined or infinite target/source ratios fall back to the neutral factor 1.
    # A finite ratio of zero is retained; factors are not globally normalized.
    reweight_factors = tgt_edge_prob / src_edge_prob  # [num_bins]
    reweight_factors = torch.where(torch.isinf(reweight_factors), torch.ones_like(reweight_factors), reweight_factors)
    reweight_factors = torch.where(torch.isnan(reweight_factors), torch.ones_like(reweight_factors), reweight_factors)

    # Step 9: Update edge_weight
    edge_weight = torch.ones(num_edges_src, dtype=torch.float, device=device)  # [num_edges_src]
    # Give every source edge the ratio associated with its embedding-distance bin.
    edge_weight = reweight_factors[src_bin_indices]  # [num_edges_src]

    return edge_weight
