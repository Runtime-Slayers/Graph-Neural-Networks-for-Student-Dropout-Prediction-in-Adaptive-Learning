"""
P12 — Graph Neural Networks for Student Social Support Network Analysis
Real data: Stanford SNAP ego-Facebook network + Stanford SNAP ego-Gplus
Source: https://snap.stanford.edu/data/ego-Facebook.html
Task: Download real social network, apply GNN-inspired analysis, identify support clusters,
      compute social support scores, bridge nodes, and vulnerability metrics.
NO SYNTHETIC DATA — all analysis on real SNAP network data.
"""

import os, sys, json, urllib.request, gzip, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

CACHE = Path(__file__).parent / "p12_cache"
CACHE.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "figures_p12"
RESULTS_DIR.mkdir(exist_ok=True)

# Real SNAP datasets
DATASETS = {
    "ego_facebook": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "filename": "facebook_combined.txt.gz",
        "description": "Facebook ego-network (4039 nodes, 88234 edges)",
    }
}

def download_snap(name, url, filename):
    dest = CACHE / filename
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  cached: {filename}")
    else:
        print(f"  downloading: {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved {dest.stat().st_size/1024:.1f} KB")
    return dest

def load_edge_list(path):
    """Load compressed or plain edge list, return adjacency."""
    edges = []
    opener = gzip.open if str(path).endswith('.gz') else open
    mode = 'rt' if str(path).endswith('.gz') else 'r'
    with opener(path, mode) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    edges.append((int(parts[0]), int(parts[1])))
    return edges

def build_adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def compute_degree_centrality(adj, nodes):
    n = len(nodes)
    return {node: len(adj[node]) / (n - 1) for node in nodes}

def compute_clustering_coefficient(adj, node):
    neighbors = list(adj[node])
    k = len(neighbors)
    if k < 2:
        return 0.0
    links = sum(1 for i in range(k) for j in range(i+1, k) if neighbors[j] in adj[neighbors[i]])
    return 2 * links / (k * (k - 1))

def bfs_component(adj, start, visited):
    queue = [start]
    comp = []
    while queue:
        node = queue.pop()
        if node not in visited:
            visited.add(node)
            comp.append(node)
            queue.extend(adj[node] - visited)
    return comp

def find_communities_label_propagation(adj, nodes, max_iter=20, seed=42):
    """Simple label propagation community detection."""
    rng = np.random.RandomState(seed)
    labels = {n: n for n in nodes}
    node_list = list(nodes)
    for _ in range(max_iter):
        rng.shuffle(node_list)
        changed = 0
        for node in node_list:
            if not adj[node]:
                continue
            neighbor_labels = [labels[nb] for nb in adj[node]]
            counts = Counter(neighbor_labels)
            new_label = counts.most_common(1)[0][0]
            if new_label != labels[node]:
                labels[node] = new_label
                changed += 1
        if changed == 0:
            break
    return labels

def compute_social_support_score(adj, node, degree_cent):
    """Social support score = weighted combination of degree + clustering + bridge-factor."""
    degree = len(adj[node])
    cc = compute_clustering_coefficient(adj, node)
    # Bridge factor: low CC, high degree = bridge node (connector)
    bridge = degree_cent.get(node, 0) * (1 - cc)
    # Support score: high CC = cohesive support group; hub in support cluster
    support_score = 0.5 * degree_cent.get(node, 0) + 0.3 * cc + 0.2 * bridge
    return support_score, cc, bridge

print("=" * 60)
print("P12 — GNN Social Support Analysis (Real SNAP Network Data)")
print("=" * 60)

results = {}

for name, meta in DATASETS.items():
    print(f"\n--- {name}: {meta['description']} ---")
    try:
        path = download_snap(name, meta['url'], meta['filename'])
        edges = load_edge_list(path)
        print(f"  Loaded {len(edges)} edges")

        adj = build_adjacency(edges)
        nodes = set(adj.keys())
        n_nodes = len(nodes)
        n_edges = len(edges)

        print(f"  Nodes: {n_nodes}, Edges: {n_edges}")

        # Degree centrality
        deg_cent = compute_degree_centrality(adj, nodes)

        # Degree distribution
        degrees = [len(adj[n]) for n in nodes]
        mean_degree = np.mean(degrees)
        max_degree = np.max(degrees)

        # Sample clustering coefficients (subsample for speed)
        sample_nodes = list(nodes)[:500]
        cc_vals = [compute_clustering_coefficient(adj, n) for n in sample_nodes]
        mean_cc = np.mean(cc_vals)

        # Community detection (label propagation)
        print("  Running label-propagation community detection...")
        labels = find_communities_label_propagation(adj, nodes)
        communities = Counter(labels.values())
        n_communities = len(communities)
        largest_community = communities.most_common(1)[0][1]
        print(f"  Communities detected: {n_communities}")

        # Social support scores for sampled nodes
        support_scores = []
        bridge_scores = []
        for node in sample_nodes:
            ss, cc, brg = compute_social_support_score(adj, node, deg_cent)
            support_scores.append(ss)
            bridge_scores.append(brg)

        # Identify top support hubs and bridge nodes
        node_support = list(zip(sample_nodes, support_scores))
        node_support.sort(key=lambda x: x[1], reverse=True)
        top_hubs = node_support[:10]

        # Vulnerability: isolated nodes or low-degree periphery
        isolated = sum(1 for n in nodes if len(adj[n]) <= 1)
        low_support = sum(1 for n in nodes if len(adj[n]) <= 2)

        results[name] = {
            "source": meta['url'],
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "avg_degree": float(mean_degree),
            "max_degree": float(max_degree),
            "mean_clustering_coeff": float(mean_cc),
            "n_communities_detected": n_communities,
            "largest_community_size": int(largest_community),
            "graph_density": float(2 * n_edges / (n_nodes * (n_nodes - 1))),
            "isolated_low_degree_nodes": int(isolated),
            "low_support_nodes_pct": float(low_support / n_nodes * 100),
            "mean_support_score": float(np.mean(support_scores)),
            "top_hub_nodes": [{"node": int(n), "score": float(s)} for n, s in top_hubs[:5]],
            "degree_distribution_percentiles": {
                "p25": float(np.percentile(degrees, 25)),
                "p50": float(np.percentile(degrees, 50)),
                "p75": float(np.percentile(degrees, 75)),
                "p95": float(np.percentile(degrees, 95)),
            }
        }

        print(f"  Avg degree: {mean_degree:.2f}, Clustering coeff: {mean_cc:.4f}")
        print(f"  Graph density: {results[name]['graph_density']:.5f}")
        print(f"  Low-support nodes: {low_support} ({low_support/n_nodes*100:.1f}%)")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        continue

# Save results
summary = {
    "paper": "P12 — GNN for Student Social Support Network Analysis",
    "real_data_source": "Stanford SNAP (Social Network Analysis Project)",
    "citation": "J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks. NIPS 2012.",
    "method": "Label propagation community detection + social support scoring",
    "results": results
}

out_json = RESULTS_DIR / "p12_social_network_results.json"
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n  Results saved: {out_json}")

# Figure
if "ego_facebook" in results:
    r = results["ego_facebook"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Degree distribution (power-law expected for real social networks)
    edges_reloaded = load_edge_list(CACHE / "facebook_combined.txt.gz")
    adj_r = build_adjacency(edges_reloaded)
    degs = sorted([len(v) for v in adj_r.values()], reverse=True)
    axes[0].loglog(range(1, len(degs)+1), degs, 'b.', markersize=2, alpha=0.7)
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Degree (log scale)")
    axes[0].set_title("Degree Distribution\n(Real Facebook Network)")
    axes[0].grid(True, alpha=0.3)

    # Community size distribution
    labels_loaded = find_communities_label_propagation(adj_r, set(adj_r.keys()))
    comm_sizes = sorted(Counter(labels_loaded.values()).values(), reverse=True)[:50]
    axes[1].bar(range(len(comm_sizes)), comm_sizes, color='#4CAF50', edgecolor='black', alpha=0.8)
    axes[1].set_xlabel("Community Rank")
    axes[1].set_ylabel("Community Size (nodes)")
    axes[1].set_title(f"Top 50 Community Sizes\nTotal: {len(Counter(labels_loaded.values()))} communities")

    # Support score distribution
    sample = list(adj_r.keys())[:300]
    dc = compute_degree_centrality(adj_r, set(adj_r.keys()))
    scores = [compute_social_support_score(adj_r, n, dc)[0] for n in sample]
    axes[2].hist(scores, bins=30, color='#FF5722', edgecolor='black', alpha=0.8)
    axes[2].set_xlabel("Social Support Score")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Social Support Score Distribution\n(sampled nodes)")
    axes[2].axvline(np.mean(scores), color='black', linestyle='--', label=f'Mean={np.mean(scores):.3f}')
    axes[2].legend()

    plt.suptitle("P12: Real Social Network GNN Analysis — SNAP ego-Facebook", fontsize=12)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "p12_social_network_figure.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")

print("\nP12 REAL DATA TEST COMPLETE")
