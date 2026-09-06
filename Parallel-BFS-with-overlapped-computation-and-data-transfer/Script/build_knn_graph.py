#!/usr/bin/env python3
import numpy as np
import faiss
import csv
import argparse

def read_fvecs(filename):
    """Reads fvecs file format used in SIFT datasets."""
    with open(filename, "rb") as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)
        data = data.reshape(-1, d + 1)
        return data[:, 1:].astype('float32')

def build_knn_graph(fvec_path, N=10000, R=10, output_csv="knn_graph.csv"):
    """Build a KNN graph and export adjacency list to CSV."""
    # Step 1: Load dataset
    X = read_fvecs(fvec_path)
    X = X[:N]
    d = X.shape[1]

    # Step 2: Build FAISS index
    index = faiss.IndexFlatL2(d)
    index.add(X)

    # Step 3: Search for R+1 neighbors (first is the point itself)
    D, I = index.search(X, R + 1)
    neighbors = I[:, 1:]     # remove self
    distances = D[:, 1:]     # remove self distance (0)

    # Step 4: Write adjacency list (no coordinates)
    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        header = ["node_id"] + [f"n{j}" for j in range(R)]
        writer.writerow(header)
        for i in range(N):
            row = [i] + list(neighbors[i])
            writer.writerow(row)

    # Step 5: Compute BFS start node
    total_dist = np.sum(distances, axis=1)
    best_node = int(np.argmax(total_dist))

    print(f"KNN adjacency graph written to {output_csv}")
    print(f"Nodes: {N}, Degree: {R}, Dimensions: {d}")
    print(f"\n[BFS Start Node]")
    print(f"Node ID: {best_node}")

    return best_node

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build adjacency list from KNN graph (.fvecs dataset)")
    parser.add_argument("fvec_path", help="Path to SIFT .fvecs file (e.g. sift_base.fvecs)")
    parser.add_argument("-N", type=int, default=10000, help="Number of points to use")
    parser.add_argument("-R", type=int, default=10, help="Degree (number of neighbors)")
    parser.add_argument("-o", "--output", default="knn_graph.csv", help="Output CSV file name")

    args = parser.parse_args()
    build_knn_graph(args.fvec_path, args.N, args.R, args.output)