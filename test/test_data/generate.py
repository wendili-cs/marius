import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Generate Datasets.')
    parser.add_argument('num_nodes', metavar='nodes', type=int, help='Number of nodes')
    parser.add_argument('num_rels', metavar='relations', type=int, help='Number of Relations')
    parser.add_argument('num_train_edges', metavar='train', type=int, help='Number of Training Edges')
    parser.add_argument('num_valid_edges', metavar='valid', type=int, help='Number of Validation Edges')
    parser.add_argument('num_test_edges', metavar='test', type=int, help='Number of Test Edges')
    args = parser.parse_args()

    num_nodes = args.num_nodes
    num_rels = args.num_rels
    num_train_edges = args.num_train_edges
    num_valid_edges = args.num_valid_edges
    num_test_edges = args.num_test_edges

    edge_ids = np.arange(num_nodes * num_nodes)
    np.random.shuffle(edge_ids)
    train_edge_ids = edge_ids[:num_train_edges]
    valid_edge_ids = edge_ids[num_train_edges:num_train_edges + num_valid_edges]
    test_edge_ids = edge_ids[num_train_edges + num_valid_edges : num_train_edges + num_valid_edges + num_test_edges]

    src_train_id = np.floor_divide(train_edge_ids, num_nodes)
    dst_train_id = np.remainder(train_edge_ids, num_nodes)
    rel_train_id = np.random.randint(0, num_rels, len(src_train_id))

    src_valid_id = np.floor_divide(valid_edge_ids, num_nodes)
    dst_valid_id = np.remainder(valid_edge_ids, num_nodes)
    rel_valid_id = np.random.randint(0, num_rels, len(src_valid_id))

    src_test_id = np.floor_divide(test_edge_ids, num_nodes)
    dst_test_id = np.remainder(test_edge_ids, num_nodes)
    rel_test_id = np.random.randint(0, num_rels, len(src_test_id))

    train_edges_np = np.stack([src_train_id, rel_train_id, dst_train_id]).T
    valid_edges_np = np.stack([src_valid_id, rel_valid_id, dst_valid_id]).T
    test_edges_np = np.stack([src_test_id, rel_test_id, dst_test_id]).T

    train_edges_df = pd.DataFrame(data=train_edges_np)
    valid_edges_df = pd.DataFrame(data=valid_edges_np)
    test_edges_df = pd.DataFrame(data=test_edges_np)

    # write to csv
    train_edges_df.to_csv("train_edges.txt", " ", header=False, index=False)
    valid_edges_df.to_csv("valid_edges.txt", " ", header=False, index=False)
    test_edges_df.to_csv("test_edges.txt", " ", header=False, index=False)

    # convert to tensor blob
    with open("train_edges.pt", "wb") as f:
        f.write(bytes(train_edges_np))
    with open("valid_edges.pt", "wb") as f:
        f.write(bytes(valid_edges_np))
    with open("test_edges.pt", "wb") as f:
        f.write(bytes(test_edges_np))


if __name__ == '__main__':
    main()