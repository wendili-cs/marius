import marius as m
import numpy as np
import torch

def get_outdeg(
    model: m.nn.Model,
    graph_storage: m.storage.GraphModelStorage,
):
    nbr_sampler = None
    num_nbrs = [-1]
    if num_nbrs is not None:
        nbr_sampler = m.samplers.LayeredNeighborSampler(graph_storage, num_nbrs)

    dataloader = m.data.DataLoader(
        graph_storage=graph_storage,
        neg_sampler=None,
        nbr_sampler=nbr_sampler,
        batch_size=50000,
        learning_task="lp",
    )

    # total_nodes = 14541
    dataloader.initializeBatches()

    while dataloader.hasNextBatch():
        batch = dataloader.getBatch(model.device)
        
        # total_nodes = int(batch.dense_graph.node_ids.shape) # int(batch.dense_graph.node_ids[-1].cpu().detach())
        # nodes_out_degs = np.zeros([total_nodes])

        # in_offsets = batch.dense_graph.in_offsets.cpu()
        out_offsets = batch.dense_graph.out_offsets.cpu()
        # tmp_in_offsets = torch.cat((in_offsets, torch.tensor([batch.dense_graph.in_neighbors.size(0)])))
        tmp_out_offsets = torch.cat((out_offsets, torch.tensor([batch.dense_graph.out_neighbors.size(0)])))
        out_num_neighbors = tmp_out_offsets.narrow(0, 1, out_offsets.size(0)) - tmp_out_offsets.narrow(0, 0, out_offsets.size(0))
        # in_num_neighbors = tmp_in_offsets.narrow(0, 1, in_offsets.size(0)) - tmp_out_offsets.narrow(0, 0, in_offsets.size(0))
        # print("out_num_neighbors shape", out_num_neighbors.shape)
        assert not dataloader.hasNextBatch()
    return np.asarray(out_num_neighbors.cpu().detach())
        


def infer_pr(
    model: m.nn.Model,
    graph_storage: m.storage.GraphModelStorage,
    output_dir: str,
    metrics: list = None,
    save_labels: bool = False,
    batch_size: int = 1000,
    num_nbrs: list = None,
):
    # batch_size = 10000
    print("batch_size", batch_size)
    print("this is infer_pr function")

    nbr_sampler = None
    num_nbrs = [-1]
    if num_nbrs is not None:
        nbr_sampler = m.samplers.LayeredNeighborSampler(graph_storage, num_nbrs)

    dataloader = m.data.DataLoader(
        graph_storage=graph_storage,
        neg_sampler=None,
        nbr_sampler=nbr_sampler,
        batch_size=batch_size,
        learning_task="lp",
    )

    
    # dataloader.initializeBatches()
    # print(model.device)

    total_nodes = 14541
    
    # while dataloader.hasNextBatch():
    #     batch = dataloader.getBatch(model.device)
    #     total_nodes = max(total_nodes, batch.unique_node_indices[-1] + 1)
    print("total_nodes", total_nodes)

    
    
    

    nodes_out_degs = get_outdeg(model, graph_storage)
    print("nodes_out_degs shape:", nodes_out_degs.shape)

    # TODO: this is not necessary after the out degree calculation is correct
    tmp = np.zeros([total_nodes]) + 1e-5
    tmp[:len(nodes_out_degs)] = nodes_out_degs
    nodes_out_degs = tmp

    

    cur_contrib = np.ones([total_nodes])
    for epoch in range(10):
        new_contrib = np.zeros([total_nodes])
        dataloader.initializeBatches()
        while dataloader.hasNextBatch():
            batch = dataloader.getBatch(model.device)
            # print(batch.edges)
            ebatch = batch.edges.cpu().numpy()
            new_contrib[ebatch[:, 2]] += cur_contrib[ebatch[:, 0]]/(nodes_out_degs[ebatch[:, 0]]+1e-5)
        cur_contrib = new_contrib*0.85 + 0.15
            
            # # print("node_ids[-1]", batch.dense_graph.node_ids[-1].cpu().detach())
            # # batch.dense_graph.performMap()
            # # print("getNumNeighbors shape", batch.dense_graph.getNumNeighbors().cpu().shape)

            # # print("batch.dense_graph.node_ids:", batch.dense_graph.node_ids)
            # # print("batch.dense_graph.out_neighbors:", batch.dense_graph.out_neighbors)
            # # print("batch.dense_graph.in_num_neighbors:", batch.dense_graph.in_num_neighbors)
            # # print("batch.dense_graph.in_offsets:", batch.dense_graph.in_offsets)
            # # print("batch.dense_graph.out_offsets:", batch.dense_graph.out_offsets)
            
            # in_offsets = batch.dense_graph.in_offsets.cpu()
            # out_offsets = batch.dense_graph.out_offsets.cpu()
            # tmp_in_offsets = torch.cat((in_offsets, torch.tensor([batch.dense_graph.in_neighbors.size(0)])))
            # tmp_out_offsets = torch.cat((out_offsets, torch.tensor([batch.dense_graph.out_neighbors.size(0)])))
            # out_num_neighbors = tmp_out_offsets.narrow(0, 1, out_offsets.size(0)) - tmp_out_offsets.narrow(0, 0, out_offsets.size(0))
            # # in_num_neighbors = tmp_in_offsets.narrow(0, 1, in_offsets.size(0)) - tmp_out_offsets.narrow(0, 0, in_offsets.size(0))
            # # print("out_num_neighbors shape", out_num_neighbors.shape)
                
            
            
            # # print(type(batch), batch.unique_node_indices.shape)
            # node_indices = batch.unique_node_indices.cpu()
            # print("node_ids shape:", batch.dense_graph.node_ids.shape)
            # print("unique_node_indices shape:", node_indices.shape)
            # out_degs = model.forward_pr(batch).cpu()
            # # print("out_degs", torch.sum(out_degs))
            # contribs = page_ranks[node_indices]/out_degs
            # page_ranks_new[node_indices] = 0.85*contribs + 0.15

            # assert not dataloader.hasNextBatch()
        print("page_ranks", cur_contrib)
        print("numner of numbers", len(set(cur_contrib)))
    

    print("this is the end of infer_pr function")
