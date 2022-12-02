import marius as m

def infer_pr(
    graph_storage: m.storage.GraphModelStorage,
    output_dir: str,
    metrics: list = None,
    save_labels: bool = False,
    batch_size: int = 1000,
    num_nbrs: list = None,
):
    print("this is infer_pr function")

    dataloader = m.data.DataLoader(
        graph_storage=graph_storage,
        neg_sampler=None,
        nbr_sampler=None,
        batch_size=batch_size,
        learning_task="pr",
    )

    print("this is the end of infer_pr function")