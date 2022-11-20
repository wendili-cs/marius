import marius as m

def infer_pr(
    output_dir: str,
    metrics: list = None,
    save_labels: bool = False,
    batch_size: int = 1000,
    num_nbrs: list = None,
):
    print("this is infer_pr function")