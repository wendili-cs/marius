import argparse
import os
import pathlib
from argparse import RawDescriptionHelpFormatter

import numpy as np
import pandas as pd
import torch

import marius as m
from marius.tools.configuration.constants import PathConstants
from marius.tools.prediction.link_prediction import infer_lp
from marius.tools.prediction.node_classification import infer_nc
from marius.tools.prediction.page_rank import infer_pr
from marius.tools.preprocess.converters.partitioners.torch_partitioner import partition_edges
from marius.tools.preprocess.converters.readers.pandas_readers import PandasDelimitedFileReader
from marius.tools.marius_predict import set_args, get_metrics, get_input_file_storage, get_nbrs_config, str2bool
# from marius.tools.preprocess.converters.torch_converter import (
#     SUPPORTED_DELIM_FORMATS,
#     apply_mapping1d,
#     apply_mapping_edges,
#     dataframe_to_tensor,
# )

def run_pagerank(args):
    config = m.config.loadConfig(args.config)
    metrics = get_metrics(config, args)

    model_dir_path = pathlib.Path(config.storage.model_dir)

    graph_storage: m.storage.GraphModelStorage = m.storage.load_storage(args.config, train=False)

    if args.input_file != "":
        input_storage = get_input_file_storage(config, args)

        if config.model.learning_task == m.config.LearningTask.PAGE_RANK:
            graph_storage.storage_ptrs.edges = input_storage
        else:
            raise RuntimeError("Unsupported learning task for page rank.")
    else:
        graph_storage.setTestSet()

    output_dir = args.output_dir
    if output_dir == "":
        output_dir = config.storage.model_dir

    nbrs = get_nbrs_config(config, args)

    if config.model.learning_task == m.config.LearningTask.PAGE_RANK:
        infer_pr(
            graph_storage=graph_storage,
            output_dir=output_dir,
            metrics=metrics,
            save_labels=args.save_labels,
            batch_size=args.batch_size,
            num_nbrs=nbrs,
        )
    else:
        raise RuntimeError("Unsupported learning task for page rank.")

    print("Results output to: {}".format(output_dir))

def main():
    parser = set_args()
    args = parser.parse_args()
    run_pagerank(args)