# preprocess the twitter graph and put preprocessed graph into output dir
python3 tools/preprocess.py twitter output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/twitter_multi_gpu.ini info