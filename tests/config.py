def get_test_config():
    config = {
        'batch_size': 500,
        'data_dir': '../robustbench_experiments/data',
        'model_dir': '../robustbench_experiments/models',
        'device': 'cuda:2'
    }
    return config
