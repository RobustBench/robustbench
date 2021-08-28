def get_test_config():
    config = {
        'batch_size': 128,
        'datasets': ['imagenet'],  #['cifar10', 'cifar100', 'imagenet'],
        'threat_models': ['Linf', 'L2', 'corruptions'],
        'data_dir': 'data',
        'imagenet_data_dir': '/tmldata1/andriush/imagenet',
        'model_dir': 'models',
        'device': 'cpu',  # 'cuda:0'
    }
    return config
