def get_test_config():
    config = {
        'batch_size': 128,
        'datasets': ['cifar10', 'cifar100', 'imagenet'],
        'threat_models': ['Linf', 'L2', 'corruptions'],
        'data_dir': 'data',
        'imagenet_data_dir': '/tmldata1/andriush/imagenet/val',
        'model_dir': 'models',
        'device': 'cuda:0'
    }
    return config
