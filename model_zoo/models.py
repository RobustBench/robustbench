from model_zoo.model_base import WideResNet


class Carmon2019UnlabeledNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Carmon2019UnlabeledNet, self).__init__(depth=depth, widen_factor=widen_factor)


model_dicts = {
    'carmon2019unlabeled': {
        'model': Carmon2019UnlabeledNet(28, 10),
        'gdrive_id': '15tUx-gkZMYx7BfEOw1GY5OKC-jECIsPQ'
    }
}
