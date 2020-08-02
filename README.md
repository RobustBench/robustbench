# AdvBench



## Main idea
The goal of `AdvBench` is to systematically benchmark adversarial robustness to be able to track the real progress in 
the field. 

We start from benchmarking the Linf-robustness since it is the most studied setting in the literature. 
There are numerous interesting applications of Linf-robustness that span 
transfer learning ([Salman et al. (2020)](https://arxiv.org/abs/2007.08489), [Utrera et al. (2020)](https://arxiv.org/abs/2007.05869)), 
interpretability ([Tsipras et al. (2018)](https://arxiv.org/abs/1805.12152), [Kaur et al. (2019)](https://arxiv.org/abs/1910.08640), [Engstrom et al. (2019)](https://arxiv.org/abs/1906.00945))
generalization ([Xie et al. (2019)](https://arxiv.org/abs/1911.09665), [Zhu et al. (2019)](https://arxiv.org/abs/1909.11764), [Bochkovskiy et al. (2020)](https://arxiv.org/abs/2004.10934)), 
security ([Tramèr et al. (2018)](https://arxiv.org/abs/1811.03194), [Saadatpanah et al. (2019)](https://arxiv.org/abs/1906.07153)). 
We plan to extend the benchmark to other threat models in the future: first to other Lp-norms and then to more general perturbation sets.

`AdvBench` consists from two parts: a website with the leaderboard, and a collection of robust models **Model Zoo**.
The tutorial below shows how one can use the **Model Zoo**.



## AdvBench tutorial
First install `AdvBench`:
```bash
pip install -r requirements.txt
git clone https://github.com/fra31/advbench && cd advbench
```
TODO: currently the repo is not publicly visible. TODO: and pip would be better.

Main points:
```python
from data import load_cifar10
from utils import load_model, clean_accuracy

x_test, y_test = load_cifar10(n_examples=100)
model = load_model(model_name='Carmon2019Unlabeled').cuda().eval()

acc = clean_accuracy(model=model, x=x_test, y=y_test, batch_size=128)
print('Clean accuracy: {:.2%}'.format(acc))

# TODO: add AutoAttack eval (the old version is fine for now)
```



## Adding a new model
In order to add a new model, submit a pull request where you specify the claim, model definition, and model checkpoint:

- **Claim**: `model_claims/<Name><Year><FirstWordOfTheTitle>.json`: follow the convention of the existing json-files to specify the information to be displayed on the website. 
Here is an example from `model_info/Rice2020Overfitting.json`:
```json
{
  "link": "https://arxiv.org/abs/2002.11569",
  "name": "Overfitting in adversarially robust deep learning",
  "authors": "Leslie Rice, Eric Wong, J. Zico Kolter",
  "additional_data": false,
  "number_forward_passes": 1,
  "dataset": "cifar10",
  "venue": "ICML 2020",
  "architecture": "WideResNet-34-20",
  "eps": "8/255",
  "clean_acc": "85.34",
  "reported": "58",
  "AA": "53.60",
  "AA+": "53.35"
}
```

- **Model definition**: `model_zoo/models.py`: add your model definition as a new class. For standard architectures like `WideResNet` consider
 inheriting the class defined in `wide_resnet.py`, `resnet.py`, `resnetv2.py`. For example:
```python
class Rice2020OverfittingNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Rice2020OverfittingNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=False)
        self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.2471, 0.2435, 0.2616]).float().view(3, 1, 1).cuda()

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNet, self).forward(x)
```

- **Model checkpoint**: `model_zoo/models.py`: And also add your model entry in `model_dicts` which should also contain 
the *Google Drive ID* with your pytorch model so that it can be downloaded automatically:
```
    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNet(34, 20),
        'gdrive_id': '1vC_Twazji7lBjeMQvAD9uEQxi9Nx2oG-',
    })
```



## Testing
Run the following scripts to test the existing models from the **Model Zoo**:
- `python tests/test_clean_acc_fast.py`: fast testing on 200 examples that clean accuracy exceeds some threshold.
- `python tests/test_clean_acc_jsons.py`: testing on 10'000 examples that clean accuracy of the models matches the one 
mentioned in the `model_info` jsons.

Note that you can specify some configurations like `batch_size`, `data_dir`, `model_dir` in `config.py` either as 
default parameters or as parameters from the command line.


## Citation
Our white paper about `AdvBench` is in preparation. Stay tuned!

