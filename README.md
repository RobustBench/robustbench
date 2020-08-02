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
We plan to extend the benchmark to other threat models in the future going beyond Lp-bounded perturbations.

`AdvBench` consists from two parts: a website with the leaderboard, and a collection of robust models `model zoo`
The tutorial below shows what one can do with the `model_zoo`.



## AdvBench tutorial



## Adding a new model
In order to add a new model, submit a pull request where you specify the following:
- `model_claims/NameYearFirst-word-of-the-title.json`: follow the convention of the existing json-files to specify the information to be displayed on the website. 
Here is an example from `model_info/Rice2020Overfitting.json`:
```python
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
- `model_zoo/models.py`: add your model definition as a new class, for example:
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
And also add your gdrive-id:
```
    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNet(34, 20),
        'gdrive_id': '1vC_Twazji7lBjeMQvAD9uEQxi9Nx2oG-',
    })
```



## Testing
Run the following scripts to test the existing models
- `python tests/test_clean_acc_fast.py`
- `python tests/test_clean_acc_jsons.py`


