# 📊 AdvBench: tracking the progress in adversarial robustness



## Main idea
The goal of **`AdvBench`** is to systematically track the *real* progress in adversarial robustness. 
There are already [more than 2'000 papers](https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html) 
on this topic, but it is still unclear which approaches *really* work and which only lead to [overestimated robustness](https://arxiv.org/abs/1802.00420).
We start from benchmarking the Linf-robustness since it is the most studied setting in the literature. 
We plan to extend the benchmark to other threat models in the future: first to other Lp-norms and then to more general perturbation sets 
(Wasserstein perturbations, common corruptions, etc).

Robustness evaluation *in general* is not straightforward and requires adaptive attacks ([Tramer et al., (2020)](https://arxiv.org/abs/2002.08347)).
Thus, in order to establish a reliable *standardized* benchmark, we need to impose some restrictions on the defenses we consider.
In particular, **we accept only defenses that are (1) have non-zero gradient almost everywhere wrt the inputs, (2) have a fully deterministic forward pass (i.e. no randomness) that
(3) does not have an optimization loop.** Usually, defenses that violate these 3 principles only make gradient-based attacks 
harder but do not substantially improve robustness ([Carlini et al., (2019)](https://arxiv.org/abs/1902.06705)) except those
that can present concrete provable guarantees (e.g. [Cohen et al., (2019)](https://arxiv.org/abs/1902.02918)).

**`AdvBench`** consists of two parts: 
- a website with the leaderboard based on many recent papers (plots below 👇)
- a collection of the most robust models, **Model Zoo**, which are very easy to use for any application (see the tutorial below after FAQ 👇)

<p align="center"><img src="images/aa_robustness_vs_venues.png" height="275">  <img src="images/aa_robustness_vs_years.png" height="275"></p>
<p align="center"><img src="images/aa_robustness_vs_reported.png" height="260">  <img src="images/aa_robustness_vs_clean.png" height="260"></p>




## FAQ
**Q**: Wait, how is it different from [robust-ml.org](https://www.robust-ml.org/)? 🤔 \
**A**: [robust-ml.org](https://www.robust-ml.org/) focuses on *adaptive* evaluations, but we provide a **standardized benchmark**. Adaptive evaluations
are great (e.g., see [Tramer et al., 2020](https://arxiv.org/abs/2002.08347)) but very time consuming and not standardized.

**Q**: How is it related to libraries like `foolbox` / `cleverhans` / `advertorch`? 🤔 \
**A**: These libraries provide implementations of different *attacks*. Besides the standardized benchmark, **`AdvBench`** 
additionally provides a repository of the most robust models. So you can start using the
robust models in one line of code (see the tutorial below 👇).

**Q**: I've heard that Lp-robustness is boring. Why would you even evaluate Lp-robustness in 2020? 🤔 \
**A**: There are numerous interesting applications of Lp-robustness that span 
transfer learning ([Salman et al. (2020)](https://arxiv.org/abs/2007.08489), [Utrera et al. (2020)](https://arxiv.org/abs/2007.05869)), 
interpretability ([Tsipras et al. (2018)](https://arxiv.org/abs/1805.12152), [Kaur et al. (2019)](https://arxiv.org/abs/1910.08640), [Engstrom et al. (2019)](https://arxiv.org/abs/1906.00945)),
security ([Tramèr et al. (2018)](https://arxiv.org/abs/1811.03194), [Saadatpanah et al. (2019)](https://arxiv.org/abs/1906.07153)),
generalization ([Xie et al. (2019)](https://arxiv.org/abs/1911.09665), [Zhu et al. (2019)](https://arxiv.org/abs/1909.11764), [Bochkovskiy et al. (2020)](https://arxiv.org/abs/2004.10934)), 
robustness to unseen perturbations ([Xie et al. (2019)](https://arxiv.org/abs/1911.09665), [Kang et al. (2019)](https://arxiv.org/abs/1905.01034)),
stabilization of GAN training ([Zhong et al. (2020)](https://arxiv.org/abs/2008.03364)).

**Q**: Is this benchmark only focused on Lp-robustness? 🤔 \
**A**: Not at all! Lp-robustness is the most well-studied area, so we focus on it first. However, in the future, we plan 
to extend the benchmark to other perturbations sets beyond Lp-balls.

**Q**: What if I have a better attack than the one used in this benchmark? 🤔 \
**A**: We will be happy to add a better attack or any adaptive evaluation that would complement our default standardized attacks.




## Model Zoo: quick tour
The goal of our **Model Zoo** is to simplify the usage of robust models as much as possible.
Check out our Colab notebook here 👉 [AdvBench: quick start](https://colab.research.google.com/drive/1JrOOMSkwszNE31VgcHD94htxizFOam7C) 
for a quick introduction. It is also summarized below 👇.

First, install **`AdvBench`**:
```bash
pip install git+https://github.com/AdvBench/advbench
```

Now let's try to load CIFAR-10 and the most robust CIFAR-10 model from [Carmon2019Unlabeled](https://arxiv.org/abs/1905.13736) 
that achieves 59.53% robust accuracy evaluated with AA under eps=8/255:
```python
from advbench.data import load_cifar10
x_test, y_test = load_cifar10(n_examples=50)

from advbench.utils import load_model
model = load_model(model_name='Carmon2019Unlabeled', norm='Linf')
```

Let's try to evaluate the robustness of this model. We can use any favourite library for this. For example, [FoolBox](https://github.com/bethgelab/foolbox)
implements many different attacks. We can start from a simple PGD attack:
```python
!pip install -q foolbox
import foolbox as fb
fmodel = fb.PyTorchModel(model, bounds=(0, 1))

_, advs, success = fb.attacks.LinfPGD()(fmodel, x_test, y_test, epsilons=[8/255])
print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
```
```
>>> Robust accuracy: 58.0%
```
Wonderful! Can we do better with a more accurate attack?

Let's try to evaluate its robustness with a cheap version [AutoAttack](https://arxiv.org/abs/2003.01690) from ICML 2020 with 2/4 attacks (only APGD-CE and APGD-DLR):
```python
!pip install -q git+https://github.com/fra31/auto-attack
from autoattack import AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test, y_test)
```
```
>>> initial accuracy: 92.00%
>>> apgd-ce - 1/1 - 19 out of 46 successfully perturbed
>>> robust accuracy after APGD-CE: 54.00% (total time 10.3 s)
>>> apgd-dlr - 1/1 - 1 out of 27 successfully perturbed
>>> robust accuracy after APGD-DLR: 52.00% (total time 17.0 s)
>>> max Linf perturbation: 0.03137, nan in tensor: 0, max: 1.00000, min: 0.00000
>>> robust accuracy: 52.00%
```
Note that for our standardized evaluation of Linf-robustness we use the *full* version of AutoAttack which is slower but 
more accurate (for that just use `adversary = AutoAttack(model, norm='Linf', eps=8/255)`).

What about other types of perturbations? Is Lp-robustness useful there? We can evaluate the available models on more general perturbations. 
For example, let's take images corrupted by fog perturbations from CIFAR-10-C with the highest level of severity (5). 
Are different Linf robust models perform better on them?
```python
from advbench.data import load_cifar10c
from advbench.utils import clean_accuracy

corruptions = ['fog']
x_test, y_test = load_cifar10c(n_examples=1000, corruptions=corruptions, severity=5)   

for model_name in ['Natural', 'Engstrom2019Robustness', 'Rice2020Overfitting', 'Carmon2019Unlabeled']:
  model = load_model(model_name)
  acc = clean_accuracy(model, x_test, y_test)
  print('Model: {}, CIFAR-10-C accuracy: {:.1%}'.format(model_name, acc))
``` 
```
>>> Model: Natural, CIFAR-10-C accuracy: 74.4%
>>> Model: Engstrom2019Robustness, CIFAR-10-C accuracy: 38.8%
>>> Model: Rice2020Overfitting, CIFAR-10-C accuracy: 22.0%
>>> Model: Carmon2019Unlabeled, CIFAR-10-C accuracy: 31.1%
```
As we can see, **all** these Linf robust models perform considerably worse than the natural model on this type of corruptions. 
This curious phenomenon was first noticed in [Adversarial Examples Are a Natural Consequence of Test Error in Noise](https://arxiv.org/abs/1901.10513) 
and explained from the frequency perspective in [A Fourier Perspective on Model Robustness in Computer Vision](https://arxiv.org/abs/1906.08988). 

However, on average adversarial training *does* help on CIFAR-10-C. One can check this easily by loading all types of corruptions 
via `load_cifar10c(n_examples=1000, severity=5)`, and repeating evaluation on them.



## Model Zoo
In order to use a model, you just need to know its ID, e.g. **Carmon2019Unlabeled**, and to run: 
```python 
model = load_model(model_name='Carmon2019Unlabeled', norm='Linf)
```
which automatically downloads the model (all models are defined in `model_zoo/models.py`).

You can find all available model IDs in the table below (note that the full leaderboard contains more models): 

### Linf
| # | Model ID | Paper | Clean accuracy | Robust accuracy | Architecture | Venue |
|:---:|---|---|:---:|:---:|:---:|:---:|
| <sub>**1**</sub> | <sub>**Carmon2019Unlabeled**</sub> | <sub>*[Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/abs/1905.13736)*</sub> | <sub>89.69%</sub> | <sub>59.53%</sub> | <sub>WideResNet-28-10</sub> | <sub>NeurIPS 2019</sub> |
| <sub>**2**</sub> | <sub>**Sehwag2020Hydra**</sub> | <sub>*[HYDRA: Pruning Adversarially Robust Neural Networks](https://arxiv.org/abs/2002.10509)*</sub> | <sub>88.98%</sub> | <sub>57.14%</sub> | <sub>WideResNet-28-10</sub> | <sub>Unpublished</sub> |
| <sub>**3**</sub> | <sub>**Wang2020Improving**</sub> | <sub>*[Improving Adversarial Robustness Requires Revisiting Misclassified Examples](https://openreview.net/forum?id=rklOg6EFwS)*</sub> | <sub>87.50%</sub> | <sub>56.29%</sub> | <sub>WideResNet-28-10</sub> | <sub>ICLR 2020</sub> |
| <sub>**4**</sub> | <sub>**Hendrycks2019Using**</sub> | <sub>*[Using Pre-Training Can Improve Model Robustness and Uncertainty](https://arxiv.org/abs/1901.09960)*</sub> | <sub>87.11%</sub> | <sub>54.92%</sub> | <sub>WideResNet-28-10</sub> | <sub>ICML 2019</sub> |
| <sub>**5**</sub> | <sub>**Pang2020Boosting**</sub> | <sub>*[Boosting Adversarial Training with Hypersphere Embedding](https://arxiv.org/abs/2002.08619)*</sub> | <sub>85.14%</sub> | <sub>53.74%</sub> | <sub>WideResNet-34-20</sub> | <sub>Unpublished</sub> |
| <sub>**6**</sub> | <sub>**Zhang2020Attacks**</sub> | <sub>*[Attacks Which Do Not Kill Training Make Adversarial Learning Stronger](https://arxiv.org/abs/2002.11242)*</sub> | <sub>84.52%</sub> | <sub>53.51%</sub> | <sub>WideResNet-34-10</sub> | <sub>ICML 2020</sub> |
| <sub>**7**</sub> | <sub>**Rice2020Overfitting**</sub> | <sub>*[Overfitting in adversarially robust deep learning](https://arxiv.org/abs/2002.11569)*</sub> | <sub>85.34%</sub> | <sub>53.42%</sub> | <sub>WideResNet-34-20</sub> | <sub>ICML 2020</sub> |
| <sub>**8**</sub> | <sub>**Huang2020Self**</sub> | <sub>*[Self-Adaptive Training: beyond Empirical Risk Minimization](https://arxiv.org/abs/2002.10319)*</sub> | <sub>83.48%</sub> | <sub>53.34%</sub> | <sub>WideResNet-34-10</sub> | <sub>Unpublished</sub> |
| <sub>**9**</sub> | <sub>**Zhang2019Theoretically**</sub> | <sub>*[Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/abs/1901.08573)*</sub> | <sub>84.92%</sub> | <sub>53.08%</sub> | <sub>WideResNet-34-10</sub> | <sub>ICML 2019</sub> |
| <sub>**10**</sub> | <sub>**Chen2020Adversarial**</sub> | <sub>*[Adversarial Robustness: From Self-Supervised Pre-Training to Fine-Tuning](https://arxiv.org/abs/2003.12862)*</sub> | <sub>86.04%</sub> | <sub>51.56%</sub> | <sub>ResNet-50 <br/> (3x ensemble)</sub> | <sub>CVPR 2020</sub> |
| <sub>**11**</sub> | <sub>**Engstrom2019Robustness**</sub> | <sub>*[Robustness library](https://github.com/MadryLab/robustness)*</sub> | <sub>87.03%</sub> | <sub>49.25%</sub> | <sub>ResNet-50</sub> | <sub>Unpublished</sub> |
| <sub>**12**</sub> | <sub>**Zhang2019You**</sub> | <sub>*[You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle](https://arxiv.org/abs/1905.00877)*</sub> | <sub>87.20%</sub> | <sub>44.83%</sub> | <sub>WideResNet-34-10</sub> | <sub>NeurIPS 2019</sub> |
| <sub>**13**</sub> | <sub>**Wong2020Fast**</sub> | <sub>*[Fast is better than free: Revisiting adversarial training](https://arxiv.org/abs/2001.03994)*</sub> | <sub>83.34%</sub> | <sub>43.21%</sub> | <sub>ResNet-18</sub> | <sub>ICLR 2020</sub> |
| <sub>**14**</sub> | <sub>**Ding2020MMA**</sub> | <sub>*[MMA Training: Direct Input Space Margin Maximization through Adversarial Training](https://openreview.net/forum?id=HkeryxBtPB)*</sub> | <sub>84.36%</sub> | <sub>41.44%</sub> | <sub>WideResNet-28-4</sub> | <sub>ICLR 2020</sub> |
| <sub>**15**</sub> | <sub>**Natural**</sub> | <sub>*Naturally trained model*</sub> | <sub>94.78%</sub> | <sub>0.00%</sub> | <sub>WideResNet-28-10</sub> | <sub>Unpublished</sub> |


### L2
| # | Model ID | Paper | Clean accuracy | Robust accuracy | Architecture | Venue |
|:---:|---|---|:---:|:---:|:---:|:---:|
| <sub>**1**</sub> | <sub>**Augustin2020Adversarial**</sub> | <sub>*[Adversarial Robustness on In- and Out-Distribution Improves Explainability](https://arxiv.org/abs/2003.09461)*</sub> | <sub>91.08%</sub> | <sub>72.91%</sub> | <sub>ResNet-50</sub> | <sub>ECCV 2020</sub> |
| <sub>**2**</sub> | <sub>**Engstrom2019Robustness**</sub> | <sub>*[Robustness library](https://github.com/MadryLab/robustness)*</sub> | <sub>90.83%</sub> | <sub>69.24%</sub> | <sub>ResNet-50</sub> | <sub>Unpublished</sub> |
| <sub>**3**</sub> | <sub>**Rice2020Overfitting**</sub> | <sub>*[Overfitting in adversarially robust deep learning](https://arxiv.org/abs/2002.11569)*</sub> | <sub>88.67%</sub> | <sub>67.68%</sub> | <sub>ResNet-18</sub> | <sub>ICML 2020</sub> |
| <sub>**4**</sub> | <sub>**Rony2019Decoupling**</sub> | <sub>*[Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and Defenses](https://arxiv.org/abs/1811.09600)*</sub> | <sub>89.05%</sub> | <sub>66.44%</sub> | <sub>WideResNet-28-10</sub> | <sub>CVPR 2019</sub> |
| <sub>**5**</sub> | <sub>**Natural**</sub> | <sub>*Naturally trained model*</sub> | <sub>94.78%</sub> | <sub>0.00%</sub> | <sub>WideResNet-28-10</sub> | <sub>Unpublished</sub> |



## Notebooks
We host all the notebooks at Google Colab:
- [AdvBench: quick start](https://colab.research.google.com/drive/1MQY_7O9vj7ixD5ilVRbdQwlNPFvxifHV): a quick tutorial 
to get started that illustrates the main features of **`AdvBench`**.
- [AdvBench: json stats](https://colab.research.google.com/drive/19tgblr13SvaCpG8hoOTv6QCULVJbCec6): various plots based 
on the jsons from `model_info` (robustness over venues, robustness vs accuracy, etc).

Feel free to suggest a new notebook based on the **Model Zoo** or the jsons from `model_info`. We are very interested in
collecting new insights about benefits and tradeoffs between different perturbation types.



## How to contribute
Contributions to **`AdvBench`** are very welcome! Here is how you can help us:
- Do you know some interesting paper that is not listed in the leaderboard? Consider adding new models (see the instructions below 👇).
- Do you have in mind some better *standardized* attack? Do you want to extend **`AdvBench`** to other threat models? We'll be glad to discuss that!
- Do you have an idea how to make the existing codebase better? Just open a pull request or create an issue and we'll be happy to discuss potential changes. 



## Adding a new model to AdvBench
In order to add a new model, submit a pull request where you specify the claim, model definition, and model checkpoint:

- **Claim**: `model_info/<Name><Year><FirstWordOfTheTitle>.json`: follow the convention of the existing json-files to specify the information to be displayed on the website. 
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
  "AA": "53.42"
}
```

- **Model definition**: `advbench/model_zoo/models.py`: add your model definition as a new class. For standard architectures (e.g., `WideResNet`) consider
 inheriting the class defined in `wide_resnet.py` or `resnet.py`. For example:
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

- **Model checkpoint**: `advbench/model_zoo/models.py`: And also add your model entry in `model_dicts` which should also contain 
the *Google Drive ID* with your pytorch model so that it can be downloaded automatically from Google Drive:
```
    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNet(34, 20),
        'gdrive_id': '1vC_Twazji7lBjeMQvAD9uEQxi9Nx2oG-',
    })
```




## Automatic tests
In order to run the tests, run:
- `python -m unittest discover tests -t . -v` for fast testing
- `RUN_SLOW=true python -m unittest discover tests -t . -v` for slower testing

For example, one can test if the clean accuracy on 200 examples exceeds some threshold (70%) or if clean accuracy on 
10'000 examples for each model matches the ones from the jsons located at `advbench/model_info`.

Note that one can specify some configurations like `batch_size`, `data_dir`, `model_dir` in `tests/config.py` for 
running the tests.




## Citation
Would you like to refer to the **`AdvBench`** leaderboard? Or are you using models from the **Model Zoo**? \
Then consider citing our white paper about **`AdvBench`** (currently in preparation, stay tuned).



## Contact 
Feel free to contact us about anything related to **`AdvBench`** by creating an issue, a pull request or 
by email at `adversarial.benchmark@gmail.com`.
