# 2d corruptions
for model_name in AlexNet Standard_R50 Salman2020Do_50_2_Linf Hendrycks2020AugMix Hendrycks2020Many Geirhos2018_SIN Geirhos2018_SIN_IN Geirhos2018_SIN_IN_IN Erichson2022NoisyMix Erichson2022NoisyMix_new Tian2022Deeper_DeiT-S Tian2022Deeper_DeiT-B;
do
    python -m robustbench.eval --n_ex=5000 --dataset=imagenet --threat_model=corruptions --model_name=$model_name --data_dir=/tmldata1/andriush/imagenet --batch_size=256 --to_disk=True 
done

for model_name in AlexNet Standard_R50 Salman2020Do_50_2_Linf Hendrycks2020AugMix Hendrycks2020Many Geirhos2018_SIN Geirhos2018_SIN_IN Geirhos2018_SIN_IN_IN Erichson2022NoisyMix Erichson2022NoisyMix_new Tian2022Deeper_DeiT-S Tian2022Deeper_DeiT-B;
do
    python -m robustbench.eval --n_ex=5000 --dataset=imagenet_3d --threat_model=corruptions --model_name=$model_name --data_dir=/tmldata1/andriush/data/3DCommonCorruptions/ --batch_size=256 --to_disk=True 
done

python -m robustbench.eval --n_ex=5 --dataset=imagenet --threat_model=corruptions --model_name=AlexNet --data_dir=/tmldata1/andriush/imagenet --corruptions_data_dir=/tmldata1/andriush/imagenet --batch_size=256 --to_disk=True
python -m robustbench.eval --n_ex=5 --dataset=imagenet_3d --threat_model=corruptions --model_name=AlexNet --data_dir=/tmldata1/andriush/imagenet --corruptions_data_dir=/tmldata1/andriush/data/3DCommonCorruptions/ --batch_size=256 --to_disk=True

# TODO: add computation of mCE and save it to the json.
# TODO: then rerun both scripts from scratch (as some models failed to evaluate) in parallel

# TODO: do deit results (or at least rankings) match with https://github.com/RobustBench/robustbench/issues/105 / https://github.com/RobustBench/robustbench/compare/master...tian2022deeper?
# TODO: once evals are done, curate carefully the JSONs (should be fine)
# TODO: push as a pull request (based on Edoardo's https://github.com/RobustBench/robustbench/pull/111 and https://github.com/RobustBench/robustbench/compare/master...tian2022deeper); mention that Res224 is now by default in eval.py.


# andriush@overfit-pod-1-1-0-0:/tmldata1/andriush/robustbench$ python eval_imagenet_3d.py
# Standard_R50
# /usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /home/andriush/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 97.8M/97.8M [00:01<00:00, 57.0MB/s]
# Model: Standard_R50, ImageNet-3DCC corruption: bit_error severity: 1 accuracy: 63.38% (avg 63.38%)
# Model: Standard_R50, ImageNet-3DCC corruption: bit_error severity: 2 accuracy: 56.96% (avg 60.17%)
# Model: Standard_R50, ImageNet-3DCC corruption: bit_error severity: 3 accuracy: 48.86% (avg 56.40%)
# Model: Standard_R50, ImageNet-3DCC corruption: bit_error severity: 4 accuracy: 40.26% (avg 52.36%)
# Model: Standard_R50, ImageNet-3DCC corruption: bit_error severity: 5 accuracy: 24.76% (avg 46.84%)
# Model: Standard_R50, ImageNet-3DCC corruption: h265_abr severity: 1 accuracy: 70.64% (avg 50.81%)
# Model: Standard_R50, ImageNet-3DCC corruption: h265_abr severity: 2 accuracy: 69.64% (avg 53.50%)
# Model: Standard_R50, ImageNet-3DCC corruption: h265_abr severity: 3 accuracy: 64.64% (avg 54.89%)
# Model: Standard_R50, ImageNet-3DCC corruption: h265_abr severity: 4 accuracy: 51.40% (avg 54.50%)
# Model: Standard_R50, ImageNet-3DCC corruption: h265_abr severity: 5 accuracy: 52.16% (avg 54.27%)
# Model: Standard_R50, ImageNet-3DCC corruption: h265_crf severity: 1 accuracy: 70.78% (avg 55.77%)
# Model: Standard_R50, ImageNet-3DCC corruption: h265_crf severity: 2 accuracy: 70.64% (avg 57.01%)
# Model: Standard_R50, ImageNet-3DCC corruption: h265_crf severity: 3 accuracy: 68.88% (avg 57.92%)
# Model: Standard_R50, ImageNet-3DCC corruption: h265_crf severity: 4 accuracy: 65.50% (avg 58.46%)
# Model: Standard_R50, ImageNet-3DCC corruption: h265_crf severity: 5 accuracy: 62.48% (avg 58.73%)
# Model: Standard_R50, ImageNet-3DCC corruption: near_focus severity: 1 accuracy: 66.70% (avg 59.23%)
# Model: Standard_R50, ImageNet-3DCC corruption: near_focus severity: 2 accuracy: 62.98% (avg 59.45%)
# Model: Standard_R50, ImageNet-3DCC corruption: near_focus severity: 3 accuracy: 58.92% (avg 59.42%)
# Model: Standard_R50, ImageNet-3DCC corruption: near_focus severity: 4 accuracy: 55.90% (avg 59.24%)
# Model: Standard_R50, ImageNet-3DCC corruption: near_focus severity: 5 accuracy: 52.30% (avg 58.89%)
# Model: Standard_R50, ImageNet-3DCC corruption: far_focus severity: 1 accuracy: 67.08% (avg 59.28%)
# Model: Standard_R50, ImageNet-3DCC corruption: far_focus severity: 2 accuracy: 60.64% (avg 59.34%)
# Model: Standard_R50, ImageNet-3DCC corruption: far_focus severity: 3 accuracy: 53.72% (avg 59.10%)
# Model: Standard_R50, ImageNet-3DCC corruption: far_focus severity: 4 accuracy: 48.88% (avg 58.67%)
# Model: Standard_R50, ImageNet-3DCC corruption: far_focus severity: 5 accuracy: 44.56% (avg 58.11%)
# Model: Standard_R50, ImageNet-3DCC corruption: color_quant severity: 1 accuracy: 73.00% (avg 58.68%)
# Model: Standard_R50, ImageNet-3DCC corruption: color_quant severity: 2 accuracy: 71.76% (avg 59.16%)
# Model: Standard_R50, ImageNet-3DCC corruption: color_quant severity: 3 accuracy: 67.76% (avg 59.47%)
# Model: Standard_R50, ImageNet-3DCC corruption: color_quant severity: 4 accuracy: 50.74% (avg 59.17%)
# Model: Standard_R50, ImageNet-3DCC corruption: color_quant severity: 5 accuracy: 18.54% (avg 57.82%)
# Model: Standard_R50, ImageNet-3DCC corruption: flash severity: 1 accuracy: 49.24% (avg 57.54%)
# Model: Standard_R50, ImageNet-3DCC corruption: flash severity: 2 accuracy: 45.32% (avg 57.16%)
# Model: Standard_R50, ImageNet-3DCC corruption: flash severity: 3 accuracy: 39.76% (avg 56.63%)
# Model: Standard_R50, ImageNet-3DCC corruption: flash severity: 4 accuracy: 33.12% (avg 55.94%)
# Model: Standard_R50, ImageNet-3DCC corruption: flash severity: 5 accuracy: 21.96% (avg 54.97%)
# Model: Standard_R50, ImageNet-3DCC corruption: fog_3d severity: 1 accuracy: 65.42% (avg 55.26%)
# Model: Standard_R50, ImageNet-3DCC corruption: fog_3d severity: 2 accuracy: 51.76% (avg 55.16%)
# Model: Standard_R50, ImageNet-3DCC corruption: fog_3d severity: 3 accuracy: 40.16% (avg 54.77%)
# Model: Standard_R50, ImageNet-3DCC corruption: fog_3d severity: 4 accuracy: 31.28% (avg 54.17%)
# Model: Standard_R50, ImageNet-3DCC corruption: fog_3d severity: 5 accuracy: 25.26% (avg 53.44%)
# Model: Standard_R50, ImageNet-3DCC corruption: iso_noise severity: 1 accuracy: 40.60% (avg 53.13%)
# Model: Standard_R50, ImageNet-3DCC corruption: iso_noise severity: 2 accuracy: 37.00% (avg 52.75%)
# Model: Standard_R50, ImageNet-3DCC corruption: iso_noise severity: 3 accuracy: 28.96% (avg 52.19%)
# Model: Standard_R50, ImageNet-3DCC corruption: iso_noise severity: 4 accuracy: 19.00% (avg 51.44%)
# Model: Standard_R50, ImageNet-3DCC corruption: iso_noise severity: 5 accuracy: 8.04% (avg 50.47%)
# Model: Standard_R50, ImageNet-3DCC corruption: low_light severity: 1 accuracy: 55.50% (avg 50.58%)
# Model: Standard_R50, ImageNet-3DCC corruption: low_light severity: 2 accuracy: 51.90% (avg 50.61%)
# Model: Standard_R50, ImageNet-3DCC corruption: low_light severity: 3 accuracy: 46.44% (avg 50.52%)
# Model: Standard_R50, ImageNet-3DCC corruption: low_light severity: 4 accuracy: 39.90% (avg 50.31%)
# Model: Standard_R50, ImageNet-3DCC corruption: low_light severity: 5 accuracy: 28.28% (avg 49.87%)
# Model: Standard_R50, ImageNet-3DCC corruption: xy_motion_blur severity: 1 accuracy: 51.12% (avg 49.89%)
# Model: Standard_R50, ImageNet-3DCC corruption: xy_motion_blur severity: 2 accuracy: 39.34% (avg 49.69%)
# Model: Standard_R50, ImageNet-3DCC corruption: xy_motion_blur severity: 3 accuracy: 29.72% (avg 49.31%)
# Model: Standard_R50, ImageNet-3DCC corruption: xy_motion_blur severity: 4 accuracy: 22.92% (avg 48.82%)
# Model: Standard_R50, ImageNet-3DCC corruption: xy_motion_blur severity: 5 accuracy: 17.90% (avg 48.26%)
# Model: Standard_R50, ImageNet-3DCC corruption: z_motion_blur severity: 1 accuracy: 47.60% (avg 48.25%)
# Model: Standard_R50, ImageNet-3DCC corruption: z_motion_blur severity: 2 accuracy: 39.78% (avg 48.10%)
# Model: Standard_R50, ImageNet-3DCC corruption: z_motion_blur severity: 3 accuracy: 32.80% (avg 47.84%)
# Model: Standard_R50, ImageNet-3DCC corruption: z_motion_blur severity: 4 accuracy: 27.60% (avg 47.49%)
# Model: Standard_R50, ImageNet-3DCC corruption: z_motion_blur severity: 5 accuracy: 24.92% (avg 47.12%)
# 0.47117666666666663 [0.6338, 0.5696, 0.4886, 0.4026, 0.2476, 0.7064, 0.6964, 0.6464, 0.514, 0.5216, 0.7078, 0.7064, 0.6888, 0.655, 0.6248, 0.667, 0.6298, 0.5892, 0.559, 0.523, 0.6708, 0.6064, 0.5372, 0.4888, 0.4456, 0.73, 0.7176, 0.6776, 0.5074, 0.1854, 0.4924, 0.4532, 0.3976, 0.3312, 0.2196, 0.6542, 0.5176, 0.4016, 0.3128, 0.2526, 0.406, 0.37, 0.2896, 0.19, 0.0804, 0.555, 0.519, 0.4644, 0.399, 0.2828, 0.5112, 0.3934, 0.2972, 0.2292, 0.179, 0.476, 0.3978, 0.328, 0.276, 0.2492]




# Salman2020Do_50_2
# Traceback (most recent call last):
#   File "eval_imagenet_3d.py", line 14, in <module>
#     model = load_model(model_name, dataset='imagenet', threat_model='corruptions').to(device)
#   File "/tmldata1/andriush/robustbench/robustbench/utils.py", line 132, in load_model
#     if models[model_name]['gdrive_id'] is None:
# KeyError: 'Salman2020Do_50_2'
# andriush@overfit-pod-1-1-0-0:/tmldata1/andriush/robustbench$ %
# ➜  overfit git:(main) ✗


# andriush@overfit-pod-1-1-0-0:/tmldata1/andriush/robustbench$ ./eval_corruptions_2d.sh
# /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'robustbench.eval' found in sys.modules after import of package 'robustbench', but prior to execution of 'robustbench.eval'; this may result in unpredictable behaviour
#   warn(RuntimeWarning(msg))
# /usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Clean accuracy: 76.72%
# Evaluating over 15 corruptions
#   0%|                                                                                                                        | 0/15 [00:00<?, ?it/s]corruption=shot_noise, severity=1: 58.28% accuracy
# corruption=shot_noise, severity=2: 44.86% accuracy
# corruption=shot_noise, severity=3: 28.56% accuracy
# corruption=shot_noise, severity=4: 10.26% accuracy
# corruption=shot_noise, severity=5: 3.52% accuracy
#   7%|███████▍                                                                                                        | 1/15 [01:21<18:55, 81.12s/it]corruption=motion_blur, severity=1: 63.60% accuracy
# corruption=motion_blur, severity=2: 53.42% accuracy
# corruption=motion_blur, severity=3: 37.54% accuracy
# corruption=motion_blur, severity=4: 21.58% accuracy
# corruption=motion_blur, severity=5: 14.52% accuracy
#  13%|██████████████▉                                                                                                 | 2/15 [02:39<17:11, 79.37s/it]corruption=snow, severity=1: 54.68% accuracy
# corruption=snow, severity=2: 31.20% accuracy
# corruption=snow, severity=3: 34.94% accuracy
# corruption=snow, severity=4: 23.44% accuracy
# corruption=snow, severity=5: 16.50% accuracy
#  20%|██████████████████████▍                                                                                         | 3/15 [03:59<15:58, 79.87s/it]corruption=pixelate, severity=1: 63.80% accuracy
# corruption=pixelate, severity=2: 64.34% accuracy
# corruption=pixelate, severity=3: 45.98% accuracy
# corruption=pixelate, severity=4: 29.30% accuracy
# corruption=pixelate, severity=5: 20.88% accuracy
#  27%|█████████████████████████████▊                                                                                  | 4/15 [05:15<14:21, 78.28s/it]corruption=gaussian_noise, severity=1: 60.20% accuracy
# corruption=gaussian_noise, severity=2: 48.78% accuracy
# corruption=gaussian_noise, severity=3: 31.24% accuracy
# corruption=gaussian_noise, severity=4: 14.22% accuracy
# corruption=gaussian_noise, severity=5: 2.88% accuracy
#  33%|█████████████████████████████████████▎                                                                          | 5/15 [06:33<13:02, 78.25s/it]corruption=defocus_blur, severity=1: 59.14% accuracy
# corruption=defocus_blur, severity=2: 51.54% accuracy
# corruption=defocus_blur, severity=3: 38.08% accuracy
# corruption=defocus_blur, severity=4: 27.52% accuracy
# corruption=defocus_blur, severity=5: 18.24% accuracy
#  40%|████████████████████████████████████████████▊                                                                   | 6/15 [07:53<11:48, 78.69s/it]corruption=brightness, severity=1: 73.60% accuracy
# corruption=brightness, severity=2: 72.22% accuracy
# corruption=brightness, severity=3: 69.58% accuracy
# corruption=brightness, severity=4: 64.96% accuracy
# corruption=brightness, severity=5: 58.86% accuracy
#  47%|████████████████████████████████████████████████████▎                                                           | 7/15 [09:09<10:21, 77.74s/it]corruption=fog, severity=1: 60.92% accuracy
# corruption=fog, severity=2: 56.00% accuracy
# corruption=fog, severity=3: 46.74% accuracy
# corruption=fog, severity=4: 40.24% accuracy
# corruption=fog, severity=5: 23.60% accuracy
#  53%|███████████████████████████████████████████████████████████▋                                                    | 8/15 [10:23<08:55, 76.56s/it]corruption=zoom_blur, severity=1: 52.58% accuracy
# corruption=zoom_blur, severity=2: 43.02% accuracy
# corruption=zoom_blur, severity=3: 35.60% accuracy
# corruption=zoom_blur, severity=4: 28.44% accuracy
# corruption=zoom_blur, severity=5: 21.94% accuracy
#  60%|███████████████████████████████████████████████████████████████████▏                                            | 9/15 [11:36<07:33, 75.64s/it]corruption=frost, severity=1: 61.06% accuracy
# corruption=frost, severity=2: 43.66% accuracy
# corruption=frost, severity=3: 30.64% accuracy
# corruption=frost, severity=4: 28.64% accuracy
# corruption=frost, severity=5: 22.58% accuracy
#  67%|██████████████████████████████████████████████████████████████████████████                                     | 10/15 [12:52<06:17, 75.59s/it]corruption=glass_blur, severity=1: 53.62% accuracy
# corruption=glass_blur, severity=2: 39.74% accuracy
# corruption=glass_blur, severity=3: 17.08% accuracy
# corruption=glass_blur, severity=4: 13.06% accuracy
# corruption=glass_blur, severity=5: 10.34% accuracy
#  73%|█████████████████████████████████████████████████████████████████████████████████▍                             | 11/15 [14:06<05:01, 75.26s/it]corruption=impulse_noise, severity=1: 48.54% accuracy
# corruption=impulse_noise, severity=2: 38.62% accuracy
# corruption=impulse_noise, severity=3: 29.04% accuracy
# corruption=impulse_noise, severity=4: 11.30% accuracy
# corruption=impulse_noise, severity=5: 2.44% accuracy
#  80%|████████████████████████████████████████████████████████████████████████████████████████▊                      | 12/15 [15:24<03:47, 75.91s/it]corruption=contrast, severity=1: 64.28% accuracy
# corruption=contrast, severity=2: 58.24% accuracy
# corruption=contrast, severity=3: 46.60% accuracy
# corruption=contrast, severity=4: 20.98% accuracy
# corruption=contrast, severity=5: 5.42% accuracy
#  87%|████████████████████████████████████████████████████████████████████████████████████████████████▏              | 13/15 [16:37<02:30, 75.15s/it]corruption=jpeg_compression, severity=1: 66.44% accuracy
# corruption=jpeg_compression, severity=2: 62.34% accuracy
# corruption=jpeg_compression, severity=3: 60.04% accuracy
# corruption=jpeg_compression, severity=4: 47.86% accuracy
# corruption=jpeg_compression, severity=5: 32.54% accuracy
#  93%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌       | 14/15 [17:51<01:14, 74.79s/it]corruption=elastic_transform, severity=1: 66.84% accuracy
# corruption=elastic_transform, severity=2: 45.14% accuracy
# corruption=elastic_transform, severity=3: 54.56% accuracy
# corruption=elastic_transform, severity=4: 41.22% accuracy
# corruption=elastic_transform, severity=5: 17.02% accuracy
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [19:06<00:00, 76.43s/it]
# Adversarial accuracy: 39.48%

# andriush@overfit-pod-1-1-0-0:/tmldata1/andriush/robustbench$ python -m robustbench.eval --n_ex=5000 --dataset=imagenet --threat_model=corruptions --model_name=Salman2020Do_50_2_Linf --data_dir=/tmldata1/andriush/imagenet --batch_size=256 --to_disk=True
# /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'robustbench.eval' found in sys.modules after import of package 'robustbench', but prior to execution of 'robustbench.eval'; this may result in unpredictable behaviour
#   warn(RuntimeWarning(msg))
# Download started: path=models/imagenet/corruptions/Salman2020Do_50_2_Linf.pt (gdrive_id=1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB)
# Download finished: path=models/imagenet/corruptions/Salman2020Do_50_2_Linf.pt (gdrive_id=1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB)
# Clean accuracy: 68.64%
# Evaluating over 15 corruptions
#   0%|                                                                                                                        | 0/15 [00:00<?, ?it/s]corruption=shot_noise, severity=1: 57.82% accuracy
# corruption=shot_noise, severity=2: 42.84% accuracy
# corruption=shot_noise, severity=3: 25.68% accuracy
# corruption=shot_noise, severity=4: 9.30% accuracy
# corruption=shot_noise, severity=5: 3.32% accuracy
#   7%|███████▍                                                                                                        | 1/15 [01:15<17:41, 75.81s/it]corruption=motion_blur, severity=1: 58.10% accuracy
# corruption=motion_blur, severity=2: 48.54% accuracy
# corruption=motion_blur, severity=3: 35.88% accuracy
# corruption=motion_blur, severity=4: 22.80% accuracy
# corruption=motion_blur, severity=5: 16.68% accuracy
#  13%|██████████████▉                                                                                                 | 2/15 [02:24<15:34, 71.86s/it]corruption=snow, severity=1: 54.80% accuracy
# corruption=snow, severity=2: 38.56% accuracy
# corruption=snow, severity=3: 37.34% accuracy
# corruption=snow, severity=4: 24.64% accuracy
# corruption=snow, severity=5: 23.24% accuracy
#  20%|██████████████████████▍                                                                                         | 3/15 [03:41<14:48, 74.07s/it]corruption=pixelate, severity=1: 65.94% accuracy
# corruption=pixelate, severity=2: 65.26% accuracy
# corruption=pixelate, severity=3: 62.52% accuracy
# corruption=pixelate, severity=4: 57.42% accuracy
# corruption=pixelate, severity=5: 53.86% accuracy
#  27%|█████████████████████████████▊                                                                                  | 4/15 [04:52<13:19, 72.67s/it]corruption=gaussian_noise, severity=1: 59.60% accuracy
# corruption=gaussian_noise, severity=2: 46.70% accuracy
# corruption=gaussian_noise, severity=3: 26.66% accuracy
# corruption=gaussian_noise, severity=4: 10.20% accuracy
# corruption=gaussian_noise, severity=5: 2.18% accuracy
#  33%|█████████████████████████████████████▎                                                                          | 5/15 [06:05<12:10, 73.04s/it]corruption=defocus_blur, severity=1: 45.92% accuracy
# corruption=defocus_blur, severity=2: 37.32% accuracy
# corruption=defocus_blur, severity=3: 23.88% accuracy
# corruption=defocus_blur, severity=4: 15.60% accuracy
# corruption=defocus_blur, severity=5: 10.16% accuracy
#  40%|████████████████████████████████████████████▊                                                                   | 6/15 [07:15<10:48, 72.03s/it]corruption=brightness, severity=1: 67.84% accuracy
# corruption=brightness, severity=2: 65.74% accuracy
# corruption=brightness, severity=3: 61.50% accuracy
# corruption=brightness, severity=4: 54.76% accuracy
# corruption=brightness, severity=5: 45.08% accuracy
#  47%|████████████████████████████████████████████████████▎                                                           | 7/15 [08:29<09:39, 72.41s/it]corruption=fog, severity=1: 23.86% accuracy
# corruption=fog, severity=2: 10.06% accuracy
# corruption=fog, severity=3: 3.74% accuracy
# corruption=fog, severity=4: 2.96% accuracy
# corruption=fog, severity=5: 1.02% accuracy
#  53%|███████████████████████████████████████████████████████████▋                                                    | 8/15 [09:37<08:18, 71.24s/it]corruption=zoom_blur, severity=1: 52.00% accuracy
# corruption=zoom_blur, severity=2: 45.48% accuracy
# corruption=zoom_blur, severity=3: 37.78% accuracy
# corruption=zoom_blur, severity=4: 33.38% accuracy
# corruption=zoom_blur, severity=5: 27.04% accuracy
#  60%|███████████████████████████████████████████████████████████████████▏                                            | 9/15 [10:49<07:09, 71.53s/it]corruption=frost, severity=1: 58.98% accuracy
# corruption=frost, severity=2: 42.48% accuracy
# corruption=frost, severity=3: 27.96% accuracy
# corruption=frost, severity=4: 25.30% accuracy
# corruption=frost, severity=5: 18.10% accuracy
#  67%|██████████████████████████████████████████████████████████████████████████                                     | 10/15 [12:05<06:04, 72.86s/it]corruption=glass_blur, severity=1: 55.56% accuracy
# corruption=glass_blur, severity=2: 46.64% accuracy
# corruption=glass_blur, severity=3: 37.80% accuracy
# corruption=glass_blur, severity=4: 30.06% accuracy
# corruption=glass_blur, severity=5: 18.76% accuracy
#  73%|█████████████████████████████████████████████████████████████████████████████████▍                             | 11/15 [13:14<04:46, 71.59s/it]corruption=impulse_noise, severity=1: 39.78% accuracy
# corruption=impulse_noise, severity=2: 25.42% accuracy
# corruption=impulse_noise, severity=3: 16.30% accuracy
# corruption=impulse_noise, severity=4: 5.58% accuracy
# corruption=impulse_noise, severity=5: 1.62% accuracy
#  80%|████████████████████████████████████████████████████████████████████████████████████████▊                      | 12/15 [14:31<03:39, 73.30s/it]corruption=contrast, severity=1: 34.48% accuracy
# corruption=contrast, severity=2: 14.98% accuracy
# corruption=contrast, severity=3: 2.56% accuracy
# corruption=contrast, severity=4: 0.54% accuracy
# corruption=contrast, severity=5: 0.54% accuracy
#  87%|████████████████████████████████████████████████████████████████████████████████████████████████▏              | 13/15 [15:40<02:23, 71.86s/it]corruption=jpeg_compression, severity=1: 66.34% accuracy
# corruption=jpeg_compression, severity=2: 65.34% accuracy
# corruption=jpeg_compression, severity=3: 65.28% accuracy
# corruption=jpeg_compression, severity=4: 63.70% accuracy
# corruption=jpeg_compression, severity=5: 61.58% accuracy
#  93%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌       | 14/15 [16:51<01:11, 71.52s/it]corruption=elastic_transform, severity=1: 58.58% accuracy
# corruption=elastic_transform, severity=2: 40.54% accuracy
# corruption=elastic_transform, severity=3: 62.24% accuracy
# corruption=elastic_transform, severity=4: 58.54% accuracy
# corruption=elastic_transform, severity=5: 46.36% accuracy
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [18:05<00:00, 72.34s/it]
# Adversarial accuracy: 36.09%


# /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'robustbench.eval' found in sys.modules after import of package 'robustbench', but prior to execution of 'robustbench.eval'; this may result in unpredictable behaviour
#   warn(RuntimeWarning(msg))
# Clean accuracy: 77.34%
# Evaluating over 15 corruptions
#   0%|                                                                                                                        | 0/15 [00:00<?, ?it/s]corruption=shot_noise, severity=1: 67.22% accuracy
# corruption=shot_noise, severity=2: 59.80% accuracy
# corruption=shot_noise, severity=3: 50.22% accuracy
# corruption=shot_noise, severity=4: 31.60% accuracy
# corruption=shot_noise, severity=5: 19.24% accuracy
#   7%|███████▍                                                                                                        | 1/15 [01:04<15:08, 64.92s/it]corruption=motion_blur, severity=1: 71.48% accuracy
# corruption=motion_blur, severity=2: 67.34% accuracy
# corruption=motion_blur, severity=3: 57.78% accuracy
# corruption=motion_blur, severity=4: 41.92% accuracy
# corruption=motion_blur, severity=5: 29.66% accuracy
#  13%|██████████████▉                                                                                                 | 2/15 [02:02<13:08, 60.68s/it]corruption=snow, severity=1: 60.60% accuracy
# corruption=snow, severity=2: 41.00% accuracy
# corruption=snow, severity=3: 42.56% accuracy
# corruption=snow, severity=4: 30.14% accuracy
# corruption=snow, severity=5: 23.16% accuracy
#  20%|██████████████████████▍                                                                                         | 3/15 [03:02<12:02, 60.22s/it]corruption=pixelate, severity=1: 69.86% accuracy
# corruption=pixelate, severity=2: 68.36% accuracy
# corruption=pixelate, severity=3: 59.78% accuracy
# corruption=pixelate, severity=4: 47.64% accuracy
# corruption=pixelate, severity=5: 42.12% accuracy
#  27%|█████████████████████████████▊                                                                                  | 4/15 [04:01<10:59, 59.91s/it]corruption=gaussian_noise, severity=1: 67.20% accuracy
# corruption=gaussian_noise, severity=2: 61.00% accuracy
# corruption=gaussian_noise, severity=3: 50.22% accuracy
# corruption=gaussian_noise, severity=4: 33.68% accuracy
# corruption=gaussian_noise, severity=5: 11.88% accuracy
#  33%|█████████████████████████████████████▎                                                                          | 5/15 [05:07<10:18, 61.88s/it]corruption=defocus_blur, severity=1: 64.50% accuracy
# corruption=defocus_blur, severity=2: 60.66% accuracy
# corruption=defocus_blur, severity=3: 50.68% accuracy
# corruption=defocus_blur, severity=4: 38.54% accuracy
# corruption=defocus_blur, severity=5: 25.90% accuracy
#  40%|████████████████████████████████████████████▊                                                                   | 6/15 [06:12<09:27, 63.04s/it]corruption=brightness, severity=1: 74.70% accuracy
# corruption=brightness, severity=2: 73.00% accuracy
# corruption=brightness, severity=3: 70.60% accuracy
# corruption=brightness, severity=4: 66.76% accuracy
# corruption=brightness, severity=5: 60.86% accuracy
#  47%|████████████████████████████████████████████████████▎                                                           | 7/15 [07:47<09:47, 73.50s/it]corruption=fog, severity=1: 63.62% accuracy
# corruption=fog, severity=2: 58.34% accuracy
# corruption=fog, severity=3: 48.04% accuracy
# corruption=fog, severity=4: 41.62% accuracy
# corruption=fog, severity=5: 24.20% accuracy
#  53%|███████████████████████████████████████████████████████████▋                                                    | 8/15 [08:53<08:18, 71.20s/it]corruption=zoom_blur, severity=1: 61.98% accuracy
# corruption=zoom_blur, severity=2: 55.24% accuracy
# corruption=zoom_blur, severity=3: 51.62% accuracy
# corruption=zoom_blur, severity=4: 43.28% accuracy
# corruption=zoom_blur, severity=5: 35.58% accuracy
#  60%|███████████████████████████████████████████████████████████████████▏                                            | 9/15 [09:58<06:54, 69.06s/it]corruption=frost, severity=1: 64.24% accuracy
# corruption=frost, severity=2: 48.70% accuracy
# corruption=frost, severity=3: 36.72% accuracy
# corruption=frost, severity=4: 34.78% accuracy
# corruption=frost, severity=5: 27.76% accuracy
#  67%|██████████████████████████████████████████████████████████████████████████                                     | 10/15 [11:01<05:36, 67.39s/it]corruption=glass_blur, severity=1: 60.48% accuracy
# corruption=glass_blur, severity=2: 48.26% accuracy
# corruption=glass_blur, severity=3: 26.86% accuracy
# corruption=glass_blur, severity=4: 22.66% accuracy
# corruption=glass_blur, severity=5: 17.40% accuracy
#  73%|█████████████████████████████████████████████████████████████████████████████████▍                             | 11/15 [12:05<04:25, 66.35s/it]corruption=impulse_noise, severity=1: 64.52% accuracy
# corruption=impulse_noise, severity=2: 57.60% accuracy
# corruption=impulse_noise, severity=3: 50.96% accuracy
# corruption=impulse_noise, severity=4: 32.70% accuracy
# corruption=impulse_noise, severity=5: 12.74% accuracy
#  80%|████████████████████████████████████████████████████████████████████████████████████████▊                      | 12/15 [13:13<03:20, 66.94s/it]corruption=contrast, severity=1: 70.64% accuracy
# corruption=contrast, severity=2: 67.64% accuracy
# corruption=contrast, severity=3: 60.94% accuracy
# corruption=contrast, severity=4: 41.74% accuracy
# corruption=contrast, severity=5: 15.60% accuracy
#  87%|████████████████████████████████████████████████████████████████████████████████████████████████▏              | 13/15 [14:15<02:10, 65.21s/it]corruption=jpeg_compression, severity=1: 67.90% accuracy
# corruption=jpeg_compression, severity=2: 64.96% accuracy
# corruption=jpeg_compression, severity=3: 63.12% accuracy
# corruption=jpeg_compression, severity=4: 56.80% accuracy
# corruption=jpeg_compression, severity=5: 48.44% accuracy
#  93%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌       | 14/15 [15:14<01:03, 63.33s/it]corruption=elastic_transform, severity=1: 69.38% accuracy
# corruption=elastic_transform, severity=2: 47.98% accuracy
# corruption=elastic_transform, severity=3: 62.32% accuracy
# corruption=elastic_transform, severity=4: 51.90% accuracy
# corruption=elastic_transform, severity=5: 28.72% accuracy
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [16:22<00:00, 65.51s/it]
# Adversarial accuracy: 49.33%
# /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'robustbench.eval' found in sys.modules after import of package 'robustbench', but prior to execution of 'robustbench.eval'; this may result in unpredictable behaviour
#   warn(RuntimeWarning(msg))
# Download started: path=models/imagenet/corruptions/Hendrycks2020Many.pt (gdrive_id=1kylueoLtYtxkpVzoOA1B6tqdbRl2xt9X)
# Download finished: path=models/imagenet/corruptions/Hendrycks2020Many.pt (gdrive_id=1kylueoLtYtxkpVzoOA1B6tqdbRl2xt9X)
# Clean accuracy: 76.86%
# Evaluating over 15 corruptions
#   0%|                                                                                                                        | 0/15 [00:00<?, ?it/s]corruption=shot_noise, severity=1: 72.14% accuracy
# corruption=shot_noise, severity=2: 68.36% accuracy
# corruption=shot_noise, severity=3: 61.50% accuracy
# corruption=shot_noise, severity=4: 46.78% accuracy
# corruption=shot_noise, severity=5: 35.26% accuracy
#   7%|███████▍                                                                                                        | 1/15 [01:05<15:22, 65.92s/it]corruption=motion_blur, severity=1: 70.36% accuracy
# corruption=motion_blur, severity=2: 64.68% accuracy
# corruption=motion_blur, severity=3: 51.54% accuracy
# corruption=motion_blur, severity=4: 33.16% accuracy
# corruption=motion_blur, severity=5: 21.70% accuracy
#  13%|██████████████▉                                                                                                 | 2/15 [02:08<13:50, 63.88s/it]corruption=snow, severity=1: 60.54% accuracy
# corruption=snow, severity=2: 45.34% accuracy
# corruption=snow, severity=3: 46.12% accuracy
# corruption=snow, severity=4: 35.28% accuracy
# corruption=snow, severity=5: 29.68% accuracy
#  20%|██████████████████████▍                                                                                         | 3/15 [03:09<12:32, 62.69s/it]corruption=pixelate, severity=1: 75.02% accuracy
# corruption=pixelate, severity=2: 74.78% accuracy
# corruption=pixelate, severity=3: 72.00% accuracy
# corruption=pixelate, severity=4: 63.42% accuracy
# corruption=pixelate, severity=5: 42.90% accuracy
#  27%|█████████████████████████████▊                                                                                  | 4/15 [04:05<11:01, 60.17s/it]corruption=gaussian_noise, severity=1: 72.08% accuracy
# corruption=gaussian_noise, severity=2: 69.72% accuracy
# corruption=gaussian_noise, severity=3: 63.24% accuracy
# corruption=gaussian_noise, severity=4: 51.90% accuracy
# corruption=gaussian_noise, severity=5: 36.98% accuracy
#  33%|█████████████████████████████████████▎                                                                          | 5/15 [05:08<10:10, 61.04s/it]corruption=defocus_blur, severity=1: 68.52% accuracy
# corruption=defocus_blur, severity=2: 64.42% accuracy
# corruption=defocus_blur, severity=3: 52.72% accuracy
# corruption=defocus_blur, severity=4: 41.46% accuracy
# corruption=defocus_blur, severity=5: 29.98% accuracy
#  40%|████████████████████████████████████████████▊                                                                   | 6/15 [06:06<08:58, 59.83s/it]corruption=brightness, severity=1: 74.96% accuracy
# corruption=brightness, severity=2: 74.06% accuracy
# corruption=brightness, severity=3: 72.00% accuracy
# corruption=brightness, severity=4: 69.44% accuracy
# corruption=brightness, severity=5: 65.42% accuracy
#  47%|████████████████████████████████████████████████████▎                                                           | 7/15 [07:04<07:55, 59.48s/it]corruption=fog, severity=1: 64.40% accuracy
# corruption=fog, severity=2: 58.76% accuracy
# corruption=fog, severity=3: 51.76% accuracy
# corruption=fog, severity=4: 48.06% accuracy
# corruption=fog, severity=5: 34.04% accuracy
#  53%|███████████████████████████████████████████████████████████▋                                                    | 8/15 [08:03<06:53, 59.09s/it]corruption=zoom_blur, severity=1: 57.52% accuracy
# corruption=zoom_blur, severity=2: 47.04% accuracy
# corruption=zoom_blur, severity=3: 38.82% accuracy
# corruption=zoom_blur, severity=4: 31.80% accuracy
# corruption=zoom_blur, severity=5: 23.86% accuracy
#  60%|███████████████████████████████████████████████████████████████████▏                                            | 9/15 [09:01<05:52, 58.80s/it]corruption=frost, severity=1: 66.32% accuracy
# corruption=frost, severity=2: 54.80% accuracy
# corruption=frost, severity=3: 45.44% accuracy
# corruption=frost, severity=4: 42.80% accuracy
# corruption=frost, severity=5: 36.72% accuracy
#  67%|██████████████████████████████████████████████████████████████████████████                                     | 10/15 [10:00<04:55, 59.02s/it]corruption=glass_blur, severity=1: 66.84% accuracy
# corruption=glass_blur, severity=2: 57.46% accuracy
# corruption=glass_blur, severity=3: 31.42% accuracy
# corruption=glass_blur, severity=4: 26.18% accuracy
# corruption=glass_blur, severity=5: 18.46% accuracy
#  73%|█████████████████████████████████████████████████████████████████████████████████▍                             | 11/15 [10:57<03:53, 58.27s/it]corruption=impulse_noise, severity=1: 70.78% accuracy
# corruption=impulse_noise, severity=2: 67.22% accuracy
# corruption=impulse_noise, severity=3: 63.78% accuracy
# corruption=impulse_noise, severity=4: 53.08% accuracy
# corruption=impulse_noise, severity=5: 41.00% accuracy
#  80%|████████████████████████████████████████████████████████████████████████████████████████▊                      | 12/15 [12:07<03:06, 62.03s/it]corruption=contrast, severity=1: 69.76% accuracy
# corruption=contrast, severity=2: 65.48% accuracy
# corruption=contrast, severity=3: 57.08% accuracy
# corruption=contrast, severity=4: 35.30% accuracy
# corruption=contrast, severity=5: 12.50% accuracy
#  87%|████████████████████████████████████████████████████████████████████████████████████████████████▏              | 13/15 [13:03<02:00, 60.09s/it]corruption=jpeg_compression, severity=1: 70.36% accuracy
# corruption=jpeg_compression, severity=2: 67.68% accuracy
# corruption=jpeg_compression, severity=3: 66.10% accuracy
# corruption=jpeg_compression, severity=4: 55.50% accuracy
# corruption=jpeg_compression, severity=5: 38.22% accuracy
#  93%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌       | 14/15 [14:02<00:59, 59.60s/it]corruption=elastic_transform, severity=1: 69.70% accuracy
# corruption=elastic_transform, severity=2: 46.92% accuracy
# corruption=elastic_transform, severity=3: 61.74% accuracy
# corruption=elastic_transform, severity=4: 50.62% accuracy
# corruption=elastic_transform, severity=5: 25.08% accuracy
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [14:59<00:00, 59.95s/it]
# Adversarial accuracy: 52.90%
# /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'robustbench.eval' found in sys.modules after import of package 'robustbench', but prior to execution of 'robustbench.eval'; this may result in unpredictable behaviour
#   warn(RuntimeWarning(msg))
# Download started: path=models/imagenet/corruptions/Geirhos2018_SIN.pt (gdrive_id=1hLgeY_rQIaOT4R-t_KyOqPNkczfaedgs)
# Download finished: path=models/imagenet/corruptions/Geirhos2018_SIN.pt (gdrive_id=1hLgeY_rQIaOT4R-t_KyOqPNkczfaedgs)
# Clean accuracy: 60.08%
# Evaluating over 15 corruptions
#   0%|                                                                                                                        | 0/15 [00:00<?, ?it/s]corruption=shot_noise, severity=1: 50.42% accuracy
# corruption=shot_noise, severity=2: 44.26% accuracy
# corruption=shot_noise, severity=3: 38.54% accuracy
# corruption=shot_noise, severity=4: 28.30% accuracy
# corruption=shot_noise, severity=5: 21.70% accuracy
#   7%|███████▍                                                                                                        | 1/15 [01:05<15:11, 65.14s/it]corruption=motion_blur, severity=1: 52.12% accuracy
# corruption=motion_blur, severity=2: 45.40% accuracy
# corruption=motion_blur, severity=3: 36.94% accuracy
# corruption=motion_blur, severity=4: 28.34% accuracy
# corruption=motion_blur, severity=5: 23.66% accuracy
#  13%|██████████████▉                                                                                                 | 2/15 [02:05<13:32, 62.51s/it]corruption=snow, severity=1: 48.46% accuracy
# corruption=snow, severity=2: 38.48% accuracy
# corruption=snow, severity=3: 39.84% accuracy
# corruption=snow, severity=4: 33.52% accuracy
# corruption=snow, severity=5: 31.88% accuracy
#  20%|██████████████████████▍                                                                                         | 3/15 [03:06<12:22, 61.88s/it]corruption=pixelate, severity=1: 57.58% accuracy
# corruption=pixelate, severity=2: 57.78% accuracy
# corruption=pixelate, severity=3: 48.88% accuracy
# corruption=pixelate, severity=4: 39.94% accuracy
# corruption=pixelate, severity=5: 36.32% accuracy
#  27%|█████████████████████████████▊                                                                                  | 4/15 [04:06<11:11, 61.02s/it]corruption=gaussian_noise, severity=1: 52.84% accuracy
# corruption=gaussian_noise, severity=2: 48.18% accuracy
# corruption=gaussian_noise, severity=3: 40.80% accuracy
# corruption=gaussian_noise, severity=4: 32.24% accuracy
# corruption=gaussian_noise, severity=5: 21.42% accuracy
#  33%|█████████████████████████████████████▎                                                                          | 5/15 [05:13<10:31, 63.18s/it]corruption=defocus_blur, severity=1: 40.88% accuracy
# corruption=defocus_blur, severity=2: 33.14% accuracy
# corruption=defocus_blur, severity=3: 26.00% accuracy
# corruption=defocus_blur, severity=4: 19.78% accuracy
# corruption=defocus_blur, severity=5: 14.46% accuracy
#  40%|████████████████████████████████████████████▊                                                                   | 6/15 [06:11<09:13, 61.45s/it]corruption=brightness, severity=1: 59.88% accuracy
# corruption=brightness, severity=2: 57.66% accuracy
# corruption=brightness, severity=3: 56.00% accuracy
# corruption=brightness, severity=4: 54.30% accuracy
# corruption=brightness, severity=5: 51.50% accuracy
#  47%|████████████████████████████████████████████████████▎                                                           | 7/15 [07:14<08:14, 61.77s/it]corruption=fog, severity=1: 54.08% accuracy
# corruption=fog, severity=2: 50.50% accuracy
# corruption=fog, severity=3: 45.76% accuracy
# corruption=fog, severity=4: 43.22% accuracy
# corruption=fog, severity=5: 36.64% accuracy
#  53%|███████████████████████████████████████████████████████████▋                                                    | 8/15 [08:14<07:08, 61.23s/it]corruption=zoom_blur, severity=1: 38.54% accuracy
# corruption=zoom_blur, severity=2: 31.70% accuracy
# corruption=zoom_blur, severity=3: 30.14% accuracy
# corruption=zoom_blur, severity=4: 24.12% accuracy
# corruption=zoom_blur, severity=5: 18.40% accuracy
#  60%|███████████████████████████████████████████████████████████████████▏                                            | 9/15 [09:15<06:07, 61.31s/it]corruption=frost, severity=1: 50.84% accuracy
# corruption=frost, severity=2: 41.56% accuracy
# corruption=frost, severity=3: 35.42% accuracy
# corruption=frost, severity=4: 33.84% accuracy
# corruption=frost, severity=5: 29.80% accuracy
#  67%|██████████████████████████████████████████████████████████████████████████                                     | 10/15 [10:21<05:13, 62.72s/it]corruption=glass_blur, severity=1: 49.46% accuracy
# corruption=glass_blur, severity=2: 40.26% accuracy
# corruption=glass_blur, severity=3: 24.46% accuracy
# corruption=glass_blur, severity=4: 20.40% accuracy
# corruption=glass_blur, severity=5: 15.58% accuracy
#  73%|█████████████████████████████████████████████████████████████████████████████████▍                             | 11/15 [11:21<04:07, 61.86s/it]corruption=impulse_noise, severity=1: 45.92% accuracy
# corruption=impulse_noise, severity=2: 40.88% accuracy
# corruption=impulse_noise, severity=3: 36.88% accuracy
# corruption=impulse_noise, severity=4: 28.74% accuracy
# corruption=impulse_noise, severity=5: 21.00% accuracy
#  80%|████████████████████████████████████████████████████████████████████████████████████████▊                      | 12/15 [12:28<03:10, 63.43s/it]corruption=contrast, severity=1: 56.34% accuracy
# corruption=contrast, severity=2: 54.24% accuracy
# corruption=contrast, severity=3: 50.40% accuracy
# corruption=contrast, severity=4: 41.06% accuracy
# corruption=contrast, severity=5: 25.92% accuracy
#  87%|████████████████████████████████████████████████████████████████████████████████████████████████▏              | 13/15 [13:29<02:05, 62.77s/it]corruption=jpeg_compression, severity=1: 55.08% accuracy
# corruption=jpeg_compression, severity=2: 51.58% accuracy
# corruption=jpeg_compression, severity=3: 48.68% accuracy
# corruption=jpeg_compression, severity=4: 40.86% accuracy
# corruption=jpeg_compression, severity=5: 31.96% accuracy
#  93%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌       | 14/15 [14:31<01:02, 62.34s/it]corruption=elastic_transform, severity=1: 53.22% accuracy
# corruption=elastic_transform, severity=2: 35.34% accuracy
# corruption=elastic_transform, severity=3: 56.00% accuracy
# corruption=elastic_transform, severity=4: 52.92% accuracy
# corruption=elastic_transform, severity=5: 41.18% accuracy
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [15:34<00:00, 62.30s/it]
# Adversarial accuracy: 39.92%
# /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'robustbench.eval' found in sys.modules after import of package 'robustbench', but prior to execution of 'robustbench.eval'; this may result in unpredictable behaviour
#   warn(RuntimeWarning(msg))
# Traceback (most recent call last):
#   File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
#     return _run_code(code, main_globals, None,
#   File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
#     exec(code, run_globals)
#   File "/tmldata1/andriush/robustbench/robustbench/eval.py", line 237, in <module>
#     main(args_)
#   File "/tmldata1/andriush/robustbench/robustbench/eval.py", line 211, in main
#     model = load_model(args.model_name,
#   File "/tmldata1/andriush/robustbench/robustbench/utils.py", line 132, in load_model
#     if models[model_name]['gdrive_id'] is None:
# KeyError: 'Geirhos2018_SIN_and_IN'
# /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'robustbench.eval' found in sys.modules after import of package 'robustbench', but prior to execution of 'robustbench.eval'; this may result in unpredictable behaviour
#   warn(RuntimeWarning(msg))
# Traceback (most recent call last):
#   File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
#     return _run_code(code, main_globals, None,
#   File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
#     exec(code, run_globals)
#   File "/tmldata1/andriush/robustbench/robustbench/eval.py", line 237, in <module>
#     main(args_)
#   File "/tmldata1/andriush/robustbench/robustbench/eval.py", line 211, in main
#     model = load_model(args.model_name,
#   File "/tmldata1/andriush/robustbench/robustbench/utils.py", line 132, in load_model
#     if models[model_name]['gdrive_id'] is None:
# KeyError: 'Geirhos2018_SIN_and_IN_then_finetuned_on_IN'
# /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'robustbench.eval' found in sys.modules after import of package 'robustbench', but prior to execution of 'robustbench.eval'; this may result in unpredictable behaviour
#   warn(RuntimeWarning(msg))
# Clean accuracy: 76.98%
# Evaluating over 15 corruptions
#   0%|                                                                                                                        | 0/15 [00:00<?, ?it/s]corruption=shot_noise, severity=1: 68.40% accuracy
# corruption=shot_noise, severity=2: 63.46% accuracy
# corruption=shot_noise, severity=3: 57.56% accuracy
# corruption=shot_noise, severity=4: 44.16% accuracy
# corruption=shot_noise, severity=5: 33.08% accuracy
#   7%|███████▍                                                                                                        | 1/15 [01:07<15:44, 67.47s/it]corruption=motion_blur, severity=1: 70.50% accuracy
# corruption=motion_blur, severity=2: 65.08% accuracy
# corruption=motion_blur, severity=3: 54.90% accuracy
# corruption=motion_blur, severity=4: 40.90% accuracy
# corruption=motion_blur, severity=5: 30.60% accuracy
#  13%|██████████████▉                                                                                                 | 2/15 [02:09<13:57, 64.43s/it]corruption=snow, severity=1: 63.60% accuracy
# corruption=snow, severity=2: 45.72% accuracy
# corruption=snow, severity=3: 48.44% accuracy
# corruption=snow, severity=4: 37.02% accuracy
# corruption=snow, severity=5: 28.60% accuracy
#  20%|██████████████████████▍                                                                                         | 3/15 [03:13<12:47, 63.95s/it]corruption=pixelate, severity=1: 69.28% accuracy
# corruption=pixelate, severity=2: 66.68% accuracy
# corruption=pixelate, severity=3: 57.46% accuracy
# corruption=pixelate, severity=4: 41.76% accuracy
# corruption=pixelate, severity=5: 31.86% accuracy
#  27%|█████████████████████████████▊                                                                                  | 4/15 [04:16<11:42, 63.86s/it]corruption=gaussian_noise, severity=1: 68.88% accuracy
# corruption=gaussian_noise, severity=2: 65.24% accuracy
# corruption=gaussian_noise, severity=3: 57.58% accuracy
# corruption=gaussian_noise, severity=4: 47.06% accuracy
# corruption=gaussian_noise, severity=5: 30.82% accuracy
#  33%|█████████████████████████████████████▎                                                                          | 5/15 [05:24<10:53, 65.30s/it]corruption=defocus_blur, severity=1: 64.00% accuracy
# corruption=defocus_blur, severity=2: 58.34% accuracy
# corruption=defocus_blur, severity=3: 47.18% accuracy
# corruption=defocus_blur, severity=4: 37.58% accuracy
# corruption=defocus_blur, severity=5: 26.96% accuracy
#  40%|████████████████████████████████████████████▊                                                                   | 6/15 [06:26<09:38, 64.23s/it]corruption=brightness, severity=1: 75.58% accuracy
# corruption=brightness, severity=2: 74.36% accuracy
# corruption=brightness, severity=3: 72.04% accuracy
# corruption=brightness, severity=4: 68.34% accuracy
# corruption=brightness, severity=5: 63.36% accuracy
#  47%|████████████████████████████████████████████████████▎                                                           | 7/15 [07:31<08:34, 64.26s/it]corruption=fog, severity=1: 66.30% accuracy
# corruption=fog, severity=2: 61.84% accuracy
# corruption=fog, severity=3: 53.94% accuracy
# corruption=fog, severity=4: 48.88% accuracy
# corruption=fog, severity=5: 33.20% accuracy
#  53%|███████████████████████████████████████████████████████████▋                                                    | 8/15 [08:34<07:28, 64.12s/it]corruption=zoom_blur, severity=1: 60.08% accuracy
# corruption=zoom_blur, severity=2: 52.82% accuracy
# corruption=zoom_blur, severity=3: 49.30% accuracy
# corruption=zoom_blur, severity=4: 41.80% accuracy
# corruption=zoom_blur, severity=5: 34.16% accuracy
#  60%|███████████████████████████████████████████████████████████████████▏                                            | 9/15 [09:37<06:21, 63.62s/it]corruption=frost, severity=1: 68.64% accuracy
# corruption=frost, severity=2: 57.84% accuracy
# corruption=frost, severity=3: 48.26% accuracy
# corruption=frost, severity=4: 46.76% accuracy
# corruption=frost, severity=5: 39.90% accuracy
#  67%|██████████████████████████████████████████████████████████████████████████                                     | 10/15 [10:39<05:15, 63.13s/it]corruption=glass_blur, severity=1: 61.84% accuracy
# corruption=glass_blur, severity=2: 51.16% accuracy
# corruption=glass_blur, severity=3: 29.36% accuracy
# corruption=glass_blur, severity=4: 24.88% accuracy
# corruption=glass_blur, severity=5: 20.26% accuracy
#  73%|█████████████████████████████████████████████████████████████████████████████████▍                             | 11/15 [11:39<04:08, 62.00s/it]corruption=impulse_noise, severity=1: 66.26% accuracy
# corruption=impulse_noise, severity=2: 61.44% accuracy
# corruption=impulse_noise, severity=3: 57.94% accuracy
# corruption=impulse_noise, severity=4: 46.74% accuracy
# corruption=impulse_noise, severity=5: 32.22% accuracy
#  80%|████████████████████████████████████████████████████████████████████████████████████████▊                      | 12/15 [12:45<03:10, 63.35s/it]corruption=contrast, severity=1: 71.08% accuracy
# corruption=contrast, severity=2: 68.62% accuracy
# corruption=contrast, severity=3: 62.04% accuracy
# corruption=contrast, severity=4: 42.94% accuracy
# corruption=contrast, severity=5: 17.60% accuracy
#  87%|████████████████████████████████████████████████████████████████████████████████████████████████▏              | 13/15 [13:45<02:04, 62.34s/it]corruption=jpeg_compression, severity=1: 69.78% accuracy
# corruption=jpeg_compression, severity=2: 67.44% accuracy
# corruption=jpeg_compression, severity=3: 65.82% accuracy
# corruption=jpeg_compression, severity=4: 61.22% accuracy
# corruption=jpeg_compression, severity=5: 53.06% accuracy
#  93%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌       | 14/15 [14:50<01:03, 63.02s/it]corruption=elastic_transform, severity=1: 69.46% accuracy
# corruption=elastic_transform, severity=2: 49.14% accuracy
# corruption=elastic_transform, severity=3: 62.20% accuracy
# corruption=elastic_transform, severity=4: 51.88% accuracy
# corruption=elastic_transform, severity=5: 30.74% accuracy
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [15:52<00:00, 63.52s/it]
# Adversarial accuracy: 52.47%
# /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'robustbench.eval' found in sys.modules after import of package 'robustbench', but prior to execution of 'robustbench.eval'; this may result in unpredictable behaviour
#   warn(RuntimeWarning(msg))
# Clean accuracy: 76.90%
# Evaluating over 15 corruptions
#   0%|                                                                                                                        | 0/15 [00:00<?, ?it/s]corruption=shot_noise, severity=1: 69.44% accuracy
# corruption=shot_noise, severity=2: 64.40% accuracy
# corruption=shot_noise, severity=3: 57.12% accuracy
# corruption=shot_noise, severity=4: 41.26% accuracy
# corruption=shot_noise, severity=5: 30.88% accuracy
#   7%|███████▍                                                                                                        | 1/15 [01:33<21:46, 93.35s/it]corruption=motion_blur, severity=1: 70.92% accuracy
# corruption=motion_blur, severity=2: 65.58% accuracy
# corruption=motion_blur, severity=3: 55.98% accuracy
# corruption=motion_blur, severity=4: 42.88% accuracy
# corruption=motion_blur, severity=5: 33.42% accuracy
#  13%|██████████████▉                                                                                                 | 2/15 [03:17<21:38, 99.86s/it]corruption=snow, severity=1: 64.20% accuracy
# corruption=snow, severity=2: 45.44% accuracy
# corruption=snow, severity=3: 47.84% accuracy
# corruption=snow, severity=4: 35.86% accuracy
# corruption=snow, severity=5: 27.32% accuracy
#  20%|██████████████████████▍                                                                                         | 3/15 [04:41<18:29, 92.48s/it]corruption=pixelate, severity=1: 70.26% accuracy
# corruption=pixelate, severity=2: 68.18% accuracy
# corruption=pixelate, severity=3: 57.40% accuracy
# corruption=pixelate, severity=4: 41.74% accuracy
# corruption=pixelate, severity=5: 31.58% accuracy
#  27%|█████████████████████████████▊                                                                                  | 4/15 [05:53<15:30, 84.57s/it]corruption=gaussian_noise, severity=1: 70.38% accuracy
# corruption=gaussian_noise, severity=2: 66.18% accuracy
# corruption=gaussian_noise, severity=3: 57.64% accuracy
# corruption=gaussian_noise, severity=4: 46.10% accuracy
# corruption=gaussian_noise, severity=5: 29.18% accuracy
#  33%|█████████████████████████████████████▎                                                                          | 5/15 [07:12<13:45, 82.54s/it]corruption=defocus_blur, severity=1: 64.42% accuracy
# corruption=defocus_blur, severity=2: 58.96% accuracy
# corruption=defocus_blur, severity=3: 48.02% accuracy
# corruption=defocus_blur, severity=4: 38.18% accuracy
# corruption=defocus_blur, severity=5: 28.44% accuracy
#  40%|████████████████████████████████████████████▊                                                                   | 6/15 [08:21<11:39, 77.78s/it]corruption=brightness, severity=1: 75.26% accuracy
# corruption=brightness, severity=2: 74.00% accuracy
# corruption=brightness, severity=3: 72.42% accuracy
# corruption=brightness, severity=4: 69.26% accuracy
# corruption=brightness, severity=5: 63.98% accuracy
#  47%|████████████████████████████████████████████████████▎                                                           | 7/15 [09:29<09:56, 74.50s/it]corruption=fog, severity=1: 68.04% accuracy
# corruption=fog, severity=2: 63.40% accuracy
# corruption=fog, severity=3: 55.84% accuracy
# corruption=fog, severity=4: 51.46% accuracy
# corruption=fog, severity=5: 37.44% accuracy
#  53%|███████████████████████████████████████████████████████████▋                                                    | 8/15 [10:36<08:26, 72.38s/it]corruption=zoom_blur, severity=1: 61.90% accuracy
# corruption=zoom_blur, severity=2: 55.22% accuracy
# corruption=zoom_blur, severity=3: 52.04% accuracy
# corruption=zoom_blur, severity=4: 44.48% accuracy
# corruption=zoom_blur, severity=5: 37.92% accuracy
#  60%|███████████████████████████████████████████████████████████████████▏                                            | 9/15 [11:47<07:11, 71.85s/it]corruption=frost, severity=1: 68.82% accuracy
# corruption=frost, severity=2: 58.06% accuracy
# corruption=frost, severity=3: 48.36% accuracy
# corruption=frost, severity=4: 45.56% accuracy
# corruption=frost, severity=5: 40.06% accuracy
#  67%|██████████████████████████████████████████████████████████████████████████                                     | 10/15 [12:58<05:57, 71.55s/it]corruption=glass_blur, severity=1: 62.90% accuracy
# corruption=glass_blur, severity=2: 53.48% accuracy
# corruption=glass_blur, severity=3: 31.42% accuracy
# corruption=glass_blur, severity=4: 26.84% accuracy
# corruption=glass_blur, severity=5: 21.94% accuracy
#  73%|█████████████████████████████████████████████████████████████████████████████████▍                             | 11/15 [14:05<04:40, 70.24s/it]corruption=impulse_noise, severity=1: 67.82% accuracy
# corruption=impulse_noise, severity=2: 62.74% accuracy
# corruption=impulse_noise, severity=3: 58.20% accuracy
# corruption=impulse_noise, severity=4: 45.52% accuracy
# corruption=impulse_noise, severity=5: 29.78% accuracy
#  80%|████████████████████████████████████████████████████████████████████████████████████████▊                      | 12/15 [15:25<03:39, 73.17s/it]corruption=contrast, severity=1: 72.10% accuracy
# corruption=contrast, severity=2: 69.30% accuracy
# corruption=contrast, severity=3: 63.98% accuracy
# corruption=contrast, severity=4: 48.42% accuracy
# corruption=contrast, severity=5: 25.04% accuracy
#  87%|████████████████████████████████████████████████████████████████████████████████████████████████▏              | 13/15 [16:34<02:23, 71.97s/it]corruption=jpeg_compression, severity=1: 69.98% accuracy
# corruption=jpeg_compression, severity=2: 67.62% accuracy
# corruption=jpeg_compression, severity=3: 66.32% accuracy
# corruption=jpeg_compression, severity=4: 60.54% accuracy
# corruption=jpeg_compression, severity=5: 54.26% accuracy
#  93%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌       | 14/15 [17:44<01:11, 71.13s/it]corruption=elastic_transform, severity=1: 69.26% accuracy
# corruption=elastic_transform, severity=2: 49.06% accuracy
# corruption=elastic_transform, severity=3: 62.58% accuracy
# corruption=elastic_transform, severity=4: 52.94% accuracy
# corruption=elastic_transform, severity=5: 31.54% accuracy
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [18:55<00:00, 75.70s/it]
# Adversarial accuracy: 53.28%
# ./eval_corruptions_2d.sh: line 6: syntax error near unexpected token `done'
# ./eval_corruptions_2d.sh: line 6: `done'
# andriush@overfit-pod-1-1-0-0:/tmldata1/andriush/robustbench$ %
# ➜  overfit git:(main) ✗