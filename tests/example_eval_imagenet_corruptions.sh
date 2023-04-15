# for debugging
python -m robustbench.eval --n_ex=2 --dataset=imagenet --threat_model=corruptions --model_name=AlexNet --data_dir=/tmldata1/andriush/imagenet --corruptions_data_dir=/tmldata1/andriush/imagenet --batch_size=256 --to_disk=True
python -m robustbench.eval --n_ex=2 --dataset=imagenet --threat_model=corruptions_3d --model_name=AlexNet --data_dir=/tmldata1/andriush/imagenet --corruptions_data_dir=/tmldata1/andriush/data/3DCommonCorruptions/ --batch_size=256 --to_disk=True

# 2d corruptions
for model_name in AlexNet Standard_R50 Salman2020Do_50_2_Linf Hendrycks2020AugMix Hendrycks2020Many Geirhos2018_SIN Geirhos2018_SIN_IN Geirhos2018_SIN_IN_IN Erichson2022NoisyMix Erichson2022NoisyMix_new Tian2022Deeper_DeiT-S Tian2022Deeper_DeiT-B;
do
    python -m robustbench.eval --n_ex=5000 --dataset=imagenet --threat_model=corruptions --model_name=$model_name --data_dir=/tmldata1/andriush/imagenet --corruptions_data_dir=/tmldata1/andriush/imagenet --batch_size=256 --to_disk=True 
done

# 3d corruptions
for model_name in AlexNet Standard_R50 Salman2020Do_50_2_Linf Hendrycks2020AugMix Hendrycks2020Many Geirhos2018_SIN Geirhos2018_SIN_IN Geirhos2018_SIN_IN_IN Erichson2022NoisyMix Erichson2022NoisyMix_new Tian2022Deeper_DeiT-S Tian2022Deeper_DeiT-B;
do
    python -m robustbench.eval --n_ex=5000 --dataset=imagenet --threat_model=corruptions_3d --model_name=$model_name --data_dir=/tmldata1/andriush/imagenet --corruptions_data_dir=/tmldata1/andriush/data/3DCommonCorruptions/ --batch_size=256 --to_disk=True 
done

