### Working code -- all 3D corruptions -- save them as pickle files
import pickle as pkl
import torch 
from robustbench.data import load_imagenetc
from robustbench.utils import clean_accuracy
from robustbench.utils import load_model

#corruptions_to_test_2d = ['jpeg_compression', 'pixelate', 'defocus_blur', 'defocus_blur', 'brightness', 'fog', 'jpeg_compression', 'jpeg_compression', 'shot_noise', 'impulse_noise', 'motion_blur', 'zoom_blur']
corruptions_to_test_3d = ['bit_error', 'color_quant', 'near_focus', 'far_focus', 'flash', 'fog_3d', 'h265_abr', 'h265_crf', 'iso_noise', 'low_light', 'xy_motion_blur', 'z_motion_blur']

device = torch.device("cuda:0")

all_corruptions = corruptions_to_test_3d

for model_name in ['Hendrycks2020Many', 'Hendrycks2020AugMix', 'Geirhos2018_SIN_IN', 'Geirhos2018_SIN_IN_IN', 'Standard_R50', 'Geirhos2018_SIN', 'Salman2020Do_50_2_Linf']:
#for model_name in ['Geirhos2018_SIN_IN', 'Geirhos2018_SIN_IN_IN', 'Standard_R50', 'Geirhos2018_SIN', 'Salman2020Do_50_2_Linf']:    
    print(model_name)
    corrs_all = {}
    model = load_model(model_name, dataset='imagenet', threat_model='corruptions')
    model.to(device)
    for corruption in all_corruptions:
        corrs_curr = []
        for s in [1,2,3,4,5]:
            corruptions = [corruption]
            x_test, y_test = load_imagenetc(n_examples=5000, corruptions=corruptions, severity=s, data_dir='/datasets/home/oguzhan/release_3dcc')
            x_test, y_test = x_test.to(device), y_test.to(device)
        
            acc = clean_accuracy(model, x_test, y_test, device=device)
            print(f'Model: {model_name}, ImageNet-C corruption: {corruption} severity: {s} accuracy: {acc:.1%}')
            corrs_curr.append(acc)
        corrs_all[corruption] = corrs_curr
        pkl.dump(corrs_all,open(f'eval_3dcc_{model_name}.pkl','wb'))
