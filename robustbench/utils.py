import os
import json
import math
import requests
import torch
from collections import OrderedDict


def download_gdrive(gdrive_id, fname_save):
    """ source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('Download started: path={} (gdrive_id={})'.format(fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print('Download finished: path={} (gdrive_id={})'.format(fname_save, gdrive_id))


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def load_model(model_name, model_dir='./models', norm='Linf'):
    from .model_zoo.models import model_dicts as all_models
    model_dir += '/{}'.format(norm)
    model_path = '{}/{}.pt'.format(model_dir, model_name)
    model_dicts = all_models[norm]
    if not isinstance(model_dicts[model_name]['gdrive_id'], list):
        model = model_dicts[model_name]['model']()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.isfile(model_path):
            download_gdrive(model_dicts[model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location='cuda')
    
        # needed for the model of `Carmon2019Unlabeled`
        try:
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
        
        model.load_state_dict(state_dict)
        return model.cuda().eval()

    # If we have an ensemble of models (e.g., Chen2020Adversarial)
    else:
        model = model_dicts[model_name]['model']()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for i, gid in enumerate(model_dicts[model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i), map_location='cuda')
            try:
                state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')
            except:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            model.models[i].load_state_dict(state_dict)
            model.models[i].cuda().eval()
        return model.cuda().eval()


def clean_accuracy(model, x, y, batch_size=100):
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].cuda()
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].cuda()

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]


def list_available_models(norm='Linf'):
    from .model_zoo.models import model_dicts
    models = model_dicts[norm].keys()

    json_dicts = []
    for model_name in models:
        with open('./model_info/{}/{}.json'.format(norm, model_name), 'r') as model_info:
            json_dict = json.load(model_info)
        json_dict['model_name'] = model_name
        json_dict['venue'] = 'Unpublished' if json_dict['venue'] == '' else json_dict['venue']
        json_dict['AA'] = float(json_dict['AA']) / 100
        json_dict['clean_acc'] = float(json_dict['clean_acc']) / 100
        json_dicts.append(json_dict)

    json_dicts = sorted(json_dicts, key=lambda d: -d['AA'])
    print('| # | Model ID | Paper | Clean accuracy | Robust accuracy | Architecture | Venue |')
    print('|:---:|---|---|:---:|:---:|:---:|:---:|')
    for i, json_dict in enumerate(json_dicts):
        if json_dict['model_name'] == 'Chen2020Adversarial':
            json_dict['architecture'] = json_dict['architecture'] + ' <br/> (3x ensemble)'
        if json_dict['model_name'] != 'Natural':
            print('| <sub>**{}**</sub> | <sub>**{}**</sub> | <sub>*[{}]({})*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'.format(
                i+1, json_dict['model_name'], json_dict['name'], json_dict['link'], json_dict['clean_acc'], json_dict['AA'],
                json_dict['architecture'], json_dict['venue']))
        else:
            print('| <sub>**{}**</sub> | <sub>**{}**</sub> | <sub>*{}*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'.format(
                i + 1, json_dict['model_name'], json_dict['name'], json_dict['clean_acc'],
                json_dict['AA'], json_dict['architecture'], json_dict['venue']))

