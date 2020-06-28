import os
import requests
import torch
from collections import OrderedDict
from model_zoo.models import model_dicts


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


def load_model(model_name, model_dir):
    model_path = '{}/{}.pt'.format(model_dir, model_name)
    model = model_dicts[model_name]['model']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.isfile(model_path):
        download_gdrive(model_dicts[model_name]['gdrive_id'], model_path)
    checkpoint = torch.load(model_path)

    # needed for the model of `carmon2019unlabeled`
    state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')

    model.load_state_dict(state_dict)
    return model.cuda().eval()

