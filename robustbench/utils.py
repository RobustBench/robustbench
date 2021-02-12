import argparse
import json
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests
import torch
from scipy import stats
from torch import nn

from robustbench.model_zoo import model_dicts as all_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel


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

    print('Download started: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print('Download finished: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def load_model(model_name: str,
               model_dir: Union[str, Path] = './models',
               dataset: Union[str,
                              BenchmarkDataset] = BenchmarkDataset.cifar_10,
               threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
               norm: Optional[str] = None) -> nn.Module:

    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    if norm is None:
        threat_model_: ThreatModel = ThreatModel(threat_model)
    else:
        threat_model_: ThreatModel = ThreatModel(norm)
        warnings.warn(
            "`norm` has been deprecated and will be removed in a future version.",
            DeprecationWarning)

    model_dir_ = Path(model_dir) / dataset_.value / threat_model_.value
    model_path = model_dir_ / f'{model_name}.pt'

    models = all_models[dataset_][threat_model_]

    if not isinstance(models[model_name]['gdrive_id'], list):
        model = models[model_name]['model']()
        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        if not os.path.isfile(model_path):
            download_gdrive(models[model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # needed for the model of `Carmon2019Unlabeled`
        try:
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'],
                                                   'module.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')

        model.load_state_dict(state_dict, strict=True)
        return model.eval()

    # If we have an ensemble of models (e.g., Chen2020Adversarial)
    else:
        model = models[model_name]['model']()
        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        for i, gid in enumerate(models[model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i),
                                    map_location=torch.device('cpu'))
            try:
                state_dict = rm_substr_from_state_dict(
                    checkpoint['state_dict'], 'module.')
            except KeyError:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            model.models[i].load_state_dict(state_dict, strict=True)
            model.models[i].eval()
        return model.eval()


def clean_accuracy(model, x, y, batch_size=100, device=None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]


def list_available_models(
        dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
        threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
        norm: Optional[str] = None):
    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)

    if norm is None:
        threat_model_: ThreatModel = ThreatModel(threat_model)
    else:
        threat_model_: ThreatModel = ThreatModel(norm)
        warnings.warn(
            "`norm` has been deprecated and will be removed in a future version.",
            DeprecationWarning)

    models = all_models[dataset_][threat_model_].keys()

    json_dicts = []

    jsons_dir = Path("./model_info") / dataset_.value / threat_model_.value

    for model_name in models:
        json_path = jsons_dir / f"{model_name}.json"

        # Some models might not yet be in model_info
        if not json_path.exists():
            continue

        with open(json_path, 'r') as model_info:
            json_dict = json.load(model_info)

        json_dict['model_name'] = model_name
        json_dict['venue'] = 'Unpublished' if json_dict[
            'venue'] == '' else json_dict['venue']
        json_dict['AA'] = float(json_dict['AA']) / 100
        json_dict['clean_acc'] = float(json_dict['clean_acc']) / 100
        json_dicts.append(json_dict)

    json_dicts = sorted(json_dicts, key=lambda d: -d['AA'])
    print(
        '| # | Model ID | Paper | Clean accuracy | Robust accuracy | Architecture | Venue |'
    )
    print('|:---:|---|---|:---:|:---:|:---:|:---:|')
    for i, json_dict in enumerate(json_dicts):
        if json_dict['model_name'] == 'Chen2020Adversarial':
            json_dict['architecture'] = json_dict[
                'architecture'] + ' <br/> (3x ensemble)'
        if json_dict['model_name'] != 'Natural':
            print(
                '| <sub>**{}**</sub> | <sub>**{}**</sub> | <sub>*[{}]({})*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'
                .format(i + 1, json_dict['model_name'], json_dict['name'],
                        json_dict['link'], json_dict['clean_acc'],
                        json_dict['AA'], json_dict['architecture'],
                        json_dict['venue']))
        else:
            print(
                '| <sub>**{}**</sub> | <sub>**{}**</sub> | <sub>*{}*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'
                .format(i + 1, json_dict['model_name'], json_dict['name'],
                        json_dict['clean_acc'], json_dict['AA'],
                        json_dict['architecture'], json_dict['venue']))


def update_json(dataset: BenchmarkDataset, threat_model: ThreatModel,
                model_name: str, accuracy: float, adv_accuracy: float,
                eps: Optional[float]) -> None:
    json_path = Path(
        "model_info"
    ) / dataset.value / threat_model.value / f"{model_name}.json"
    model_info = {
        "link": None,
        "name": None,
        "authors": None,
        "additional_data": None,
        "number_forward_passes": None,
        "dataset": dataset.value,
        "venue": None,
        "architecture": None,
        "eps": eps,
        "clean_acc": accuracy,
        "reported": None,
        "AA": adv_accuracy
    }

    with open(json_path, "w") as f:
        f.write(json.dumps(model_info, indent=2))


def severity_weighted_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0, axis=1).agg(
        lambda x: np.average(x, weights=x.columns.get_level_values(1), axis=1))


def severity_harmonic_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0, axis=1).agg(lambda x: stats.hmean(x, axis=1))


def severity_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0, axis=1).mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='Carmon2019Unlabeled')
    parser.add_argument('--threat_model',
                        type=str,
                        default='Linf',
                        choices=[x.value for x in ThreatModel])
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=[x.value for x in BenchmarkDataset])
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--n_ex',
                        type=int,
                        default=100,
                        help='number of examples to evaluate on')
    parser.add_argument('--batch_size',
                        type=int,
                        default=500,
                        help='batch size for evaluation')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data',
                        help='where to store downloaded datasets')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./models',
                        help='where to store downloaded models')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device to use for computations')
    parser.add_argument('--to_disk', type=bool, default=True)
    args = parser.parse_args()
    return args
