import json
from argparse import ArgumentParser
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape


def generate_leaderboard(folder_name: str) -> str:
    """Prints the HTML leaderboard starting from the .json results.

    The result is a <table> that can be put directly into the RobustBench index.html page,
    and looks the same as the tables that are already existing.

    The .json results must have the same structure as the following:
    ``
    {
      "link": "https://arxiv.org/abs/2003.09461",
      "name": "Adversarial Robustness on In- and Out-Distribution Improves Explainability",
      "authors": "Maximilian Augustin, Alexander Meinke, Matthias Hein",
      "additional_data": true,
      "number_forward_passes": 1,
      "dataset": "cifar10",
      "venue": "ECCV 2020",
      "architecture": "ResNet-50",
      "eps": "0.5",
      "clean_acc": "91.08",
      "reported": "73.27",
      "AA": "72.91"
    }
    ``

    :param folder_name: the name of the folder where the .json files are placed.
    :return: The resulting table.
    """
    folder = Path(folder_name)

    models = []

    for model_path in folder.glob("*.json"):
        with open(model_path) as fp:
            model = json.load(fp)

        models.append(model)

    models.sort(key=lambda x: x["AA"], reverse=True)

    env = Environment(
        loader=PackageLoader('robustbench', 'leaderboard'),
        autoescape=select_autoescape(['html', 'xml'])
    )

    template = env.get_template('leaderboard.html.j2')
    result = template.render(models=models)
    print(result)
    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--models_folder", type=str,
                        help="The folder containing the .json file with the models information")
    args = parser.parse_args()

    generate_leaderboard(args.models_folder)
