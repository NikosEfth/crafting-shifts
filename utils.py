import torch
import os
import numpy as np
import csv
import yaml
import random
import sys


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_device(gpu_id):
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {gpu_id}")
    else:
        print(f"GPU {gpu_id} not available. Using CPU instead.")
        device = torch.device("cpu")
    return device


def try_make_dir(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        print("Folder was created by another process")


def vit_normalization(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "name" and value == "normalize":
                data["mean"] = [0.5, 0.5, 0.5]
                data["std"] = [0.5, 0.5, 0.5]
            else:
                vit_normalization(value)
    elif isinstance(data, list):
        for item in data:
            vit_normalization(item)


def generate_points(range_tuple, points, log_scale=False):
    start, end = range_tuple
    if log_scale:
        if start == 0:
            start = sys.float_info.min
        points = np.logspace(np.log10(start), np.log10(end), points)
    else:
        points = np.linspace(start, end, points)
    return list(points)


def add_free_log(data, save_dir):
    final_data = []
    for i, col in enumerate(data):
        if col != []:
            final_data.append(data[i])
    rows = zip(*final_data)
    try:
        with open(save_dir, "a", newline="") as file:
            writer = csv.writer(file)
            for row in rows:
                writer.writerow(row)
        return ()
    except PermissionError:
        return ()


def report_time(seconds):
    seconds = round(seconds)
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    return f"Hours: {hours} Min: {minutes} Sec: {seconds}"


def float_tuple(value):
    try:
        if isinstance(value, float):
            value = (value, value)
        elif isinstance(value, str):
            value = (float(value), float(value))
        elif isinstance(value, list):
            if len(value) < 2:
                value = (float(value[0]), float(value[0]))
            else:
                value = tuple([float(i) for i in value])
        return value
    except ValueError:
        print("input cannot change to float")


def dirlist(dirs):
    if dirs == "None":
        return None, None

    if isinstance(dirs, str):
        dirs = [dirs]
    images = []
    classes = []
    for current_dir in dirs:
        with open(current_dir, mode="r", encoding="utf-8") as file:
            for line in file:
                if line != "":
                    if len(line.split()) > 1:
                        images.append(line.split()[0])
                        classes.append(str(line.split()[1]))
    return images, classes


def save_architecture(network, direct, name="architecture"):
    file_path = os.path.join(direct, f"{name}.txt")
    if os.path.exists(file_path):
        return
    with open(file_path, "w") as file:
        print(network, file=file)
        print("", file=file)
        for param_name, param in network.named_parameters():
            print(param_name, param.requires_grad, file=file)


def save_model(network, optimizer, epoch, direct, is_nan=False):
    if is_nan:
        state = {"epoch": None, "state_dict": None, "optimizer": None}
    else:
        state = {
            "epoch": epoch,
            "state_dict": network.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
    torch.save(state, direct)
    return


def load_model(network, model_location):
    device = next(network.parameters()).device
    state = torch.load(model_location, map_location=device)
    if state["state_dict"] is not None:
        network.load_state_dict(state["state_dict"])
    network.eval()


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def nan_in_grad(model):
    found_nan = False
    try:
        for idx, param in enumerate(model.backbone.parameters()):
            if isinstance(param.grad, torch.Tensor):
                if param.grad.isnan().any():
                    found_nan = True
    except:
        print("Could not calculate gradient")
    return found_nan


def save_yaml(structure, direct):
    with open(direct, "w") as file:
        documents = yaml.dump(structure, file, width=10000, Dumper=NoAliasDumper)


def read_yaml(direct):
    with open(direct) as file:
        structure = yaml.full_load(file)
    return structure


def initialize_csv_file(loader_info, csv_file_name, test_idx_list, domains):
    if not os.path.isfile(csv_file_name):
        headers = (
            [["lr"]]
            + [["method_loss"]]
            + [[name] for name in loader_info["output_names_val"]]
            + [[f"Imgaug_average_{name}"] for name in loader_info["output_names_val"]]
            + [[f"test_{name}"] for name in loader_info["output_names_val"]]
            + [
                [f"{val}_{name}" for name in loader_info["output_names_val"]]
                for val in loader_info["pd_names_val_only"]
            ]
            + [
                [f"{domains[idx]}_{name}" for name in loader_info["output_names_val"]]
                for idx in test_idx_list
            ]
        )
        add_free_log(
            data=[[item] for sublist in headers for item in sublist],
            save_dir=csv_file_name,
        )
