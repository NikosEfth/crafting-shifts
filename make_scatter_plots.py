import os
import csv
import argparse
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description="Training and search specifications")
    parser.add_argument(
        "--dataset", type=str, help="Choose dataset for a total scatter"
    )
    return parser.parse_args()


def find_files_in_directories(file_name, directories):
    files = []

    for directory in directories:
        for root, _, filenames in os.walk(directory):
            for file in filenames:
                if file == file_name:
                    files.append((os.path.join(root, file), directory))
    return files


def read_csv_file(file_path):
    column1 = []
    column2 = []
    headers = []

    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            if len(row) == 2:
                column1.append(float(row[0]))
                column2.append(float(row[1]))

    return column1, column2, headers


def plot_scatter(data, directories, save_path, headers):
    unique_directories = sorted(set(directories))
    colors = [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "brown",
        "pink",
    ]
    while len(colors) < len(unique_directories):
        colors = colors + colors

    total_x_vals, total_y_vals = [], []
    for i, directory in enumerate(unique_directories):
        dir_data = [(x, y) for (x, y, dir) in data if dir == directory]
        x_vals = [x for (x, y) in dir_data]
        y_vals = [y for (x, y) in dir_data]
        total_x_vals.extend(x_vals)
        total_y_vals.extend(y_vals)
        plt.scatter(x_vals, y_vals, color=colors[i], label=directory.split("/")[-1])
    correlation, _ = spearmanr(total_x_vals, total_y_vals)
    plt.title(f"Spearman Correlation: {round(correlation, 2)}")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Scatter plot saved to {save_path}")


def get_folders_starting_with(base_directory, prefix):
    matching_folders = []

    for item in os.listdir(base_directory):
        full_path = os.path.join(base_directory, item)

        if os.path.isdir(full_path) and item.startswith(prefix):
            matching_folders.append(item)

    return matching_folders


def validation_scatter(file_name, directories, save_path):
    files = find_files_in_directories(file_name=file_name, directories=directories)
    all_data = []
    dir_labels = []

    for file_path, directory in files:
        column1, column2, file_headers = read_csv_file(file_path=file_path)
        headers = file_headers
        all_data.extend([(x, y, directory) for x, y in zip(column1, column2)])
        dir_labels.extend([directory] * len(column1))

    if all_data:
        plot_scatter(
            data=all_data, directories=dir_labels, save_path=save_path, headers=headers
        )


def main():
    args = parse_args()
    directories = get_folders_starting_with(
        base_directory="./Results", prefix=args.dataset
    )
    read_csvs = [
        "Scatter_Cross_val_Imgaug_average.csv",
        "Scatter_Standard.csv",
    ]
    directories = [f"./Results/{x}" for x in directories]
    for file in read_csvs:
        validation_scatter(
            file_name=file,
            directories=directories,
            save_path=os.path.join(
                ".",
                "Results",
                f"{args.dataset}_{file.split('.')[0]}.png",
            ),
        )


if __name__ == "__main__":
    main()
