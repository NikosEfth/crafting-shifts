import os
import pandas as pd
import pdb
import argparse
import math
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Training and search specifications")
    parser.add_argument("--domain", default="photo", type=str, help="source domain")
    parser.add_argument(
        "--backbone", default="resnet18", type=str, help="Choose the backbone"
    )
    parser.add_argument("--dataset", default="PACS", type=str, help="Choose dataset")
    parser.add_argument(
        "--seeds", nargs="+", type=int, help="choose which seeds to validate"
    )
    parser.add_argument(
        "--main_exp_name", type=str, help="eg imgaug_and_canny_training_all"
    )
    parser.add_argument(
        "--cv_exp_names",
        nargs="+",
        type=str,
        help="cross-val experiment names eg experiment_first and experiment_second",
    )
    return parser.parse_args()


def cross_val(args):
    main_exp_name = os.path.join(
        ".",
        "Results",
        f"{args.dataset}_{args.backbone}",
        args.main_exp_name,
    )
    cv_exp_names = [
        os.path.join(".", "Results", f"{args.dataset}_{args.backbone}", x)
        for x in args.cv_exp_names
    ]

    possible_imgaug_set = [
        "arithmetic",
        "artistic",
        "blur",
        "color",
        "contrast",
        "convolutional",
        "edges",
        "geometric",
        "segmentation",
        "weather",
    ]
    possible_pseudo_names = [
        "output_normal",
        "output_canny",
        "output_ImgAug",
        "output_25-75",
        "output_50-50",
        "output_75-25",
    ]
    imgaug_pseudo_names = [
        f"{imgaug}_{pseudo}"
        for imgaug in possible_imgaug_set
        for pseudo in possible_pseudo_names
    ]

    seed_dict = {}
    file_names = [f"Results_source_{args.domain}_seed_{idx}.csv" for idx in args.seeds]

    for cv_exp_name in cv_exp_names:
        for file_name in file_names:

            # Read the CSV file into a DataFrame and append it to the corresponding class data
            df = pd.read_csv(os.path.join(cv_exp_name, file_name))
            df["lr_method_loss"] = df.apply(
                lambda row: f"{row['lr']:.10f}_{row['method_loss']:.2f}", axis=1
            )
            df = df.sort_values("lr_method_loss").reset_index(drop=True)
            for imgaug_pseudo_name in imgaug_pseudo_names:
                if f"Imgaug_{imgaug_pseudo_name}" in df.columns:
                    if file_name not in seed_dict:
                        seed_dict[file_name] = df[f"Imgaug_{imgaug_pseudo_name}"]
                    else:
                        seed_dict[file_name] = pd.concat(
                            [seed_dict[file_name], df[f"Imgaug_{imgaug_pseudo_name}"]],
                            axis=1,
                        )

    for key, value in seed_dict.items():
        for pseudo in possible_pseudo_names:
            if any(seed_dict[key].columns.str.contains(pseudo)):
                seed_dict[key][f"Cross_val_Imgaug_average_{pseudo}"] = (
                    seed_dict[key].filter(like=pseudo).mean(axis=1)
                )

    for file_name in file_names:

        # Read the CSV file into a DataFrame and append it to the corresponding class data
        file_path = os.path.join(main_exp_name, file_name)
        df = pd.read_csv(file_path)
        df["lr_method_loss"] = df.apply(
            lambda row: f"{row['lr']:.10f}_{row['method_loss']:.2f}", axis=1
        )
        df = df.sort_values("lr_method_loss").reset_index(drop=True)

        seed_dict[file_name] = pd.concat([df, seed_dict[file_name]], axis=1)
        seed_dict[file_name].to_csv(
            os.path.join(main_exp_name, f"Cross-Val_{file_name}"), index=False
        )


def agregate(args):
    domain = args.domain
    main_exp_name = os.path.join(
        ".",
        "Results",
        f"{args.dataset}_{args.backbone}",
        args.main_exp_name,
    )
    possible_val_types = ["", "Imgaug_average", "Cross_val_Imgaug_average", "test"]
    possible_pseudo_names = [
        "output_normal",
        "output_canny",
        "output_ImgAug",
        "output_25-75",
        "output_50-50",
        "output_75-25",
    ]
    possible_domains = ["art_painting", "cartoon", "sketch", "photo"]

    seeds = args.seeds
    file_names = [f"Results_source_{domain}_seed_{idx}.csv" for idx in seeds]
    cross_val_names = [f"Cross-Val_{name}" for name in file_names]

    if all(
        os.path.exists(os.path.join(main_exp_name, file)) for file in cross_val_names
    ):
        file_names = cross_val_names

    result_tables = {}
    scatter = {}

    for file_name in file_names:

        df = pd.read_csv(os.path.join(main_exp_name, file_name))

        df["lr_method_loss"] = df.apply(
            lambda row: f"{row['lr']:.10f}_{row['method_loss']:.2f}", axis=1
        )
        df = df.sort_values("lr_method_loss").reset_index(drop=True)

        for val_set in possible_val_types:
            if f"{val_set}_val" not in scatter.keys() and val_set != "test":
                scatter[f"{val_set}_val"], scatter[f"{val_set}_test"] = [], []
            for pseudo in possible_pseudo_names:
                col_name = f"{val_set}_{pseudo}" if val_set != "" else pseudo

                if col_name in df.columns:
                    current_val_col = df[col_name]

                    argmax_index = current_val_col.idxmax()
                    max_value = current_val_col.loc[argmax_index]
                    max_rows = df[current_val_col == max_value]
                    middle_row = max_rows.iloc[
                        [
                            (
                                len(max_rows) // 2 - 1
                                if len(max_rows) % 2 == 0
                                else len(max_rows) // 2
                            )
                        ]
                    ]

                    column_names = (
                        ["lr_method_loss"]
                        + [col_name]
                        + [f"{x}_{pseudo}" for x in possible_domains]
                    )
                    valid_column_names = [
                        col for col in column_names if col in df.columns
                    ]
                    filtered_df = middle_row[valid_column_names].copy()

                    for name in valid_column_names:
                        if name != "lr_method_loss":
                            filtered_df[name] = filtered_df[name].apply(
                                lambda x: x * 100
                            )

                    filtered_df["test_average"] = filtered_df.iloc[:, 2:].mean(axis=1)
                    filtered_df["seed"] = file_name.split(".")[0].split("seed_")[1]

                    if val_set != "test":
                        scatter[f"{val_set}_val"].append(filtered_df[col_name].item())
                        scatter[f"{val_set}_test"].append(
                            filtered_df["test_average"].item()
                        )

                    if col_name not in result_tables:
                        result_tables[col_name] = filtered_df
                    else:
                        result_tables[col_name] = pd.concat(
                            [result_tables[col_name], filtered_df]
                        )

    scatter = remove_empty_keys(scatter)
    save_val_test_pairs(scatter, main_exp_name)

    for key, value in result_tables.items():
        result_tables[key] = (
            result_tables[key].sort_values("seed").reset_index(drop=True)
        )  # Sort and reset index
        numeric_columns = result_tables[key].select_dtypes(include="number").columns
        averages = result_tables[key][numeric_columns].mean()
        averages["lr_method_loss"] = ""
        averages["seed"] = "average"
        averages_df = pd.DataFrame([averages])
        result_tables[key] = pd.concat(
            [result_tables[key], averages_df], ignore_index=True
        )
        float_columns = result_tables[key].select_dtypes(include="float").columns
        result_tables[key][float_columns] = result_tables[key][float_columns].round(2)
        result_tables[key] = result_tables[key][
            result_tables[key].columns.tolist()[-1:]
            + result_tables[key].columns.tolist()[:-1]
        ]

    ods_data = {}
    ods_data["Tables"] = []
    for table in result_tables.values():
        ods_data["Tables"].append(table.columns.tolist())
        for value in table.values.tolist():
            ods_data["Tables"].append(value)
        ods_data["Tables"].append([])  # Add an empty row for separation

    df_out = pd.DataFrame(ods_data["Tables"][1:], columns=ods_data["Tables"][0])
    df_out.to_csv(os.path.join(main_exp_name, "Total_results.csv"), index=False)


def remove_empty_keys(data):
    keys_to_remove = [key for key, value in data.items() if value == []]
    for key in keys_to_remove:
        del data[key]
    return data


def save_val_test_pairs(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    keys = data.keys()
    val_test_pairs = {}

    for key in keys:
        if key.endswith("_val"):
            base_name = key[:-4]
            test_key = f"{base_name}_test"
            if test_key in data:
                val_test_pairs[base_name] = (data[key], data[test_key])

    for base_name, (val_list, test_list) in val_test_pairs.items():
        if base_name == "":
            base_name = "Standard"
        with open(
            os.path.join(output_dir, f"Scatter_{base_name}.csv"),
            mode="w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow([f"{base_name}_Val", f"{base_name}_Test"])
            for i in range(len(val_list)):
                writer.writerow([val_list[i], test_list[i]])
        if (
            base_name == "Imgaug_average"
            and "Cross_val_Imgaug_average" not in val_test_pairs.keys()
        ):
            with open(
                os.path.join(output_dir, f"Scatter_Cross_val_Imgaug_average.csv"),
                mode="w",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [f"Cross_val_Imgaug_average_Val", f"Cross_val_Imgaug_average_Test"]
                )
                for i in range(len(val_list)):
                    writer.writerow([val_list[i], test_list[i]])
    print(f"scatter data saved in directory: {output_dir}")


def main():
    args = parse_args()
    if args.cv_exp_names is not None:
        cross_val(args)
    agregate(args)


if __name__ == "__main__":
    main()
