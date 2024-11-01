import os
import sys
import argparse
import warnings
from utils import *
from utils_dataset import *
from utils_train_inference import *
from pdb import set_trace as st


def parse_args():
    parser = argparse.ArgumentParser(description="Training and search specifications")

    # IO, seed, data, and network variables
    parser.add_argument("--run", type=str, help="Input training file")
    parser.add_argument("--name", type=str, help="Name of the experiment folder")
    parser.add_argument("--seed", default=0, type=int, help="Choose a seed")
    parser.add_argument("--dataset", type=str, help="Choose the dataset")
    parser.add_argument("--train_only", nargs="+", type=str, help="Domains to train")
    parser.add_argument("--workers", default=6, type=int, help="Number of workers")
    parser.add_argument("--gpu", default=0, type=int, help="Choose the GPU id")
    parser.add_argument("--backbone", default="resnet18", type=str, help="Backbone")
    parser.add_argument(
        "--from_scratch", action="store_false", dest="pretrained", help="No pre-train"
    )

    # Training hyperparameters
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--method_loss", type=float, help="Method loss")
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
    parser.add_argument("--batch", default=64, type=int, help="Batch size")
    parser.add_argument(
        "--bt_exp_scheduler_gamma", default=0.01, type=float, help="Total LR decrease"
    )

    # Search mode
    parser.add_argument(
        "--search_mode",
        choices=["resume", "new_test"],
        default="resume",
        type=str,
        help="What to do when the experiment already exists",
    )
    parser.add_argument(
        "--lr_search_no",
        default=33,
        type=int,
        help="Learning rate number of trainings in the search",
    )
    parser.add_argument(
        "--ml_search_no",
        default=17,
        type=int,
        help="Method loss number of trainings in the search",
    )
    parser.add_argument(
        "--lr_search_range",
        type=float,
        nargs=2,
        default=(1e-5, 1),
        help="Search range for lr tuning",
    )
    parser.add_argument(
        "--ml_search_range",
        type=float,
        nargs=2,
        default=(0, 1),
        help="Search range for method loss tuning",
    )
    return parser.parse_args()


def main():
    # read command line arguments, set device, set seeds, and deactivate warnings
    args = parse_args()
    device = setup_device(gpu_id=args.gpu)
    set_all_seeds(seed=args.seed)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # Load the configuration file, and update it according to the command line arguments
    run = read_yaml(args.run)
    run["exec"] = f"python {' '.join(sys.argv)}"
    if args.dataset is not None:
        run["io_var"]["dataset"] = args.dataset
    if args.name is not None:
        run["io_var"]["run_name"] = args.name
    if "vit" in args.backbone.lower():
        vit_normalization(data=run)

    # Create dataset path, the experiment directory, and save execution details
    dataset_path = os.path.join(".", "data", run["io_var"]["dataset"])
    experiment_dir = os.path.join(
        "Results",
        f"{run['io_var']['dataset']}_{args.backbone}",
        run["io_var"]["run_name"],
    )
    try_make_dir(directory=experiment_dir)
    save_yaml(structure=run, direct=os.path.join(experiment_dir, "run.yaml"))

    # Create domain indices for training and testing
    domains = sorted(
        os.listdir(
            os.path.join(
                dataset_path,
                f"{run['io_var']['dataset']}_{run['pseudo_domains'][0]['dir'][0]}",
            )
        )
    )
    train_domains_only = (
        args.train_only if args.train_only else domains
    )
    domain_idx = list(range(len(domains)))
    current_dom_idx = [x for x in domain_idx]
    rest_dom_idx = [
        [x for x in domain_idx if x != current_idx] for current_idx in domain_idx
    ]

    # Main training loop over each domain
    for idx in list(range(len(domains))):
        train_idx, test_idx_list = current_dom_idx[idx], rest_dom_idx[idx]
        print(
            f"Training: {domains[train_idx].lower()} "
            f"Testing: {', '.join(domains[x].lower() for x in test_idx_list)}"
        )

        # Check if the current domain should be trained
        if domains[train_idx].lower() not in [x.lower() for x in train_domains_only]:
            print(
                f"{domains[train_idx].lower()} not in the specified train domains. Skipping..."
            )
            continue

        # Setting the dataloaders and data loading info
        loader_info = set_dataloaders(
            args=args,
            run=run,
            dataset_path=dataset_path,
            train_domain_idx=train_idx,
            test_domain_idx=test_idx_list,
            domains=domains,
        )

        # Creating the directory of the current seed and initialize the reporting csv if it does not exist
        save_path = os.path.join(
            experiment_dir, domains[train_idx], f"Seed_{args.seed}"
        )
        try_make_dir(directory=save_path)
        csv_file_name = os.path.join(
            experiment_dir,
            f"Results_source_{domains[train_idx].lower()}_seed_{args.seed}.csv",
        )
        initialize_csv_file(
            loader_info=loader_info,
            csv_file_name=csv_file_name,
            test_idx_list=test_idx_list,
            domains=domains,
        )

        # Train and test the model based on the specified parameters
        if args.lr is not None and args.method_loss is not None:
            model = training_function(
                args=args,
                loader_info=loader_info,
                lr=args.lr,
                method_loss=args.method_loss,
                save_path=save_path,
                experiment_dir=experiment_dir,
                device=device,
            )
            testing_function(
                model=model,
                loader_info=loader_info,
                test_idx_list=test_idx_list,
                lr=args.lr,
                method_loss=args.method_loss,
                csv_file_name=csv_file_name,
                domains=domains,
                device=device,
            )
        else:
            search_hyperparameters(
                args=args,
                loader_info=loader_info,
                test_idx_list=test_idx_list,
                csv_file_name=csv_file_name,
                domains=domains,
                save_path=save_path,
                experiment_dir=experiment_dir,
                device=device,
            )


if __name__ == "__main__":
    main()
