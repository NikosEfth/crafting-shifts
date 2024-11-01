import torch
from torch import nn, optim
import models as my_models
from utils import *
import os
import time


def inference(model, device, loader):
    model.eval()
    set_size = len(loader.dataset)
    acc_list = None

    with torch.no_grad():
        for input_image, y, path in loader:
            y = y.to(device)
            input_image = [x.to(device) for x in input_image]
            outputs = model(input_image)
            if acc_list is None:
                acc_list = [0] * len(outputs)
            for idx, output in enumerate(outputs):
                preds = output.argmax(dim=1)
                acc_list[idx] += (preds == y).sum().item()
    acc_list = [round(acc / set_size, 4) for acc in acc_list]
    return acc_list


def training_function(
    args, loader_info, lr, method_loss, save_path, experiment_dir, device
):
    model_name_path = os.path.join(
        ".", save_path, f"Method_loss_{method_loss}_lr_{lr}.pt"
    )

    model = my_models.PseudoCombiner(
        no_classes=len(loader_info["classes"]),
        pretrained=args.pretrained,
        backbone_name=args.backbone,
    )
    model.to(device)
    model.train()
    # save a text file with the network structure and with a flag of whether each part was trainable or not
    save_architecture(network=model, direct=experiment_dir)
    if args.search_mode.lower() == "new_test" and os.path.isfile(
        model_name_path
    ):
        load_model(network=model, model_location=model_name_path)
        return model
    elif os.path.isfile(model_name_path):
        print(f"{model_name_path} exists.")
        return

    criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    gamma = pow(
        args.bt_exp_scheduler_gamma, (1.0 / float(args.epochs))
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

    # Training of the model.
    epochs = args.epochs
    for epoch in range(epochs):
        start = time.time()
        train_acc, train_count = 0, 0

        for idx, (input_image, y, path) in enumerate(loader_info["train_loader"]):
            optimizer.zero_grad()
            y = y.to(device)
            input_image = [x.to(device) for x in input_image]
            outputs = model(input_image)
            loss = torch.zeros(1)[0].to(device)
            batches = [x.shape[0] for x in outputs]

            for idx2, output in enumerate(outputs):
                current_loss = criterion(output, y)
                if idx2 != 0:
                    loss += current_loss * method_loss
                else:
                    loss += current_loss

            loss = loss / sum(batches)
            _, preds = torch.max(nn.functional.softmax(outputs[0], dim=1).data, 1)
            train_acc += torch.sum(preds == y.data).item()
            train_count += len(y)
            loss.backward()
            if nan_in_grad(model=model):
                save_model(
                    network=model,
                    optimizer=optimizer,
                    epoch=args.epochs,
                    direct=model_name_path,
                    is_nan=True,
                )
                return model
            optimizer.step()
        scheduler.step()
        print(
            f"Epoch: {epoch} "
            f"LR: {round(scheduler.get_lr()[0], 8)} "
            f"Acc: {round(train_acc / train_count, 4)} "
            f"Time: {report_time(time.time() - start)}"
        )

    save_model(
        network=model,
        optimizer=optimizer,
        epoch=args.epochs,
        direct=model_name_path,
    )
    return model


def testing_function(
    model, loader_info, test_idx_list, lr, method_loss, csv_file_name, domains, device
):
    if model is not None:
        val_only_acc, test_acc, val_only_acc, imgaug_average = ([] for _ in range(4))
        with torch.no_grad():
            print("Validating Normal")
            val_list = inference(
                model=model, device=device, loader=loader_info["val_loader"]
            )
            if loader_info["val_only_loader"] is not None:
                for idx, vo_loader in enumerate(loader_info["val_only_loader"]):
                    print(
                        f"Validating {loader_info['pd_names_val_only'][idx].replace('_', ' ')}"
                    )
                    val_only_acc.append(
                        inference(model=model, device=device, loader=vo_loader)
                    )
                imgaug_average = [
                    round(sum(col) / len(col), 4) for col in zip(*val_only_acc)
                ]
                val_only_acc = [item for sublist in val_only_acc for item in sublist]
            for idx2, test_idx in enumerate(test_idx_list):
                test_acc.append(
                    inference(
                        model=model,
                        device=device,
                        loader=loader_info["test_loader_list"][idx2],
                    )
                )
                for idx, output in enumerate(loader_info["output_names_val"]):
                    print(
                        f"Test {domains[test_idx]} domain, {output} Acc: {test_acc[-1][idx]}"
                    )

            total_test = [round(sum(col) / len(col), 4) for col in zip(*test_acc)]
            test_acc_list_out = [item for sublist in test_acc for item in sublist]
        add_free_log(
            data=[[str(lr)]]
            + [[str(method_loss)]]
            + [[str(x)] for x in val_list]
            + [[str(x)] for x in imgaug_average]
            + [[str(x)] for x in total_test]
            + [[str(x)] for x in val_only_acc]
            + [[str(x)] for x in test_acc_list_out],
            save_dir=csv_file_name,
        )
        for idx, output in enumerate(loader_info["output_names_val"]):
            print(f"Total Test {output} Acc: {total_test[idx]}")
    return


def search_hyperparameters(
    args,
    loader_info,
    test_idx_list,
    csv_file_name,
    domains,
    save_path,
    experiment_dir,
    device,
):
    lr_min, lr_max = args.lr_search_range
    ml_min, ml_max = args.ml_search_range
    if lr_max < lr_min:
        lr_max, lr_min = lr_min, lr_max
    if ml_max < ml_min:
        ml_max, ml_min = ml_min, ml_max
    if args.lr is None and args.method_loss is None:
        # Define the range for learning rate search and method loss search
        lr_search_range = generate_points(
            range_tuple=(lr_min, lr_max),
            points=args.lr_search_no,
            log_scale=True,
        )
        ml_search_range = generate_points(
            range_tuple=(ml_min, ml_max), points=args.ml_search_no
        )
        print("Searching for the best Learning Rate and the best Method Loss...")
        print(
            f"We will train a total of {len(lr_search_range)*len(ml_search_range)} models"
        )
        for lr in lr_search_range:
            for method_loss in ml_search_range:
                print(f"Trying Learning Rate: {lr} and Method Loss: {method_loss}")
                model = training_function(
                    args,
                    loader_info,
                    lr,
                    method_loss,
                    save_path,
                    experiment_dir,
                    device,
                )
                testing_function(
                    model,
                    loader_info,
                    test_idx_list,
                    lr,
                    method_loss,
                    csv_file_name,
                    domains,
                    device,
                )
    elif args.lr is None:
        # Define the range for learning rate search
        lr_search_range = generate_points(
            range_tuple=(lr_min, lr_max),
            points=args.lr_search_no,
            log_scale=True,
        )
        print("Searching for the best Learning Rate...")
        print(f"We will train a total of {len(lr_search_range)} models")
        for lr in lr_search_range:
            print(f"Trying Learning Rate: {lr}")
            model = training_function(
                args,
                loader_info,
                lr,
                args.method_loss,
                save_path,
                experiment_dir,
                device,
            )
            testing_function(
                model,
                loader_info,
                test_idx_list,
                lr,
                args.method_loss,
                csv_file_name,
                domains,
                device,
            )
    elif args.method_loss is None:
        # Define the range for method loss search
        ml_search_range = generate_points(
            range_tuple=(ml_min, ml_max), points=args.ml_search_no
        )
        print("Searching for the best Method Loss...")
        print(f"We will train a total of {len(ml_search_range)} models")
        for method_loss in ml_search_range:
            print(f"Trying Method Loss: {method_loss}")
            model = training_function(
                args,
                loader_info,
                args.lr,
                method_loss,
                save_path,
                experiment_dir,
                device,
            )
            testing_function(
                model,
                loader_info,
                test_idx_list,
                args.lr,
                method_loss,
                csv_file_name,
                domains,
                device,
            )
