from torchvision.datasets import ImageFolder
from utils import float_tuple, dirlist
from torch.utils.data import DataLoader
import os
import copy
import augmentations as aug
import torchvision as tv
import PIL
import random


def set_dataloaders(
    args, run, dataset_path, train_domain_idx, test_domain_idx, domains
):

    test_domain_idx = (
        [test_domain_idx] if isinstance(test_domain_idx, int) else test_domain_idx
    )

    geom_train, pd_dir_train, transform_train, _ = make_transform_lists(
        run=run, dataset_path=dataset_path, mode="train"
    )
    geom_val, pd_dir_val, transform_val, pd_names_val = make_transform_lists(
        run=run, dataset_path=dataset_path, mode="val"
    )
    geom_val_only, pd_dir_val_only, transform_val_only, pd_names_val_only = (
        make_transform_lists(run=run, dataset_path=dataset_path, mode="val_only")
    )
    geom_test, pd_dir_test, transform_test, _ = make_transform_lists(
        run=run, dataset_path=dataset_path, mode="test"
    )

    train_dataset_list, train_list_class = dirlist(
        os.path.join(dataset_path, f"{domains[train_domain_idx]}_train.csv")
    )
    val_dataset_list, val_list_class = dirlist(
        os.path.join(dataset_path, f"{domains[train_domain_idx]}_val.csv")
    )

    classes = list(set(train_list_class))
    train_set = ImageFolderPseudoDomains(
        is_for_train=True,
        roots=pd_dir_train,
        image_list=train_dataset_list,
        class_list=train_list_class,
        transforms=transform_train,
        geo_transforms=geom_train,
    )
    val_set = ImageFolderPseudoDomains(
        roots=pd_dir_val,
        image_list=val_dataset_list,
        class_list=val_list_class,
        transforms=transform_val,
        geo_transforms=geom_val,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
    )
    val_only_loader = None
    if transform_val_only is not None:
        val_only_loader = []
        for idx, dir_val_only in enumerate(pd_dir_val_only):
            val_only_set = ImageFolderPseudoDomains(
                roots=dir_val_only,
                image_list=val_dataset_list,
                class_list=val_list_class,
                transforms=transform_val_only[idx],
                geo_transforms=geom_val_only,
            )
            val_only_loader.append(
                DataLoader(
                    val_only_set,
                    batch_size=args.batch,
                    shuffle=False,
                    num_workers=args.workers,
                )
            )

    test_loader_list = []
    for test_domain in test_domain_idx:
        test_dataset_list, test_list_class = dirlist(
            os.path.join(dataset_path, f"{domains[test_domain]}_test.csv")
        )
        test_set = ImageFolderPseudoDomains(
            roots=pd_dir_test,
            image_list=test_dataset_list,
            class_list=test_list_class,
            transforms=transform_test,
            geo_transforms=geom_test,
        )
        test_loader_list.append(
            DataLoader(
                test_set,
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.workers,
            )
        )

    output_names_val = [f"output_{name}" for name in pd_names_val]
    if len(pd_names_val) > 1:
        output_names_val.extend(["output_25-75", "output_50-50", "output_75-25"])

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "val_only_loader": val_only_loader,
        "test_loader_list": test_loader_list,
        "pd_names_val": pd_names_val,
        "pd_names_val_only": pd_names_val_only,
        "classes": classes,
        "output_names_val": output_names_val,
    }


def make_transform_lists(run, dataset_path, mode):
    mode = mode.lower()
    if mode not in ["train", "val", "val_only", "test"]:
        raise ValueError("Mode must be one of train, val, val_only, and test")

    if mode in ["train", "val", "test"]:
        pseudo_domain_read = "pseudo_domains"
    elif mode == "val_only":
        pseudo_domain_read = "val_only_pseudo_domains"
    transform_read = f"{mode}_transforms"

    geom, pd_dir, pd_tf, pd_names = None, None, None, []
    if pseudo_domain_read in run:
        pd_tf = []
        pd_names = []
        pd_dir = [
            []
            for _ in range(
                len([x for x in run[pseudo_domain_read] if transform_read in x])
            )
        ]
        geom = create_transform(input_list=run["geometric_transforms"][mode])
        for idx, pd in enumerate(run[pseudo_domain_read]):
            if "pseudo_number" not in pd:
                if transform_read in pd:
                    for direct in pd["dir"]:
                        pd_dir[idx].append(
                            os.path.join(
                                dataset_path, f"{run['io_var']['dataset']}_{direct}"
                            )
                        )
                    transform = create_transform(input_list=pd[transform_read])
                    for idx2, tr in enumerate(transform):
                        if isinstance(tr, dict):
                            transform[idx2] = tr["augmentation"]
                    pd_tf.append(tv.transforms.Compose(transform))
                    pd_names.append(pd["name"])
            else:
                pd_tf = [[] for _ in range(len(pd["name"]))]
                pd_names = []
                pd_dir = [[] for _ in range(len(pd["name"]))]
                for idx2, pseudo in enumerate(pd["name"]):
                    pd_names.append(pd["name"][idx2])
                    for idx3, number in enumerate(range(pd["pseudo_number"])):
                        current_transform_read = f"{transform_read}_{number + 1}"
                        current_dir = f"dir_{number + 1}"
                        transform = create_transform(
                            input_list=pd[current_transform_read]
                        )
                        pd_tf[idx2].append(tv.transforms.Compose(transform))
                        pd_dir[idx2].append(
                            [
                                os.path.join(
                                    dataset_path,
                                    f"{run['io_var']['dataset']}_{pd[current_dir][idx2]}",
                                )
                            ]
                        )

    return geom, pd_dir, pd_tf, pd_names


def create_transform(input_list, nms_model=None):
    transform_list = []
    for tr_module in input_list:

        # geometric augmentations are dicts to be imported to the aligned_transforms function
        if tr_module["name"] == "resize":
            transform_list.append(
                {
                    "name": "resize",
                    "augmentation": tv.transforms.Resize(size=tr_module["size"]),
                    "param": None,
                }
            )
        elif tr_module["name"] == "center_crop":
            transform_list.append(
                {
                    "name": "center_crop",
                    "augmentation": tv.transforms.CenterCrop(size=tr_module["size"]),
                    "param": None,
                }
            )
        elif tr_module["name"] == "random_horizontal_flip":
            transform_list.append(
                {
                    "name": "random_horizontal_flip",
                    "augmentation": tv.transforms.RandomHorizontalFlip(p=1),
                    "param": tr_module["p"],
                }
            )
        elif tr_module["name"] == "random_resized_crop":
            transform_list.append(
                {
                    "name": "random_resized_crop",
                    "augmentation": tv.transforms.RandomResizedCrop(
                        size=tr_module["size"],
                        scale=float_tuple(tr_module["scale"]),
                        ratio=float_tuple(tr_module["ratio"]),
                    ),
                    "param": (
                        tr_module["size"],
                        float_tuple(tr_module["scale"]),
                        float_tuple(tr_module["ratio"]),
                    ),
                }
            )
        elif tr_module["name"] == "canny":
            transform_list.append(
                aug.Canny(
                    sigma=tr_module["sigma"],
                    thresh_rand=tr_module["thresh_rand"],
                    thresh_mode=tr_module["thresh_mode"],
                    hyst_par=tr_module["hyst_par"],
                    hyst_pert=tr_module["hyst_pert"],
                )
            )
        elif tr_module["name"] == "invert":
            transform_list.append(aug.Invert(prob=tr_module["prob"]))
        elif tr_module["name"] == "normalize":
            transform_list.append(
                aug.ListTransform(
                    tv.transforms.Normalize(
                        mean=tr_module["mean"], std=tr_module["std"]
                    )
                )
            )
        elif tr_module["name"] == "to_tensor":
            transform_list.append(aug.ListTransform(aug.ToTensor()))

    return transform_list


def aligned_transforms(transformations, image_list):
    img = image_list[0]

    for tf in transformations:
        if tf["name"] in ["resize", "center_crop"]:
            for idx in range(len(image_list)):
                image_list[idx] = tf["augmentation"](image_list[idx])

        elif tf["name"] == "random_horizontal_flip":
            if random.random() < tf["param"]:
                for idx in range(len(image_list)):
                    image_list[idx] = tf["augmentation"](image_list[idx])

        elif tf["name"] == "random_resized_crop":
            i, j, h, w = tv.transforms.RandomResizedCrop.get_params(
                img, scale=tf["param"][1], ratio=tf["param"][2]
            )
            for idx in range(len(image_list)):
                image_list[idx] = tv.transforms.functional.resized_crop(
                    image_list[idx], i, j, h, w, tf["param"][0]
                )

    return image_list


class ImageFolderPseudoDomains(tv.datasets.ImageFolder):
    def __init__(
        self,
        roots,
        image_list,
        class_list,
        is_for_train=False,
        transforms=None,
        geo_transforms=None,
    ):
        super().__init__(root=roots[0][0])
        self.is_for_train = is_for_train
        self.roots = roots
        self.transforms = transforms if transforms else [None] * len(roots)
        self.geo_transforms = geo_transforms
        self.img_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.classes = sorted(set(class_list))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = [
            (img, self.class_to_idx[class_list[i]]) for i, img in enumerate(image_list)
        ]

    def __getitem__(self, index):
        path, target = self.samples[index]
        current_roots = self._get_current_roots()

        images = self._load_images(roots=current_roots, path=path)
        images = aligned_transforms(
            transformations=self.geo_transforms, image_list=images
        )  # Apply aligned geometric transformations
        transformed_images = [
            self.transforms[i](img_list) for i, img_list in enumerate(images)
        ]
        return transformed_images, target, path

    def _get_current_roots(self):
        current_roots = []
        if len(self.roots) > 1:
            assert (
                self.roots[0] == self.roots[1]
            ), "The augmented datasets must be the same for all pseudo domains"
        # Within a given pseudo domain (eg. original, canny), check how many augmented versions of the dataset there are (eg. Original, Imgaug_convolutional, Imgaug_edges, etc)
        total_source = len(self.roots[0])

        # if training, choose one augmented version of the dataset randomly
        source_no = random.randint(0, total_source - 1) if self.is_for_train else 0
        for root in self.roots:
            # for every pseudo domain (eg. canny), choose the same augmented version of the dataset (eg. Imgaug_convolutional)
            chosen_root = root[source_no] if self.is_for_train else root[0]
            current_roots.append(chosen_root)
        return current_roots

    def _load_images(self, roots, path):
        images = []
        for root in roots:
            domain_images = []
            image_path = os.path.join(root, path)

            # Try to load the image
            image = None
            if os.path.isfile(image_path):
                image = self.pil_loader(path=image_path)
            else:
                # considering potential file extensions if not found
                base_path, _ = os.path.splitext(image_path)
                for ext in self.img_ext:
                    potential_path = base_path + ext
                    if os.path.isfile(potential_path):
                        image = self.pil_loader(path=potential_path)
                        break
            # If image could not be found or loaded, log a warning
            if image is None:
                print(f"Warning: Image not found at {image_path}")
            images.append(image)
        return images

    def pil_loader(self, path):
        with open(path, "rb") as file:
            img = PIL.Image.open(file)
            return img.convert("RGB")

    def __len__(self):
        return len(self.samples)
