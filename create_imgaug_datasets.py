import os
import random
import argparse
import numpy as np
from PIL import Image
import time
from imgaug import augmenters as iaa


def parse_args():
    parser = argparse.ArgumentParser(description="Augment the dataset")
    parser.add_argument("--dataset", type=str, help="Dataset")
    parser.add_argument(
        "--replace", action="store_true", help="replace existing images"
    )
    return parser.parse_args()


def resize_min_dimension(image, min_dimension):
    height, width = image.size
    aspect_ratio = float(width) / float(height)

    if height < width:
        new_height = min_dimension
        new_width = int(min_dimension * aspect_ratio)
    else:
        new_width = min_dimension
        new_height = int(min_dimension / aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    return resized_image


def is_corrupted(img):
    try:
        im = Image.open(img).convert("RGB")
        im.verify()
        im.close()
        return False
    except (IOError, OSError, Image.DecompressionBombError):
        print(f"Corrupted image: {img}")
        print("Recalculating")
        return True


def apply_random_augmentation(image, aug_family):
    if aug_family == "arithmetic":
        augmentations = [
            random.choice([iaa.Add((-200, -50)), iaa.Add((50, 200))]),
            random.choice(
                [iaa.AddElementwise((-200, -50)), iaa.AddElementwise((50, 200))]
            ),
            iaa.AdditiveGaussianNoise(loc=0, scale=(50, 150), per_channel=0.5),
            iaa.AdditiveLaplaceNoise(loc=0, scale=(50, 150), per_channel=0.5),
            iaa.AdditivePoissonNoise(lam=(40, 60), per_channel=0.5),
            iaa.Multiply((2, 4), per_channel=0.5),
            iaa.MultiplyElementwise((2, 4), per_channel=0.5),
            iaa.Cutout(nb_iterations=(5, 10), size=0.2, squared=True, cval=0),
            iaa.Dropout((0.2, 0.5), per_channel=0.5),
            iaa.CoarseDropout((0.2, 0.5), size_percent=(0.02, 0.05)),
            iaa.Dropout2d(p=0.5),
            iaa.ImpulseNoise(0.5),
            iaa.SaltAndPepper(0.5),
            iaa.CoarseSaltAndPepper(0.5, size_percent=(0.01, 0.03)),
            iaa.Salt(0.5),
            iaa.CoarseSalt(0.5, size_percent=(0.01, 0.1)),
            iaa.Pepper(0.5),
            iaa.CoarsePepper(0.5, size_percent=(0.01, 0.1)),
            iaa.Invert(0.5, per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.JpegCompression(compression=(90, 99)),
        ]
    elif aug_family == "geometric":
        augmentations = [
            iaa.Affine(
                scale={
                    "x": random.choice([(0.5, 0.75), (1.25, 1.75)]),
                    "y": random.choice([(0.5, 0.75), (1.25, 1.75)]),
                }
            ),
            iaa.Affine(translate_percent={"x": (-0.4, 0.4), "y": (-0.4, 0.4)}),
            iaa.Affine(rotate=random.choice([(-45, -15), (15, 45)])),
            iaa.Affine(shear=random.choice([(-25, -15), (15, 25)])),
            iaa.PiecewiseAffine(scale=(0.05, 0.1)),
            iaa.PerspectiveTransform(scale=(0.1, 0.2)),
            iaa.ElasticTransformation(alpha=(2.0, 5.0), sigma=0.25),
            iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.1, 0.1))),
            iaa.Jigsaw(nb_rows=10, nb_cols=10),
        ]
    elif aug_family == "artistic":
        augmentations = [
            iaa.Cartoon(
                blur_ksize=9, segmentation_size=1.0, saturation=2.0, edge_prevalence=1.0
            )
        ]
    elif aug_family == "blur":
        augmentations = [
            iaa.GaussianBlur(sigma=(5.0, 7.0)),
            iaa.AverageBlur(k=(9, 15)),
            iaa.MedianBlur(k=(9, 15)),
            iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
            iaa.MotionBlur(k=15),
            iaa.MeanShiftBlur(),
        ]
    elif aug_family == "color":
        augmentations = [
            iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.WithChannels(0, iaa.Add((50, 179))),
            ),
            random.choice(
                [
                    iaa.WithBrightnessChannels(iaa.Add((-200, -50))),
                    iaa.WithBrightnessChannels(iaa.Add((50, 200))),
                ]
            ),
            random.choice(
                [
                    iaa.MultiplyAndAddToBrightness(mul=(0.1, 0.3), add=(-200, -50)),
                    iaa.MultiplyAndAddToBrightness(mul=(0.1, 0.3), add=(50, 200)),
                    iaa.MultiplyAndAddToBrightness(mul=(1.8, 2.0), add=(-200, -50)),
                    iaa.MultiplyAndAddToBrightness(mul=(1.8, 2.0), add=(50, 200)),
                ]
            ),
            random.choice(
                [iaa.MultiplyBrightness((0.1, 0.3)), iaa.MultiplyBrightness((1.8, 2.0))]
            ),
            random.choice(
                [iaa.AddToBrightness((-200, -50)), iaa.AddToBrightness((50, 200))]
            ),
            iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add((50, 179)))),
            random.choice(
                [
                    iaa.MultiplyHueAndSaturation((0.1, 0.3), per_channel=True),
                    iaa.MultiplyHueAndSaturation((1.8, 2.0), per_channel=True),
                ]
            ),
            random.choice([iaa.MultiplyHue((0.1, 0.3)), iaa.MultiplyHue((1.8, 2.0))]),
            random.choice(
                [iaa.MultiplySaturation((0.1, 0.3)), iaa.MultiplySaturation((1.8, 2.0))]
            ),
            iaa.RemoveSaturation(),
            random.choice(
                [
                    iaa.AddToHueAndSaturation((-200, -50), per_channel=True),
                    iaa.AddToHueAndSaturation((50, 200), per_channel=True),
                ]
            ),
            random.choice([iaa.AddToHue((-200, -50)), iaa.AddToHue((50, 200))]),
            random.choice(
                [iaa.AddToSaturation((-200, -50)), iaa.AddToSaturation((50, 200))]
            ),
            iaa.Grayscale(alpha=(0.8, 1.0)),
            iaa.ChangeColorTemperature((8000, 10000)),
            iaa.KMeansColorQuantization(n_colors=4),
            iaa.UniformColorQuantization(n_colors=4),
            iaa.UniformColorQuantizationToNBits(nb_bits=(2, 4)),
        ]
    elif aug_family == "contrast":
        augmentations = [
            random.choice(
                [
                    iaa.GammaContrast((0.1, 0.3), per_channel=True),
                    iaa.GammaContrast((1.8, 2.0), per_channel=True),
                ]
            ),
            iaa.SigmoidContrast(gain=(10, 25), cutoff=(0.4, 0.6), per_channel=True),
            random.choice(
                [
                    iaa.LogContrast(gain=(0.1, 0.3), per_channel=True),
                    iaa.LogContrast(gain=(1.8, 2.0), per_channel=True),
                ]
            ),
            random.choice(
                [
                    iaa.LinearContrast((0.1, 0.3), per_channel=True),
                    iaa.LinearContrast((1.8, 2.0), per_channel=True),
                ]
            ),
            iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True),
            iaa.CLAHE(tile_grid_size_px=(10, 21)),
        ]
    elif aug_family == "convolutional":
        augmentations = [
            iaa.Convolve(matrix=np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])),
            iaa.Sharpen(alpha=(0.8, 1.0), lightness=(0.75, 2.0)),
            iaa.Emboss(alpha=(0.8, 1.0), strength=(0.5, 1.5)),
            iaa.EdgeDetect(alpha=(0.8, 1.0)),
            iaa.DirectedEdgeDetect(alpha=(0.8, 1.0), direction=(0.0, 1.0)),
        ]
    elif aug_family == "edges":
        augmentations = [
            iaa.Sequential(
                [iaa.GaussianBlur(sigma=(2.0, 2.0)), iaa.Canny(alpha=(0.8, 1.0))]
            )
        ]
    elif aug_family == "segmentation":
        augmentations = [
            iaa.Superpixels(p_replace=(0.5, 1.0), n_segments=(16, 128)),
            iaa.Voronoi(iaa.RegularGridPointsSampler(n_cols=20, n_rows=40)),
            iaa.UniformVoronoi((100, 500)),
            iaa.RegularGridVoronoi(10, 20),
        ]
    elif aug_family == "weather":
        augmentations = [
            iaa.FastSnowyLandscape(
                lightness_threshold=(140, 255), lightness_multiplier=(2.5, 4.0)
            ),
            iaa.Clouds(),
            iaa.Fog(),
            iaa.Snowflakes(flake_size=(0.3, 0.8), speed=(0.001, 0.05)),
            iaa.Rain(),
        ]

    chosen_augmentation = random.choice(augmentations)
    original_size = image.size
    resized_image = resize_min_dimension(image, 227)
    resized_image = np.array(resized_image)
    if resized_image.shape[2] == 3:
        resized_image = resized_image[..., ::-1]
    augmented_image = chosen_augmentation(image=resized_image)
    augmented_image = augmented_image[..., ::-1]
    augmented_image = Image.fromarray(augmented_image)
    final_image = augmented_image.resize(original_size, Image.Resampling.BICUBIC)

    return final_image


def process_image(input_path, output_path, aug_family, replace):
    if os.path.exists(output_path) and not replace:
        if not is_corrupted(output_path):
            return

    image = Image.open(input_path).convert("RGB")
    augmented_image = apply_random_augmentation(image=image, aug_family=aug_family)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    augmented_image.save(output_path)
    return


def process_images(input_directory, output_directory, aug_family, total_images, args):
    processed_images = 0
    for root, _, files in sorted(os.walk(input_directory)):
        for file in sorted(files):
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
            ):
                input_path = os.path.join(root, file)
                output_path = os.path.join(
                    output_directory, os.path.relpath(root, input_directory), file
                )
                process_image(
                    input_path=input_path,
                    output_path=output_path,
                    aug_family=aug_family,
                    replace=args.replace,
                )
                processed_images += 1
                print(f"Processed {processed_images}/{total_images} images", end="\r")


def main():
    args = parse_args()

    aug_family_list = [
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
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    input_directory = os.path.join(
        ".", "data", args.dataset, f"{args.dataset}_Original"
    )

    total_images = 0
    print("Calculating the total image number")
    for root, dirs, files in os.walk(input_directory):
        total_images += sum(
            1
            for file in files
            if any(file.lower().endswith(ext) for ext in image_extensions)
        )
    print(f"Total images: {total_images}")

    for aug_family in aug_family_list:
        print(f"Creating {aug_family} dataset.")
        output_directory = os.path.join(
            ".",
            "data",
            args.dataset,
            f"{args.dataset}_Imgaug_{aug_family}",
        )
        start = time.time()
        process_images(
            input_directory=input_directory,
            output_directory=output_directory,
            aug_family=aug_family,
            total_images=total_images,
            args=args,
        )
        print(
            f"{aug_family} augmentations completed in {round(time.time() - start)} seconds."
        )


if __name__ == "__main__":
    main()