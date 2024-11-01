import random
import numpy as np
import torch
import skimage.filters
import skimage.feature
import scipy.ndimage as ndi


def max_one(image):
    image = np.array(image)
    if np.amax(image) > 1:
        image = image / 255
    return image


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def thresh_function(thresh_mode):
    if isinstance(thresh_mode, list):
        thresh_mode = [x.lower() for x in thresh_mode]
        if "all" in thresh_mode or "normal" in thresh_mode:
            thresh_mode = "all"
        else:
            thresh_mode = thresh_mode[random.randint(0, len(thresh_mode) - 1)]
    if isinstance(thresh_mode, str):
        try:
            thresh_mode = int(thresh_mode)
        except Exception:
            thresh_mode = thresh_mode.lower()
    if thresh_mode in ["normal", "all", "n", "a"]:
        thresh_mode_function = random.choice(
            [
                skimage.filters.threshold_otsu,
                skimage.filters.threshold_yen,
                skimage.filters.threshold_mean,
                skimage.filters.threshold_isodata,
                skimage.filters.threshold_li,
            ]
        )
    elif thresh_mode in ["yen", "y"]:
        thresh_mode_function = skimage.filters.threshold_yen
    elif thresh_mode in ["mean", "m"]:
        thresh_mode_function = skimage.filters.threshold_mean
    elif thresh_mode in ["isodata", "i", "id"]:
        thresh_mode_function = skimage.filters.threshold_isodata
    elif thresh_mode in ["li", "l"]:
        thresh_mode_function = skimage.filters.threshold_li
    else:
        thresh_mode_function = skimage.filters.threshold_otsu
        if thresh_mode not in ["otsu", "o"]:
            print(f"{thresh_mode} not a valid thresh mode, Otsu used by default.")
    return thresh_mode_function


class ListTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images):
        if isinstance(images, list):
            return [self.transform(image) for image in images]
        return self.transform(images)


class ToTensor(object):
    def __call__(self, image):
        image = np.array(image)
        image = max_one(image)
        if len(image.shape) > 2:
            image = torch.from_numpy(image).type("torch.FloatTensor").permute(2, 0, 1)
        else:
            image = torch.from_numpy(image).type("torch.FloatTensor")
        return image


class Invert(object):
    def __init__(self, prob=50):
        self.prob = prob

    def __call__(self, image):
        image = max_one(image)
        if random.randint(1, 100) <= self.prob:
            image = 1.0 - image
        return image


class Canny:
    def __init__(
        self,
        sigma=1.0,
        thresh_rand=10,
        thresh_mode="normal",
        hyst_par=(0.5, 1.5),
        hyst_pert=0.2,
    ):

        self.sigma = sigma
        self.thresh_mode = thresh_mode
        self.thresh_rand = thresh_rand
        self.hyst_par = hyst_par
        self.hyst_pert = hyst_pert

    def __call__(self, image):
        if isinstance(image, list):
            image = image[0]
        image = np.array(image)
        if len(image.shape) > 2:
            image = rgb2gray(image)
        image = max_one(image)

        if isinstance(self.sigma, (list, tuple)):
            self.sigma = random.choice(list(self.sigma))

        gaussian_kwargs = dict(
            sigma=self.sigma, mode="constant", cval=0.0, preserve_range=False
        )
        image_copy = skimage.filters.gaussian(image, **gaussian_kwargs)
        jsobel = ndi.sobel(image_copy, axis=1)
        isobel = ndi.sobel(image_copy, axis=0)
        image_copy = np.hypot(isobel, jsobel)

        try:
            exact_thresh = thresh_function(self.thresh_mode)(image_copy)
        except Exception:
            exact_thresh = 0.5
        thresh = exact_thresh + random.normalvariate(0, self.thresh_rand / 2)
        if thresh >= np.amax(image_copy) or thresh <= np.amin(image_copy):
            thresh = exact_thresh

        per = random.normalvariate(0, self.hyst_pert)
        lower, upper = [
            max(0.1, self.hyst_par[0] - per),
            min(2, self.hyst_par[1] + per),
        ]
        if lower > upper:
            lower, upper = upper, lower
        try:
            edge = skimage.feature.canny(
                image,
                sigma=self.sigma,
                low_threshold=lower * thresh,
                high_threshold=upper * thresh,
            )
        except:
            edge = skimage.feature.canny(
                image,
                sigma=self.sigma,
                low_threshold=lower * 0.5,
                high_threshold=upper * 0.5,
            )

        if np.amax(edge) == np.amin(edge):
            edge = skimage.feature.canny(
                image, sigma=self.sigma, low_threshold=None, high_threshold=None
            )

        return np.dstack((edge, edge, edge))
