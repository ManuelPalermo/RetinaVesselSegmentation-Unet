import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import imgaug as ia
import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, images_paths, target_paths,
                 image_dimensions=(64, 64, 1), batch_size=64,
                 shuffle=False, augment=False):
        self.images_paths = images_paths
        self.target_paths = target_paths
        self.dim = image_dimensions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        images = np.array([plt.imread(self.images_paths[k]) for k in indexes], dtype=np.uint8)

        if self.target_paths is None:
            targets = np.array([])
        else:
            targets = np.array([plt.imread(self.target_paths[k]) for k in indexes],
                               dtype=np.uint8) / 255

        if self.augment == True:
            images, targets = self.augmentor(images, targets)

        # ---- debug input image / label pairs ----
        # for (img, lbl) in zip(images, targets):
        #	print(img.shape, img.min(), img.max(), img.dtype)
        #	print(lbl.shape, lbl.min(), lbl.max(), img.dtype)
        #	fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        #	ax[0].imshow(img, cmap='gray')
        #	ax[1].imshow(lbl, cmap='gray')
        #	plt.show()
        #	input()

        images = images.astype(np.float32) / 255.
        # images = (images - images.mean()) / images.std()

        return np.reshape(images, (*images.shape, self.dim[-1])), \
               np.reshape(targets, (*targets.shape, 1))

    def augmentor(self, images, targets):
        '''Augments each batch of data with random transformations'''
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([
                iaa.Fliplr(0.5, name="Fliplr"),
                iaa.Flipud(0.5, name="Flipud"),
                sometimes(iaa.SomeOf((0, 2), [
                        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                   rotate=(-25, 25), name="Affine"),
                        iaa.ElasticTransformation(alpha=(0.01, 0.1), sigma=0.15,
                                                  name="ElasticTransformation"),
                        iaa.PiecewiseAffine(scale=(0.001, 0.03), name="PiecewiseAffine"),
                        iaa.PerspectiveTransform(scale=(0.01, 0.05), name="PerspectiveTransform"),
                ], random_order=True)),

                sometimes(iaa.OneOf([
                        iaa.GaussianBlur(sigma=(0, 0.2)),
                        iaa.AverageBlur(k=3),
                        iaa.MedianBlur(k=3),
                ])),

                sometimes(iaa.OneOf([
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255)),
                        iaa.AddElementwise((-5, 5)),
                ])),

                sometimes(iaa.OneOf([
                        iaa.GammaContrast(gamma=(0.75, 1.50)),
                        iaa.HistogramEqualization(),
                        iaa.Multiply((0.80, 1.15)),
                        iaa.Add((-20, 15)),
                        iaa.Sharpen(alpha=(0, 0.5), lightness=(0.7, 1.5)),
                        iaa.Emboss(alpha=(0, 0.5), strength=(0.7, 1.5)),
                ])),
        ], random_order=True)

        seq_det = seq.to_deterministic()
        images = seq_det.augment_images(images)
        targets = seq_det.augment_segmentation_maps(
                [ia.SegmentationMapOnImage(t.astype(bool), shape=t.shape)
                 for t in targets])
        targets = np.array([t.get_arr_int() for t in targets])

        return images, targets
