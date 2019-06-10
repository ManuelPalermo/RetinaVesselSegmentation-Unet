import os
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from time import perf_counter
import imageio
from sklearn import metrics
from skimage import morphology as skmorphology
from sklearn.utils import class_weight
import re
from DataGenerator import DataGenerator
from UnetModel import Unet



def normalize_img(img):
	return ((img - img.min()) * (255 / (img - img.min()).max())).astype(np.uint8)


def determine_padding_needed(img_shape, patch_shape):
	px, py = 0, 0
	for px in range(patch_shape[0]):
		if (img_shape[0] + px) % patch_shape[0] == 0:
			break
	for py in range(patch_shape[1]):
		if (img_shape[1] + py) % patch_shape[1] == 0:
			break
	return px, py


def deconstruct_images(image_ids, drive_path, save_path="./PatchDataset/",
                       patch_size=(64, 64), image_size=(700, 605),
                       overlap_ratio=2, save_patches=False,
                       load_target=True):
	os.makedirs(save_path + "images/", exist_ok=True)
	os.makedirs(save_path + "gt/", exist_ok=True)

	px, py = determine_padding_needed(image_size, patch_size)

	pside = (patch_size[0] // 2,
	         patch_size[1] // 2)

	patches = (np.arange(pside[0], image_size[0] + px - pside[0]+ 1,
	                     patch_size[0] // overlap_ratio),
	           np.arange(pside[1], image_size[1] + py - pside[1] + 1,
	                     patch_size[1] // overlap_ratio))

	patch_imgs, patch_segs = [], []
	n_patches = 0
	for img_id in image_ids:
		print("Extracting patches from -> ", img_id)
		img = plt.imread(drive_path + "images/" + img_id + ".ppm")[:, :, 1]
		img = np.pad(img, ((0, py), (0, px)), mode='constant')

		if load_target: seg = plt.imread(drive_path + "gt/" + img_id + ".ah.ppm")
		else: seg = np.zeros_like(img)
		seg = np.pad(seg, ((0, py), (0, px)), mode='constant')


		for y, ypatch in enumerate(patches[1]):
			for x, xpatch in enumerate(patches[0]):
				n_patches += 1
				patch_img = img[ypatch - pside[0]:ypatch + pside[0],
				                xpatch - pside[1]:xpatch + pside[1]]
				patch_seg = seg[ypatch - pside[0]:ypatch + pside[0],
				                xpatch - pside[1]:xpatch + pside[1]]

				patch_imgs.append(patch_img)
				patch_segs.append(patch_seg)
				if save_patches:
					imageio.imwrite(f"{save_path}images/{img_id}_p{x:02}{y:02}.tif", normalize_img(patch_img))
					imageio.imwrite(f"{save_path}gt/{img_id}_p{x:02}{y:02}.tif", patch_seg)

	print("Total number of patches extracted:", n_patches)
	return patch_imgs, patch_segs


def group_patches(patch_paths):
	images = {}
	pattern = re.compile("im[0-9]{4}.*.tif")
	for p in patch_paths:
		img_id = re.search(pattern, p).group()[:-10]
		if img_id not in images:
			images[img_id] = [p]
		else:
			images[img_id].append(p)
	return images


def reconstruct_image(patch_paths, patches):
	# need to undo overlap and border values
	shape_substring = sorted(patch_paths)[-1][-8:-4]
	img_dims = (int(shape_substring[0:2]),
	            int(shape_substring[2:4]))
	patch_matrix = [[0 for _ in range(img_dims[0] + 1)] for _ in range(img_dims[1] + 1)]
	for patch_id, patch in zip(patch_paths, patches):
		pos_substring = patch_id[-8:-4]
		pos = (int(pos_substring[0:2]),
		       int(pos_substring[2:4]))
		patch_matrix[pos[1]][pos[0]] = patch
	img = np.block(patch_matrix)

	px, py = determine_padding_needed(IMAGE_SIZE, patch.shape)
	img = img[:-py, :-px]
	return img


def split_data_fundus(drive_path, data_split=(0.65, 0.85, 1.0), shuffle=True, num=np.inf):
	original_ids = [t.replace(".ah.ppm", "") for t in os.listdir(drive_path + "gt/")
	                if ('.ah.ppm' in t)]

	if shuffle:
		np.random.seed(42)
		np.random.shuffle(original_ids)
	original_ids = original_ids[:min(num, 9999999)]

	split = (int(data_split[0] * len(original_ids)),
	         int(data_split[1] * len(original_ids)),
	         int(data_split[2] * len(original_ids)))

	train_ids = original_ids[0:split[0]]
	val_ids = original_ids[split[0]:split[1]]
	test_ids = original_ids[split[1]:split[2]]
	extra_ids = ["im0291", "im0319", "im0324"]
	return train_ids, val_ids, test_ids, extra_ids


def split_data_drive(drive_path, data_split=(0.8, 1.0), shuffle=True, num=np.inf):
	original_ids = sorted([t.replace(".ah.ppm", "") for t in os.listdir(drive_path + "gt/")
	                       if ('.ah.ppm' in t)], key=str.lower)

	train_val_ids = original_ids[:20]
	test_ids = original_ids[20:]

	if shuffle: np.random.shuffle(train_val_ids)
	split = (int(data_split[0] * len(train_val_ids)),
	         int(data_split[1] * len(train_val_ids)))

	train_ids = train_val_ids[0:split[0]]
	val_ids = train_val_ids[split[0]:split[1]]
	return train_ids, val_ids, test_ids


def load_patch_data(dataset_path, image_ids, load_targets=True):
	patches = [dataset_path + "images/" + p
	           for img_id in image_ids
	           for p in os.listdir(dataset_path + "images/")
	           if img_id in p]

	if load_targets:
		targets = [dataset_path + "gt/" + p
		           for img_id in image_ids
		           for p in os.listdir(dataset_path + "gt/")
		           if img_id in p]
		return patches, targets
	return patches, []


def calc_metrics(maskpred, masktrue, mask=None):
	if mask is not None:
		maskpred = np.extract(mask, maskpred)
		masktrue = np.extract(mask, masktrue)
	((TP, FP), (FN, TN)) = metrics.confusion_matrix(masktrue, maskpred)
	return {'Accuracy:   ': round((TP + TN) / (TP + FP + FN + TN), 4),
	        'Sensitivity:': round(TP / (TP + FN), 4),
	        'Specitivity:': round(TN / (TN + FP), 4),
	        'Precision:  ': round(TP / (TP + FP), 4),
	        }


def segment_images(data, drive_path, save_path, classifier,
                   patch_dimensions, present_metrics=True,
                   show_segmentation=False, binarize=True):

	os.makedirs(save_path, exist_ok=True)

	# metricas gerais da segemntacao da diretoria
	metricas_totais = {'Accuracy:   ': 0,
	                   'Sensitivity:': 0,
	                   'Specitivity:': 0,
	                   'Precision:  ': 0,
	                   }

	start_overal_time = perf_counter()
	for i, (img_id, img_patches_path) in enumerate(group_patches(data).items()):
		start_segment_time = perf_counter()

		img_generator = DataGenerator(images_paths=img_patches_path,
		                              target_paths=None,
		                              image_dimensions=(*patch_dimensions, 1),
		                              batch_size=1,  # len(img_patches),
		                              shuffle=False,
		                              augment=False,
		                              )

		segment_patches = classifier.predict_generator(generator=img_generator,
		                                               steps=len(img_generator)
		                                               )

		segment_patches = np.squeeze(segment_patches)

		segment_vasos = normalize_img(reconstruct_image(img_patches_path,
		                                                segment_patches))

		mask = np.where((plt.imread(drive_path + "masks/" + img_id + "-msf.png") > 0), 255, 0)
		mask = skmorphology.remove_small_holes(mask.astype(bool),
		                                       area_threshold=5
		                                       ).astype(np.uint8)
		segment_vasos[mask == 0] = 0

		if binarize or present_metrics:
			threshold = 127
			segment_vasos = np.where(segment_vasos > threshold, 255, 0).astype(np.uint8)

		if present_metrics:
			target = np.where((plt.imread(drive_path + "gt/" + img_id + ".ah.ppm") > 0), 255,
			                  0).astype(np.uint8)

			metricas = calc_metrics(segment_vasos, target, mask)

			# acumulacão das metrica de cada imagem para calculo geral posterior
			for m, score in metricas.items():
				metricas_totais[m] += score

			print(f"Image '{img_id}' segmentation results:")
			for m, value in metricas.items():
				print(f"{m}         {value}")
			print(f"Processing time:     {round(perf_counter() - start_segment_time, 5)}s")

			if show_segmentation:
				'''
				idx = np.random.choice(list(range(len(segment_patches))),
				                       size=min(len(segment_patches), 2), replace=False)
				for pi in idx:
					p = segment_patches[pi]
					# print(p.shape, p.min(), p.max(), p.dtype)
					plt.imshow(p, cmap="gray")
					plt.show()
				'''
				fig, ax = plt.subplots(1, 2, figsize=(20, 10))
				ax[0].imshow(segment_vasos, cmap="gray")
				ax[0].set_title("Segmentação UNet", fontsize=15)
				ax[1].imshow(target, cmap="gray")
				ax[1].set_title("Target", fontsize=15)
				plt.show()

		new_path = save_path + img_id + "_unet.tif"
		imageio.imsave(new_path, segment_vasos)

	if present_metrics:
		# apresenta a medias das metricas associadas a segmentacao das imagens na diretoria
		print("\n\n ----- Validation overall metrics: ----- ")
		for m, score in metricas_totais.items():
			print(m, round(score / (i + 1), 5))
		print(f"Total time: {round(perf_counter() - start_overal_time, 5)}s")


def analyse_history(history):
	fig, ax = plt.subplots(3, 1, figsize=(6, 6))
	ax[0].plot(history.history['loss'], label="TrainLoss")
	ax[0].plot(history.history['val_loss'], label="ValLoss")
	ax[0].legend(loc='best', shadow=True)

	ax[1].plot(history.history['dice_coef'], label="TrainDiceCoef")
	ax[1].plot(history.history['val_dice_coef'], label="ValDiceCoef")
	ax[1].legend(loc='best', shadow=True)

	ax[2].plot(history.history['acc'], label="TrainAcc")
	ax[2].plot(history.history['val_acc'], label="ValAcc")
	ax[2].legend(loc='best', shadow=True)
	plt.show()


def delete_old_patches(directory):
	print("Removing old files from: ", directory)
	try:
		images = [os.path.join(directory + "images/", f) for f in
		          os.listdir(directory + "images/")]
		targets = [os.path.join(directory + "gt/", f)
		           for f in os.listdir(directory + "gt/")]
		for f in images + targets:
			os.remove(f)
	except:
		pass





if __name__ == "__main__":
	dataset = "drive"
	LOAD_MODEL = True
	PATCH_DIMENSIONS = (160, 160)
	EPOCHS = 16

	if dataset == "drive":
		drive_path = './DRIVE_like_FUNDUS/'
		IMAGE_SIZE = (565, 584)
		train_ids, val_ids, test_ids = split_data_drive(drive_path=drive_path)
	else:
		drive_path = './fundus_original_backup/'  # './fundus/'
		IMAGE_SIZE = (700, 605)
		train_ids, val_ids, test_ids, extra_ids = split_data_fundus(drive_path=drive_path)
	save_path = f'./SaveVasosUNet_{dataset}/'
	dataset_path = f'./PatchDataset/'


	print("Loading data")
	delete_old_patches(dataset_path)

	if not LOAD_MODEL:
		print("Training Model")
		# create train patches
		deconstruct_images(train_ids + val_ids,
		                   drive_path=drive_path,
		                   save_path=dataset_path,
		                   patch_size=PATCH_DIMENSIONS,
		                   image_size=IMAGE_SIZE,
		                   overlap_ratio=3,
		                   save_patches=True)

		(Xtrain_paths, Ytrain_paths) = load_patch_data(dataset_path, train_ids)
		(Xval_paths, Yval_paths) = load_patch_data(dataset_path, val_ids)

		train_data = DataGenerator(images_paths=Xtrain_paths,
		                           target_paths=Ytrain_paths,
		                           image_dimensions=(*PATCH_DIMENSIONS, 1),
		                           batch_size=8,
		                           shuffle=True,
		                           augment=True)

		val_data = DataGenerator(images_paths=Xval_paths,
		                         target_paths=Yval_paths,
		                         image_dimensions=(*PATCH_DIMENSIONS, 1),
		                         batch_size=16)

		learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
		                                            patience=2,
		                                            verbose=1,
		                                            factor=0.2,
		                                            min_lr=0.000001)

		checkpoint = ModelCheckpoint(f'new_trained_model_{dataset}.hdf5',
		                             monitor='val_loss',
		                             mode='min',
		                             save_best_only=True,
		                             save_weights_only=True,
		                             verbose=1)

		early_stop = EarlyStopping(monitor="val_loss",
		                           mode="min",
		                           patience=10)

		model = Unet(pretrained_weights=None,
		             input_size=(*PATCH_DIMENSIONS, 1)
		             ).get_model()

		model.summary()

		#class_weights = class_weight.compute_class_weight('balanced',
		#                                                  np.unique(train_data.target_paths),
		#                                                  train_data.target_paths)

		history = model.fit_generator(generator=train_data,
		                              validation_data=val_data,
		                              epochs=EPOCHS,
		                              steps_per_epoch=len(train_data),
		                              #class_weight=class_weights,
		                              callbacks=[learning_rate_reduction, checkpoint, early_stop],
		                              verbose=2)
		analyse_history(history)


		print("Segmenting test images")
		deconstruct_images(test_ids,
		                   patch_size=PATCH_DIMENSIONS,
		                   drive_path=drive_path,
		                   save_path=dataset_path,
		                   image_size=IMAGE_SIZE,
		                   overlap_ratio=1,
		                   save_patches=True)

		(Xtest_paths, Ytest_paths) = load_patch_data(dataset_path, test_ids)

		segment_images(data=Xtest_paths,
		               drive_path=drive_path,
		               save_path=save_path,
		               classifier=model,
		               patch_dimensions=PATCH_DIMENSIONS,
		               present_metrics=True,
		               show_segmentation=True)


	else:
		print("Loading pre-trained model")
		model = Unet(pretrained_weights = f'./attention_model_{dataset}.hdf5',
		             input_size=(*PATCH_DIMENSIONS, 1)
		             ).get_model()
		model.summary()


		if dataset == "drive":
			print("Segmenting test images")
			deconstruct_images(test_ids,
			                   patch_size=PATCH_DIMENSIONS,
			                   drive_path=drive_path,
			                   save_path=dataset_path,
			                   image_size=IMAGE_SIZE,
			                   overlap_ratio=1,
			                   save_patches=True)

			(Xtest_paths, Ytest_paths) = load_patch_data(dataset_path, test_ids)

			segment_images(data=Xtest_paths,
			               drive_path=drive_path,
			               save_path=save_path,
			               classifier=model,
			               patch_dimensions=PATCH_DIMENSIONS,
			               present_metrics=True,
			               show_segmentation=True)


		if dataset == "fundus":
			print("Segmenting all images")

			deconstruct_images(train_ids + val_ids + test_ids,
			                   patch_size=PATCH_DIMENSIONS,
			                   drive_path=drive_path,
			                   save_path=dataset_path,
			                   image_size=IMAGE_SIZE,
			                   overlap_ratio=1,
			                   save_patches=True)

			(Xtest_paths, _) = load_patch_data(dataset_path,
			                                   train_ids + val_ids + test_ids,
			                                   load_targets=False)

			segment_images(data=Xtest_paths,
			               drive_path=drive_path,
			               save_path=save_path,
			               classifier=model,
			               patch_dimensions=PATCH_DIMENSIONS,
			               present_metrics=True,
			               show_segmentation=True)


			print("\nSegmenting extra images")
			deconstruct_images(extra_ids,
			                   patch_size=PATCH_DIMENSIONS,
			                   drive_path=drive_path,
			                   save_path=dataset_path,
			                   image_size=IMAGE_SIZE,
			                   overlap_ratio=1,
			                   save_patches=True,
			                   load_target=False)

			(Xextra_paths, _) = load_patch_data(dataset_path,
			                                   extra_ids,
			                                   load_targets=False)

			segment_images(data=Xextra_paths,
			               drive_path=drive_path,
			               save_path=save_path,
			               classifier=model,
			               patch_dimensions=PATCH_DIMENSIONS,
			               present_metrics=False,
			               show_segmentation=False,
			               binarize=False)