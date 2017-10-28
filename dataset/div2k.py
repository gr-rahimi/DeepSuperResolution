import glob, os
from PIL import Image
import numpy as np

def read_dataset(root_path, hr_train_folder, hr_val_folder,lr_train_folder,lr_val_folder, patch_per_image = 5, patch_size= 96):
    train_path = os.path.join(root_path, hr_train_folder, "*.png")
    hr_train_files = glob.glob(train_path)
    hr_train_files.sort()
    hr_train_files = hr_train_files[0:5]

    val_path = os.path.join(root_path, hr_val_folder, "*.png")
    hr_val_files = glob.glob(val_path)
    hr_val_files.sort()
    hr_val_files = hr_val_files[0:5]


    hr_train = [np.array(Image.open(f)) for f in hr_train_files]


    hr_val = [np.array(Image.open(f)) for f in hr_val_files]

    train_path = os.path.join(root_path, lr_train_folder, "*.png")
    lr_train_files = glob.glob(train_path)
    lr_train_files.sort()
    lr_train_files = lr_train_files[0:5]

    val_path = os.path.join(root_path, lr_val_folder, "*.png")
    lr_val_files = glob.glob(val_path)
    lr_val_files.sort()
    lr_val_files = lr_val_files[0:5]

    lr_train = [np.array(Image.open(f)) for f in lr_train_files]

    lr_val = [np.array(Image.open(f)) for f in lr_val_files]


    orig_train_image_count = len(hr_train_files)
    orig_val_image_count = len(hr_val_files)

    lr_train_patches = np.zeros((orig_train_image_count*patch_per_image,patch_size,patch_size,3))
    lr_val_patches = np.zeros((orig_val_image_count * patch_per_image, patch_size, patch_size, 3))
    hr_train_patches = np.zeros((orig_train_image_count*patch_per_image,patch_size*4,patch_size*4,3))
    hr_val_patches = np.zeros((orig_val_image_count * patch_per_image, patch_size*4, patch_size*4, 3))

    for i in range(orig_train_image_count):
        print lr_train[i].shape
        (im_h, im_w, _) = lr_train[i].shape
        for j in range(patch_per_image):
            start_h=np.random.randint(0,im_h - patch_size)
            start_w = np.random.randint(0, im_w - patch_size)
            lr_train_patches[i * patch_per_image + j, :, :, :] = lr_train[i] [start_h:start_h + patch_size,
                                                                 start_w:start_w + patch_size, :]
            #_save_patch(lr_train_patches[i * patch_per_image + j, :, :, :],"lr"+str(i * patch_per_image + j)+".png")
            hr_train_patches[i * patch_per_image + j, :, :, :] = hr_train[i][start_h * 4:(start_h + patch_size) * 4,
                                                                 start_w * 4 : (start_w + patch_size) * 4, :]
            #_save_patch(hr_train_patches[i * patch_per_image + j, :, :, :],
            #            "hr" + str(i * patch_per_image + j) + ".png")
    for i in range(orig_val_image_count):
        (im_h, im_w, _) = lr_val[i].shape
        for j in range(patch_per_image):
            start_h=np.random.randint(0,im_h - patch_size)
            start_w = np.random.randint(0, im_w - patch_size)
            lr_val_patches[i * patch_per_image + j, :, :, :] = lr_val[i][start_h:start_h + patch_size,
                                                                 start_w:start_w + patch_size, :]
            hr_val_patches[i * patch_per_image + j, :, :, :] = hr_val[i][start_h * 4:(start_h + patch_size) * 4,
                                                                 start_w *4: (start_w + patch_size) * 4, :]


    return (hr_train_patches- 128)/128, (hr_val_patches-128)/128, lr_train_patches/256, lr_val_patches/256


def _save_patch(img,name):
    pil_img=Image.fromarray(img.astype('uint8'))
    pil_img.save(name)
