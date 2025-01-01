# %% image_acquisition.py
import os
import cv2

#%%
def load_images_from_folder(folder_path, resize_dim=(800, 600)):
    images = []
    labels = []
    loaded_count = 0
    failed_count = 0

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            if len(img.shape) == 2 or img.shape[2] != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
            resized_img = cv2.resize(img, resize_dim) 
            images.append(resized_img)
            print(f"Loaded image: {filename}")
            loaded_count +=1
            labels.append(1 if "Skin" in folder_path else 0)
        else:
            print(f"Failed to load image: {filename}")
            failed_count +=1
    print(f"Total images loaded: {loaded_count}")
    print(f"Total images failed to load: {failed_count}")
    
    return images, labels

