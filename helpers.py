
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
#PATH PROCESS
import os
from pathlib import Path
from time import sleep
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset



from PIL import Image


# the one library to handle nii images
import nibabel as nib



def load_labels(Problem_Type, Image_Type) -> dict:
    '''Helper function to return a dictionary of the labels instead of writing them by hand
    :param Problem_Type: "CINE" or "LGE", mostly we will use CINE.
    :param Image_Type: "4CH" or "2CH" or "SAX"
    
    '''
    if Problem_Type == "CINE":
        LBLS_2CH = {0: "[0] none", 1 : "[1] Left Ventricle Cavity", 2: "[2] Left Ventricle Myocardium"}
        LBLS_4CH = {0: "[0] none", 1 : "[1] Left Ventricle Cavity", 2 : "[2] Left Ventricle Myocardium", 3 : "[3] Right Ventricle Cavity", 4 : "[4] Right Atrium", 5 : "[5] Left Atrium"}
        LBLS_SAX = {0: "[0] none", 1 : "[1] Left Ventricle Myocardium", 2 : "[2] Left Ventricle Cavity", 3 : "[3] Right Ventricle Cavity"}
        LBLS_RAS = None

    elif Problem_Type == "LGE":
        LBLS_2CH = {0: "[0] none", 1 : "[1] Left Ventricle Cavity", 2: "[2] Left Ventricle Myocardium", 3 : "[3] Myocardial Scar (Scar)"}
        LBLS_4CH = {0: "[0] none", 1 : "[1] Left Ventricle Cavity", 2 : "[2] Left Ventricle Myocardium", 3 : "[3] Myocardial Scar (Scar)", 4 : "[4] Right Ventricle Cavity"}
        LBLS_SAX = {0: "[0] none", 1 : "[1] Left Ventricle Cavity", 2 : "[2] Left Ventricle Myocardium", 3 : "[3] Myocardial Scar (Scar)", 4 : "[4] Right Ventricle Cavity"}
        LBLS_RAS = {0: "[0] none", 1 : "[1] Right Atrium"}

    label_dict = {"2CH" : LBLS_2CH, "4CH": LBLS_4CH, "SAX" : LBLS_SAX, "RAS": LBLS_RAS}

    return label_dict[Image_Type] #We'll use this dictionary as a lookup for description of each label.




def visualize_nii_file(Image_Type, data_path, label_path = None, Problem_Type="CINE", Cmap="grey", image_saved_path=None, video_saved_path=None, frame_delay=50):
    '''
    This function visualizes a nii.gz file format, regardless it being a static image or a video.
    Example usage: visualize_nii_file("4CH", Data_Path, Label_Path)
    
    :param Image_Type: Type of CINE MRI "4CH", "2CH" or "SAX"
    :param data_path: Path to the .nii.gz file containing the image data
    :param label_path: Path to the .nii.gz file containing the mask
    :param Problem_Type: Problem Type "CINE" or "LGE" but it is most probably CINE
    :param Cmap: Colormap for matplotlib, default is grey, other options "hot", "jet" and None
    :param image_saved_path: Path to save resultant visual image
    :param video_saved_path: Path to save resultant video
    :param frame_delay: Delay between video frames in milliseconds
    '''


    #if a label file path is NOT supplied, alter the logic of the entire function to visualize the data file only
    LBLS = True if label_path else False
    
    current_labels = load_labels(Problem_Type, Image_Type) if LBLS else None

    


    ni_data = nib.load(data_path)
    if(LBLS): ni_label = nib.load(label_path)
    else:
        label_path = data_path
        ni_label = nib.load(label_path)

    Pxls_Data = ni_data.get_fdata()
    Pxls_Label = ni_label.get_fdata()


    if(not image_saved_path):
        image_saved_path = f"{Path(data_path).name}.png"
    
    if(Path(image_saved_path).suffix != ".png"):
        print("saved image file path must end with .png")
        raise ValueError

    if(not video_saved_path):
        video_saved_path = f"{Path(data_path).name}.gif"

    if(Path(video_saved_path).suffix != ".gif"):
        print("saved video file path must end with .gif")
        raise ValueError


    try:
        assert(Pxls_Data.shape == Pxls_Label.shape)
    except:
        print("ERROR: Data image has different shape than Label image, possibly you have supplied mismatched files")
        print(f"Data shape: {Pxls_Data.shape()} \t Pixel shape: {Pxls_Label.shape()}")
        raise AssertionError




    print("Creating Image...")

    figure, axis = plt.subplots(1, 2 ,figsize=(10,10))

    axis[0].imshow(Pxls_Data[:, :, Pxls_Data.shape[2]//2],cmap=Cmap)
    axis[0].set_xlabel(Pxls_Data.shape)
    axis[0].set_ylabel(Pxls_Data.size)
    axis[0].set_title("IMAGE")

    
    Plot_Color_Op = axis[1].imshow(Pxls_Label[:, :, Pxls_Label.shape[2]//2],cmap="hot")
    axis[1].set_xlabel(Pxls_Label[Pxls_Label.shape[0]//2].shape)
    axis[1].set_ylabel(Pxls_Label[Pxls_Label.shape[0]//2].size)
    axis[1].set_title("MASK" if LBLS else "Cooler Image")

    if LBLS:
        cbar = figure.colorbar(Plot_Color_Op, ax=axis.ravel().tolist(), shrink=0.3, ticks=list(current_labels.keys()), label='Label Legend')
        cbar.ax.set_yticklabels(list(current_labels.values()))




    plt.savefig(f"{image_saved_path}")  # Saves as PNG, ex: CINE_4CH_001.nii.gz.png


    print("Creating GIF video...")

    fig, axis = plt.subplots(1, 2, figsize=(10, 10))

    frames = []
    for i in range(Pxls_Data.shape[2]):
        img = axis[0].imshow(Pxls_Data[:, :, i], animated=True, cmap=Cmap) # Add cmap='hot' or 'jet' or 'grey' for more cool colors
        axis[0].set_title("IMAGE")

        
        mask = axis[1].imshow(Pxls_Label[:, :, i], animated=True, cmap='hot') # Add cmap='hot' or 'jet' or 'grey' for more cool colors
        axis[1].set_title("MASK" if LBLS else "Cooler Image")
        # Store both images as one frame
        frames.append([img, mask])


    if LBLS:
        cbar = fig.colorbar(mask, ax=axis.ravel().tolist(), shrink=0.3, ticks=list(current_labels.keys()), label='Label Legend')
        cbar.ax.set_yticklabels(list(current_labels.values()))


    ani = animation.ArtistAnimation(fig=fig, artists=frames, interval=frame_delay)
    #HTML(ani.to_jshtml()) #Just to make vscode render a video
    FPS = 1000/frame_delay
    ani.save(f"{video_saved_path}", writer='ffmpeg', fps=FPS)

    print("Opening files...")
    sleep(1) #give some time for the file handle to be released
    os.startfile(f"{video_saved_path}")
    sleep(1)
    img = Image.open(f"{image_saved_path}")
    img.show()
    

def recursive_flatten (input_list) -> list:
    '''Flattens a nested list to a single flat list containing all data'''
    if type(input_list) not in [list, np.ndarray]: #base case
        return input_list
    
    prev_list = []
    for element in input_list:
        if type(element) not in [list, np.ndarray]:
            prev_list.append(element)
        else:
            prev_list += recursive_flatten(element)
    
    return prev_list


def get_all_np_images(dataset_path) -> list:
    ''' returns a list containing images in np format '''
    image_nii_files = os.listdir(os.path.abspath(dataset_path))
    # I will just make each filename as an absolute path to avoid python errors
    image_nii_files = [os.path.join(os.path.abspath(dataset_path), i) for i in image_nii_files]

    all_images = []
    for i, image_path in enumerate(image_nii_files):
        np_img = nib.load(image_path).get_fdata()
        all_images.append(np_img)
    
    return all_images



def nii_to_tensor(nii_file_path, dtype=np.int16):
    data = nib.load(nii_file_path).get_fdata().astype(dtype)
    return torch.from_numpy(data)


class MRIDataset(Dataset):
    ''' This class inherits from the Dataset class in pytorch, I only made this class to use DataLoader, nothing else. '''
    def __init__(self, image_tensors, mask_tensors, scaler=None):
        self.images = image_tensors
        self.masks = mask_tensors
        self.scaler = scaler

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.scaler is not None:
            img, _ = clip_outliers_from_img(self.images[idx])
            img_flat = img.reshape(-1, 1)
            img_scaled = self.scaler.transform(img_flat)
            img = img_scaled.reshape(img.shape)
        if(type(img) == torch.Tensor):
            return img.float(), self.masks[idx]
        else: return img.astype(np.float32), self.masks[idx]
    

def Create_Dataset (nii_images_path, nii_masks_path, test_data_percentage=0.3):
    '''
    Dataset creator, it reads all files in the given pathes, converts them to tensors and returns 2 instances of torch.Datasets corresponding to a training set and a test set.
    
    :param nii_images_path: Path to the folder containing all images data
    :param nii_masks_path: Path to the folder containing all masked data
    :param test_data_percentage: Percentage of test set for splitting
    '''
    image_nii_files = sorted(os.listdir(os.path.abspath(nii_images_path)))
    # I will just make each filename as an absolute path to avoid python errors
    image_nii_files = [os.path.join(os.path.abspath(nii_images_path), i) for i in image_nii_files]

    #same with mask images
    mask_nii_files = sorted(os.listdir(os.path.abspath(nii_masks_path)))
    mask_nii_files = [os.path.join(os.path.abspath(nii_masks_path), i) for i in mask_nii_files]

    # For every file in the list of nii files, will convert it to a tensor and append them to a giant list
    list_image_tensors = []
    list_mask_tensors = []

    test_image_tensors = []
    test_mask_tensors = []

    for i, image_path in enumerate(image_nii_files):
        new_tensor = nii_to_tensor(image_path)
        list_image_tensors.append(new_tensor)

    for image_path in mask_nii_files:
        new_tensor = nii_to_tensor(image_path, dtype=np.int64)
        list_mask_tensors.append(new_tensor)

    #sanity check, image tensors and mask tensors should be of equal number
    print(f"All dataset is of length {len(list_image_tensors)}", sep=None)
    print(f"With Masks of{len(list_mask_tensors)}")

    assert len(list_image_tensors) == len(list_mask_tensors)

    print(f"Splitting data to {1-test_data_percentage} for training and {test_data_percentage} for testing...")

    
    split_idx = int(len(list_image_tensors) * test_data_percentage)

    test_image_tensors = list_image_tensors[:split_idx]
    list_image_tensors = list_image_tensors[split_idx:]

    test_mask_tensors = list_mask_tensors[:split_idx]
    list_mask_tensors = list_mask_tensors[split_idx:]

    print(f"Training with {len(list_image_tensors)}:{len(list_mask_tensors)}")
    print(f"Testing with {len(test_image_tensors)}:{len(test_mask_tensors)}")

    return MRIDataset(list_image_tensors, list_mask_tensors),  MRIDataset(test_image_tensors, test_mask_tensors)




def create_2d_dataset(nii_images_path, nii_masks_path, test_data_percentage=0.3, scaler=None):
    '''
    This function creates a dataset of 2D images, each video is dissected into standalone images and returns 2 Dataset objects
    Each dataset has a shape of (C, 1, w, h) where C is the count of the images, w and h are width and height, the 1 is simply a dummy dimension for the CNN to train effectively.
    

    :param nii_images_path: Description
    :param nii_masks_path: Description
    :param test_data_percentage: Description
    '''
    image_nii_files = sorted(os.listdir(os.path.abspath(nii_images_path)))
    # I will just make each filename as an absolute path to avoid python errors
    image_nii_files = [os.path.join(os.path.abspath(nii_images_path), i) for i in image_nii_files]

    #same with mask images
    mask_nii_files = sorted(os.listdir(os.path.abspath(nii_masks_path)))
    mask_nii_files = [os.path.join(os.path.abspath(nii_masks_path), i) for i in mask_nii_files]

    # For every file in the list of nii files, will convert it to a tensor and append them to a giant list
    list_image_tensors = []
    list_mask_tensors = []

    for i, image_path in enumerate(image_nii_files):
        new_tensor = nii_to_tensor(image_path)
        
        for j in range(new_tensor.shape[2]):
            tensor_2d = new_tensor[:, :, j]      # (H, W) the jth slice/frame
            tensor_2d = tensor_2d.unsqueeze(0)    # (1, H, W) added dummy dim
            list_image_tensors.append(tensor_2d)


    for image_path in mask_nii_files:
        new_tensor = nii_to_tensor(image_path)
        for j in range(new_tensor.shape[2]):
            tensor_2d = new_tensor[:, :, j]      # (H, W) the jth slice/frame
            tensor_2d = tensor_2d.unsqueeze(0)    # (1, H, W) added dummy dim
            list_mask_tensors.append(tensor_2d)

    
    #Concat all tensors into 1 VERY BIG TENSOR of a shape (C, 1, w, h) where C is their count, w is the width and h is the height
    image_tensors = torch.stack(list_image_tensors, dim=0)
    mask_tensors = torch.stack(list_mask_tensors, dim=0)

    #sanity check, image tensors and mask tensors should be of equal number
    print(f"All dataset is of length {image_tensors.shape} ", sep=None)
    print(f"With Masks of {mask_tensors.shape}")

    assert len(list_image_tensors) == len(list_mask_tensors)

    print(f"Splitting data to {1-test_data_percentage} for training and {test_data_percentage} for testing...")

    
    split_idx = int(len(image_tensors) * test_data_percentage)

    train_images = image_tensors[split_idx:]
    train_masks  = mask_tensors[split_idx:]

    test_images  = image_tensors[:split_idx]
    test_masks   = mask_tensors[:split_idx]

    print(f"Training with {len(train_images)}:{len(train_masks)}")
    print(f"Testing with {len(test_images)}:{len(test_masks)}")

    return MRIDataset(train_images, train_masks, scaler),  MRIDataset(test_images, test_masks, None)








def clip_outliers_from_img(image:np.ndarray):
    original_shape = image.shape
    p1, p99 = np.percentile(image.reshape(-1, 1), [1, 99])
    pixels = np.clip(image.reshape(-1, 1), p1, p99)
    clipped_img = pixels.reshape(original_shape)
    return clipped_img, pixels



def train_scaler(all_images):
    """Trains a Z-score scaler after clipping values, given all images' data in a list"""

    all_pixels_after_clipping = np.array([]).reshape(-1, 1)
    clipped_images = []
    for img in tqdm(all_images):
        clipped_img, pixels = clip_outliers_from_img(img)
        clipped_images.append(clipped_img)
        all_pixels_after_clipping = np.concatenate((all_pixels_after_clipping, pixels))

    print(f"finished clipping {len(all_images)} images")

    #all_pixels_after_clipping = helpers.recursive_flatten(clipped_images)
    all_pixels_after_clipping = np.array(all_pixels_after_clipping).reshape(-1, 1)

    print(f"finished flattening {len(all_images)} images to {len(all_pixels_after_clipping)} pixels")

    zscaler_clipped = StandardScaler()
    zscaler_clipped = zscaler_clipped.fit(all_pixels_after_clipping)

    print(f"finished training the scaler")

    return zscaler_clipped, clipped_images





# Data_Path = r"C:\Users\ayman.mohamed\Personal\Masters_Security\Deep_Learning\CINE_Cardiac_MRI_With_Deep_Learning\CMR-MULTI\CINE_MULTI\4CH_TR\image\CINE_4CH_001.nii.gz"
# Label_Path = r"C:\Users\ayman.mohamed\Personal\Masters_Security\Deep_Learning\CINE_Cardiac_MRI_With_Deep_Learning\CMR-MULTI\CINE_MULTI\4CH_TR\anno\CINE_4CH_001.nii.gz"
# PROBLEM_TYPE = "CINE"
# IMAGE_TYPE="4CH"
# CMAP = "grey"

# image_saved_path = Path(Data_Path).name
# video_saved_path = Path(Label_Path).name
# frame_delay = 50




# #example usage
# visualize_nii_file("4CH", Data_Path, Label_Path)