
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
#PATH PROCESS
import os
from pathlib import Path
from time import sleep


#from IPython.display import HTML #This is just so we can visualize gifs inside vscode



from PIL import Image


# the one library to handle nii images
import nibabel as nib



def load_labels(Problem_Type, Image_Type):
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
        image_saved_path = f"{Path(data_Path).name}.png"
    
    if(Path(image_saved_path).suffix != ".png"):
        print("saved image file path must end with .png")
        raise ValueError

    if(not video_saved_path):
        video_saved_path = f"{Path(data_Path).name}.gif"

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
    




Data_Path = r"C:\Users\ayman.mohamed\Personal\Masters_Security\Deep_Learning\CINE_Cardiac_MRI_With_Deep_Learning\CMR-MULTI\CINE_MULTI\4CH_TR\image\CINE_4CH_001.nii.gz"
Label_Path = r"C:\Users\ayman.mohamed\Personal\Masters_Security\Deep_Learning\CINE_Cardiac_MRI_With_Deep_Learning\CMR-MULTI\CINE_MULTI\4CH_TR\anno\CINE_4CH_001.nii.gz"
PROBLEM_TYPE = "CINE"
IMAGE_TYPE="4CH"
CMAP = "grey"

image_saved_path = Path(Data_Path).name
video_saved_path = Path(Label_Path).name
frame_delay = 50




#example usage
visualize_nii_file("4CH", Data_Path, Label_Path)