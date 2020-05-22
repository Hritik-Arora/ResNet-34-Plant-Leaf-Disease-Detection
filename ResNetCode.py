from fastai import *
from fastai.vision import *
from fastai.metrics import accuracy
import os
from os import listdir
%reload_ext autoreload
%autoreload 2
%matplotlib inline

path = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/"
os.listdir(path)


path = Path(path); path

#Loading the dataset
directory_root = '../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/'
image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)
​
    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)
​
        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)
​
            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(image_directory)
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")

#Transforming images
tfms = get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=0.)

#A sample image file
file_path = '../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Pepper,_bell___healthy/01dd93b0-0e34-447b-87ea-ccc9f2b62d03___JR_HL 8005.JPG'

dir_name = os.path.dirname(file_path)

#Defining a function for obtaining labels from file names
def get_labels(file_path): 
    dir_name = os.path.dirname(file_path)
    split_dir_name = dir_name.split("/")
    dir_length = len(split_dir_name)
    label  = split_dir_name[dir_length - 1]
    return(label)

#Generating the data and normalising it
data = ImageDataBunch.from_name_func(path, image_list, label_func=get_labels,  size=224, bs=64,num_workers=2,ds_tfms=tfms)

data.normalize(imagenet_stats)

#Training the model
learn = cnn_learner(data, models.resnet34, metrics=[error_rate,accuracy], model_dir='/tmp/models/')

#Making the model learn for 10 epochs
learn.fit_one_cycle(10)

learn.recorder.plot_losses()

#Looking at the predictions
interpretation = ClassificationInterpretation.from_learner(learn)
losses, indices = interpretation.top_losses()
interpretation.plot_top_losses(4, figsize=(15,11))

#Plotting the confusion matrix
interpretation.plot_confusion_matrix(figsize=(20,20), dpi=60)

#Seeing the most confused classes
interpretation.most_confused(min_val=2)

#Fine tuning the model
learn.save('classification-1')
learn.unfreeze()
learn.fit_one_cycle(1)

#Finding an optimal learning rate
learn.lr_find()

learn.recorder.plot()

#Keeping learning rate equal to otimum value obtained from above
learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-3))
learn.recorder.plot_losses()

#Looking at the predictions again
interpretation = ClassificationInterpretation.from_learner(learn)
losses, indices = interpretation.top_losses()
interpretation.plot_top_losses(4, figsize=(15,11))

learn.save('resnet34-classifier.pkl')

#Plotting confusion matrix again
interpretation.plot_confusion_matrix(figsize=(20,20), dpi=60)

#Calculating Accuracy of the model
predictions,labels = learn.get_preds(ds_type=DatasetType.Valid)
​
predictions = predictions.numpy()
labels = labels.numpy()
​
predicted_labels = np.argmax(predictions, axis = 1)
print((predicted_labels == labels ).sum().item()/ len(predicted_labels))


#Viewing the layers of the ResNet-34 model
learn.summary()

