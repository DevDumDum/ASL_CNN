# move file into target directory
import os 
import shutil 
import pathlib
import numpy as np

labels = []
testing_labels = []
training_labels = []
cwd = os.getcwd()
path = cwd+'\\archive2' # yung root folder ng dataset mo
target_path = f'{cwd}\\archive' #new directory 
train_dir = f'{target_path}\\train'
test_dir = f'{target_path}\\test'
img_path = ''
train_split = 0.8
test_split = 0.2
xdict = {}
train_len = 0
test_len = 0

# get all files and put in dictionary
for dirs in os.listdir(path):
        curpath = os.path.join(path, dirs)
        total_files = len(os.listdir(curpath))
        train_len = int(np.round(total_files * train_split))
        test_len = int(np.round(total_files * test_split))
        count = 0
        xdict.update({dirs: 
                {'path': [],
                'folder_path':curpath, 
                'train_len':train_len, 
                'test_len':test_len }
        })
        
        for i in os.listdir(curpath):
                image_path = os.path.join(curpath, i)
                xdict[dirs]['path'].append(image_path)


# iterate all images then move in target directory
for keys in xdict.keys():
        current_key = keys
        count = 0
        # create train and test directory
        train_path = f'{train_dir}\\{current_key}'
        test_path = f'{test_dir}\\{current_key}'
        pathlib.Path(train_path).mkdir(exist_ok=True, parents=True)
        pathlib.Path(test_path).mkdir(exist_ok=True, parents=True)
        for path in xdict[current_key]['path']:
                # creat
                # move file into target directory (train)
                if count > xdict[current_key]['train_len']:
                        # move file into test
                        shutil.move(path, test_path)
                else:
                        shutil.move(path, train_path)
                count += 1