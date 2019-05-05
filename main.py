import cv2
import numpy as np
import os.path as osp


data_dir = 'us'

def main():
    """
    Steps: 
    1. preprocess image
    2. find possible chars in the scene
    3. find possible plate that contain matching chars
    4. find chars in the plate
    5. recognize the chars in the plate
    6. compare the results and get the results and confidence value
    """
    orig_image = cv2.imread(osp.join(data_dir, '0b86cecf-67d1-4fc0-87c9-b36b0ee228bb.jpg'))
    



if __name__ == '__main__':
    main()