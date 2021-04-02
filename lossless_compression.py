import cv2
import os
import numpy as np
final_output_dir = './Epoch_74_no_attention/'
input_dir = './EP74_Syn_no_attention/'

total_files = os.listdir(input_dir)

if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)
print ("started")
for m in total_files:
    img = cv2.imread(input_dir+str(m))
    cv2.imwrite(final_output_dir+str(m), np.uint8(img), [cv2.IMWRITE_PNG_COMPRESSION, 9])