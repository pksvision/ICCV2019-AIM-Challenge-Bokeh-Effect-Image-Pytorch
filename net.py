import numpy as np
import paramiko
import os 
import cv2

s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect("172.16.112.151",22,username="prasen",password='Krrishna@15it60r23',timeout=4)
sftp = s.open_sftp()

def get_next_item(image_name):
    with sftp.open(image_name) as f:
        img = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)
        return img

def list_net_dir(dir_name):
    with sftp.open(dir_name) as f:
        return sftp.listdir(dir_name)