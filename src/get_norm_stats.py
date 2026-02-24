# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np
import dataloader

# แก้ไข 1: เปลี่ยน target_length เป็น 100 สำหรับเสียง 1 วินาที
# set skip_norm as True only when you are computing the normalization stats
# ปรับตัวเลขให้เหมาะกับเสียง 1 วินาที (100 เฟรม)
audio_conf = {
    'num_mel_bins': 128, 
    'target_length': 100,  # ความยาวเสียงไอของคุณ
    'freqm': 48,           # เปลี่ยนจาก 24 เป็น 48 (ปิดแกนความถี่ 48 bins)
    'timem': 20,           # ต้องลดลงจาก 192 เหลือ 20 (ประมาณ 20% ของ 100 เฟรม)
    'mixup': 0.5, 
    'skip_norm': True,     # คงไว้เป็น True เพื่อหาค่าสถิติ
    'mode': 'train', 
    'dataset': 'audioset'
}
# แก้ไข 2: เปลี่ยน Path เป็นไฟล์ของคุณ และลด batch_size/num_workers
train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset('train_data.json', label_csv='class_labels_indices.csv',
                                audio_conf=audio_conf), batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

mean=[]
std=[]
for i, (audio_input, labels) in enumerate(train_loader):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))