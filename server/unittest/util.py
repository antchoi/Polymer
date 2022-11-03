import pickle
import time

import numpy as np
import torch
from torch.autograd import Variable


def matching(query_input, gallery_input, max_rank):
    query = torch.from_numpy(query_input).float()
    query_tensor = Variable(query).view(-1, 1)

    gallery = torch.from_numpy(np.asarray(gallery_input)).float()
    gallery_tensor = Variable(gallery)

    t1 = time.time()
    cal_sim = torch.mm(gallery_tensor, query_tensor).view(-1)
    t2 = time.time()

    if len(gallery_tensor) > max_rank:
        all_values, all_indices = torch.topk(cal_sim, max_rank)
    else:
        all_values, all_indices = torch.topk(cal_sim, len(gallery_tensor))
    t3 = time.time()
    print(f"end match process {t2-t1} {t3-t2}")
    return all_values, all_indices


def match_camera(idx, query_input, camera_patch, max_rank, send_end):
    t1 = time.time()
    gallery_input = []

    for i, patch in enumerate(camera_patch):
        # TODO: 수정 필요
        gallery_input.append(np.load("test.npy"))
        # camera_input.append(np.load(patch['featurePath']))

    t2 = time.time()
    values, indices = matching(query_input, gallery_input, max_rank)
    t3 = time.time()
    print(f"end! {t2-t1} {t3-t2}")

    data = pickle.dumps([idx, values.numpy(), indices.numpy()])
    send_end.send(data)
