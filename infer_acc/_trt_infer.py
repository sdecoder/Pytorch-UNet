import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
from PIL import Image

import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torchvision.transforms
from torchvision import transforms, datasets

import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import cv2
import torch
import argparse
from tqdm import tqdm

from utils.utils import plot_img_and_mask

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
parser = argparse.ArgumentParser()
parser.add_argument(
  '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
  '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
  '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
  '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=False, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

opt = parser.parse_args()


class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()


def load_engine(trt_runtime, engine_path):
  with open(engine_path, 'rb') as f:
    engine_data = f.read()
  engine = trt_runtime.deserialize_cuda_engine(engine_data)
  return engine


def main():
  print("[trace] reach the main entry")
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if args.cuda else "cpu")

  engine_file = 'unet-simp.engine'
  if not os.path.exists(engine_file):
    print(f'[trace] target engine file {engine_file} not found, exit')
    exit(-1)

  engine = load_engine(trt.Runtime(TRT_LOGGER), engine_file)
  print('[trace] reach func@allocate_buffers')
  inputs = []
  outputs = []
  bindings = []

  binding_to_type = {}
  binding_to_type['input'] = np.float32
  binding_to_type['output'] = np.float32

  print('[trace] initiating bindings for TensorRT')
  for binding in engine:
    print(f'[trace] current binding: {str(binding)}')
    _binding_shape = engine.get_binding_shape(binding)
    _volume = trt.volume(_binding_shape)
    size = _volume * engine.max_batch_size
    print(f'[trace] current binding size: {size}')
    dtype = binding_to_type[str(binding)]
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
      inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(HostDeviceMem(host_mem, device_mem))

  print('[trace] initiating TensorRT object')
  batch_size = 1
  context = engine.create_execution_context()
  stream = cuda.Stream()
  print(f'[trace] reach line@111')
  full_img = Image.open("../data/test/input.jpg")
  from utils.data_loading import BasicDataset
  new_size = (959, 640)
  resized_img = full_img.resize(new_size)
  #resized_img.show()
  img = torch.from_numpy(BasicDataset.preprocess(resized_img, 1, is_mask=False))
  img = img.unsqueeze(0)
  #img = img.to(device=device, dtype=torch.float32)
  #  im.show()

  print(f'[trace] reach line@116')
  tensor = img
  # This method will show image in any image viewer
  np.copyto(inputs[0].host, tensor.ravel())
  [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
  # Run inference.
  context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
  [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
  stream.synchronize()
  print(f'[trace] reach line@125')
  retObj = torch.from_numpy(outputs[0].host)
  tensor = torch.reshape(retObj, (1, 2, 640, 959))
  import torch.nn.functional as F
  n_classes = 2
  probs = F.softmax(tensor, dim=1)[0]
  tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((full_img.size[1], full_img.size[0])),
    transforms.ToTensor()
  ])

  full_mask = tf(probs.cpu()).squeeze()
  onehot = F.one_hot(full_mask.argmax(dim=0), n_classes).permute(2, 0, 1).numpy()
  plot_img_and_mask(full_img, onehot)
  '''
  transform_to_pil = transforms.ToPILImage()
  _image = _transform_to_pil(tensor)
  _image.show()
  '''

  pass


if __name__ == "__main__":
  main()
  pass
