import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import onnx
import onnxruntime
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


def main():

  _onnx_file = 'unet-simp.onnx'
  onnx_model = onnx.load(_onnx_file)
  onnx.checker.check_model(onnx_model)

  def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

  # compute ONNX Runtime output prediction

  im = Image.open("../data/test/input.jpg")
  im = im.resize((959, 640))
  #  im.show()

  _transform_to_tensor = transforms.Compose([
    transforms.PILToTensor()
  ])
  print(f'[trace] reach line@116')
  tensor = _transform_to_tensor(im)
  tensor = tensor.type(torch.float32)
  ort_session = onnxruntime.InferenceSession(_onnx_file, providers=['CPUExecutionProvider'])
  ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(tensor)}
  ort_inputs = ort_inputs['input'].reshape((1, 3, 640, 959))
  input_pair = {'input': ort_inputs}

  ort_outs = ort_session.run(None, input_pair)
  im = Image.fromarray(np.uint8(ort_outs[0] * 255))
  im.show()
  # compare ONNX Runtime and PyTorch results



  pass

if __name__ == '__main__':
  main()
