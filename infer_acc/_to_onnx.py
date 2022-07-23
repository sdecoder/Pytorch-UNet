import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx



def get_args():
  parser = argparse.ArgumentParser(description='Predict masks from input images')
  parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                      help='Specify the file in which the model is stored')
  parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
  parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
  parser.add_argument('--viz', '-v', action='store_true',
                      help='Visualize the images as they are processed')
  parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
  parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                      help='Minimum probability value to consider a mask pixel white')
  parser.add_argument('--scale', '-s', type=float, default=0.5,
                      help='Scale factor for the input images')
  parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

  return parser.parse_args()

def get_output_filenames(args):
  def _generate_name(fn):
    return f'{os.path.splitext(fn)[0]}_OUT.png'

  return args.output or list(map(_generate_name, args.input))

def _to_onnx(model, device):
  # Input to the model
  batch_size = 1
  input = torch.randn(batch_size, 3, 640, 959, requires_grad=True).to(device)
  output_path = "unet.onnx"
  # Export the model
  torch.onnx.export(model,  # model being run
                    input,  # model input (or a tuple for multiple inputs)
                    output_path,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=15,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],  # the model's input names
                    output_names=['output'])

  print(f'[trace] the onnx model generated.')
  pass

def main():

  _weight_file = '../checkpoints/checkpoint_epoch6.pth'
  if not os.path.exists(_weight_file):
    print(f'[trace] target weight file {_weight_file} not found, exit(-1)')
    exit(-1)

  '''
  args = get_args()
  in_files = args.input
  out_files = get_output_filenames(args)
  '''

  net = UNet(n_channels=3, n_classes=2, bilinear=False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info(f'Loading model {_weight_file}')
  logging.info(f'Using device {device}')
  net.to(device=device)
  net.load_state_dict(torch.load(_weight_file, map_location=device))
  print(f'[trace] model weight file has been loaded')
  print(f'[trace] inspecting model: {type(net)}')
  print(f'[trace] {net}')
  _to_onnx(net, device)

  pass

if __name__ == '__main__':
  main()
