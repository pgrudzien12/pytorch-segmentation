#
# converts a saved PyTorch model to ONNX format
# 
import os
import argparse

import torch
import torchvision.models as models


# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='model_best.pth', help="path to input PyTorch model (default: model_best.pth)")
parser.add_argument('--output', type=str, default='', help="desired path of converted ONNX model (default: <ARCH>.onnx)")
parser.add_argument('--model-dir', type=str, default='', help="directory to look for the input PyTorch model in, and export the converted ONNX model to (if --output doesn't specify a directory)")

opt = parser.parse_args() 
print(opt)

# format input model path
if opt.model_dir:
   opt.model_dir = os.path.expanduser(opt.model_dir)
   opt.input = os.path.join(opt.model_dir, opt.input)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))

# load the model checkpoint
print('loading checkpoint:  ' + opt.input)
checkpoint = torch.load(opt.input, map_location=device)

arch = checkpoint['arch']
num_classes = checkpoint['num_classes']

print('checkpoint accuracy: {:.3f}% mean IoU, {:.3f}% accuracy'.format(checkpoint['mean_IoU'], checkpoint['accuracy']))

# create the model architecture
print('using model:  ' + arch)
print('num classes:  ' + str(num_classes))

model = getattr(models.segmentation, arch)(num_classes=num_classes,
                                           aux_loss=None,
                                           pretrained=False)
m = checkpoint['model']
del m["aux_classifier.0.weight"]
del m["aux_classifier.1.weight"]
del m["aux_classifier.1.bias"]
del m["aux_classifier.1.running_mean"]
del m["aux_classifier.1.running_var"]
del m["aux_classifier.1.num_batches_tracked"]
del m["aux_classifier.4.weight"]
del m["aux_classifier.4.bias"]
# load the model weights
model.load_state_dict(checkpoint['model'])

model.to(device)
model.eval()

print(model)
print('')

# create example image data
resolution = checkpoint['resolution']
input = torch.ones((1, 3, resolution[0], resolution[1])).to(device)
print('input size:  {:d}x{:d}'.format(resolution[1], resolution[0]))

opset_version = 10
# format output model path
if not opt.output:
   opt.output = arch + f'-opset-{opset_version}.onnx'

if opt.model_dir and opt.output.find('/') == -1 and opt.output.find('\\') == -1:
   opt.output = os.path.join(opt.model_dir, opt.output)

# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to ONNX...')
torch.onnx.export(model, input, opt.output, verbose=True, input_names=input_names, output_names=output_names, opset_version=opset_version)
print('model exported to:  {:s}'.format(opt.output))


