import torch
import torch.nn as nn

class InvertedResidual_nn(nn.Module):
    def __init__(self, input_channels, filters, strides, bias=False, activation=nn.functional.relu, kernel_size=3, expansion_factor=6,
                 regularizer=None, **kwargs):
        super(InvertedResidual_nn, self).__init__()
        self.filters = filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.activation = activation
        self.regularizer = regularizer
        self.bias = bias
        self.channel_axis = 1 #if backend.image_data_format() == 'channels_first' else -1
        self.expanded_channels = int(input_channels*self.expansion_factor)
        self.ptwise_conv1 = torch.nn.Conv2d(in_channels=input_channels, out_channels=self.expanded_channels,
                                   kernel_size=1, bias=self.bias)
        self.dwise = torch.nn.Conv2d(in_channels=self.expanded_channels, out_channels=self.expanded_channels, kernel_size=self.kernel_size, stride=self.strides, groups=self.expanded_channels,
                                     bias=self.bias)
        self.ptwise_conv2 = torch.nn.Conv2d(in_channels=self.expanded_channels, out_channels=self.filters, kernel_size=1,
                                bias=self.bias)
        self.bn1 = torch.nn.BatchNorm2d(int(input_channels*self.expansion_factor))
        self.bn2 = torch.nn.BatchNorm2d(int(input_channels*self.expansion_factor))
        self.bn3 = torch.nn.BatchNorm2d(self.filters)

    # def build(self, input_shape):
        

    def forward(self, input_x):
        # Expansion
        x = self.ptwise_conv1(input_x)
        x = self.bn1(x)
        x = self.activation(x)
        # Spatial filtering
        x = self.dwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        # back to low-channels w/o activation
        x = self.ptwise_conv2(x)
        x = self.bn3(x)
        # Residual connection only if i/o have same spatial and depth dims
        # tf.print("start")
        # tf.print(input_x.shape[1:])
        # tf.print(x.shape[1:])
        if input_x.shape[1:] == x.shape[1:]:
            # tf.print("here")
            x += input_x
        return x