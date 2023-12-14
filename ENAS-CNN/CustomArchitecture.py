import torch.nn as nn

# class depthwise_separable_conv(nn.Module):
#     def __init__(self, nin, kernels_per_layer, nout):
#         super(depthwise_separable_conv, self).__init__()
#         self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
#         self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)
#
#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out

class CustomModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomModule,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_conv_blocks = 2
        self.total_layers = 5
        self.reduction_positions = [i for i in range(self.num_conv_blocks,
                                                     self.total_layers + 1,
                                                     self.num_conv_blocks + 1)]
        self.module_list = nn.ModuleList([])

        self.compile_model()

    def compile_model(self):
        for i in range(self.total_layers):
            if i not in self.reduction_positions:
                conv_module = self.module(stride=1, padding=0)
                self.module_list.append(conv_module)
            else:
                reduction_module = self.module(stride=2, padding=0)
                self.module_list.append(reduction_module)


    def module(self, stride, padding):
        cells = nn.ModuleList()

        # Kernel size 3
        cells.append(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=stride,
                      padding=padding))
        # Kernel size 5
        cells.append(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=stride,
                      padding=padding))

        # Depth wise separable
        d1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=5, stride=stride,
                      padding=padding,groups=self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5, stride=stride,
                      padding=padding)
        )
        cells.append(d1)

        d2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=stride,
                      padding=padding,groups=self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=stride,
                      padding=padding)
        )
        cells.append(d2)

        cells.append(nn.AvgPool2d(kernel_size=3, stride=stride, padding=padding))
        cells.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=padding))
        return cells

    def forward(self, x, selected_modules):
        for i in selected_modules:

        pass
