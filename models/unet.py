from torch import nn
import torch

# encoding block
class encoding_block(nn.Module):
    """
    Convolutional batch norm block with relu activation (main block used in the encoding steps)
    """

    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=3,
        padding=0,
        stride=1,
        dilation=1,
        batch_norm=True,
        dropout=False,
    ):
        super().__init__()

        if batch_norm:

            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
            ]

        else:
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
            ]

        if dropout:
            layers.append(nn.Dropout())

        self.encoding_block = nn.Sequential(*layers)

    def forward(self, input):

        output = self.encoding_block(input)

        return output


# decoding block
class decoding_block(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()

        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = encoding_block(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):

        output2 = self.up(input2)

        output1 = nn.functional.interpolate(input1, output2.size()[2:], mode="bilinear")

        return self.conv(torch.cat([output1, output2], 1))

class UNet_carbon(nn.Module):
    """
    Main UNet architecture
    """

    def __init__(self, num_classes, dropout=False):
        super().__init__()

        # encoding
        # self.conv1 = encoding_block(3, 64, dropout=dropout)
        self.conv1 = encoding_block(4, 64, dropout=dropout)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = encoding_block(64, 128, dropout=dropout)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = encoding_block(128, 256, dropout=dropout)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = encoding_block(256, 512, dropout=dropout)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # center
        self.center = encoding_block(512, 1024, dropout=dropout)

        # decoding
        self.decode4 = decoding_block(1024, 512)
        self.decode3 = decoding_block(512, 256)
        self.decode2 = decoding_block(256, 128)
        self.decode1 = decoding_block(128, 64)

        # final
        self.final_cls = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_reg = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, input):

        # encoding
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # center
        center = self.center(maxpool4)

        # decoding
        decode4 = self.decode4(conv4, center)

        decode3 = self.decode3(conv3, decode4)

        decode2 = self.decode2(conv2, decode3)

        decode1 = self.decode1(conv1, decode2)

        # final
        final_cls = nn.functional.interpolate(
            self.final_cls(decode1), input.size()[2:], mode="bilinear"
        )
        ################### first upsample, after conv
        # final_reg = self.final_reg(nn.functional.interpolate(
        #     decode1, input.size()[2:], mode="bilinear"
        # ))
        ################### first conv, after upsample
        final_reg = nn.functional.interpolate(
            self.final_reg(decode1), input.size()[2:], mode="bilinear"
        )

        return final_cls, final_reg


