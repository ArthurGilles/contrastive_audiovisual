# Note : This code comes from the VisualVoice repository
import torch
import torch.nn as nn


def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class AudioEncoder(nn.Module):
    # Expects input shape: [batch, 2, freq_bins, time_frames] (e.g., mag/phase or real/imag)
    def __init__(self, ngf=64, input_nc=2):
        super(AudioEncoder, self).__init__()
        # initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = conv_block(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = conv_block(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer8 = conv_block(ngf * 8, ngf * 8)
        self.frequency_pool = nn.MaxPool2d([2, 1])  # Pool along frequency axis

    def forward(self, audio_stft):
        # Input audio_stft assumed to be [batch, channels, freq, time]
        # NO CHANGE NEEDED HERE: The CNN layers handle variable time dim,
        # and the final GAP produces a fixed-size output.
        audio_conv1feature = self.audionet_convlayer1(audio_stft)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv3feature = self.frequency_pool(audio_conv3feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv4feature = self.frequency_pool(audio_conv4feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv5feature = self.frequency_pool(audio_conv5feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv6feature = self.frequency_pool(audio_conv6feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_conv7feature = self.frequency_pool(audio_conv7feature)
        audio_conv8feature = self.audionet_convlayer8(audio_conv7feature)
        audio_conv8feature = self.frequency_pool(
            audio_conv8feature
        )  # Final feature map

        # Global pooling (average over frequency and time)
        # Output shape after pooling: [batch, ngf*8] - This is FIXED size.
        pooled_features = torch.mean(
            audio_conv8feature, dim=[2, 3]
        )  # Global Average Pooling

        return pooled_features
