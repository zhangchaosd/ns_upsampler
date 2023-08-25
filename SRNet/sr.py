import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import onnx

def float32_to_uint8(inputs):
    return torch.clip(torch.round(inputs * 255), 0, 255).to(torch.uint8)

def my_padding(input, size):
    # input = F.pad(input, (1, 2, 1, 2), mode="replicate")
    top, bottom, left, right = size
    if top != 0:
        top = input[:,:,:top,:]
        input = torch.cat([top,input], dim=2)
    if bottom != 0:
        bottom = input[:,:,-bottom:,:]
        input = torch.cat([input,bottom], dim=2)
    if left != 0:
        left = input[:,:,:,:left]
        input = torch.cat([left,input], dim=3)
    if right != 0:
        right = input[:,:,:,-right:]
        input = torch.cat([input,right], dim=3)
    return input


class BicubicUpsample(nn.Module):
    """A bicubic upsampling class with similar behavior to that in TecoGAN-Tensorflow

    Note that it's different from torch.nn.functional.interpolate and
    matlab's imresize in terms of bicubic kernel and sampling scheme

    Theoretically it can support any scale_factor >= 1, but currently only
    scale_factor = 4 is tested

    References:
        The original paper: http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
        https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsample, self).__init__()

        # calculate weights
        cubic = torch.FloatTensor(
            [
                [0, a, -2 * a, a],
                [1, 0, -(a + 3), a + 2],
                [0, -a, (2 * a + 3), -(a + 2)],
                [0, 0, a, -a],
            ]
        )  # accord to Eq.(6) in the reference paper

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s**2, s**3]))
            for s in [1.0 * d / scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer("kernels", torch.stack(kernels))

    def forward(self, input):
        n, c, h, w = input.size()
        s = self.scale_factor

        # pad input (left, right, top, bottom)
        input = my_padding(input, (1,2,1,2))

        # calculate output (height)
        kernel_h = self.kernels.repeat(c, 1).view(-1, 1, 4, 1)
        output = F.conv2d(input, kernel_h, stride=1, padding=0, groups=c)
        output = (
            output.reshape(n, c, s, -1, w + 3)
            .permute(0, 1, 3, 2, 4)
            .reshape(n, c, -1, w + 3)
        )

        # calculate output (width)
        kernel_w = self.kernels.repeat(c, 1).view(-1, 1, 1, 4)
        output = F.conv2d(output, kernel_w, stride=1, padding=0, groups=c)
        output = (
            output.reshape(n, c, s, h * s, -1)
            .permute(0, 1, 3, 4, 2)
            .reshape(n, c, h * s, -1)
        )
        return output


def space_to_depth(x, scale=4):
    """Equivalent to tf.space_to_depth()"""

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output




# -------------------- generator modules -------------------- #
class FNet(nn.Module):
    """Optical flow estimation network"""

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2 * in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True),
        )

    def forward(self, x1, x2):
        """Compute optical flow from x1 to x2"""
        out = torch.cat([x1, x2], dim=1)
        out = self.encoder1(out)
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode="bilinear", align_corners=False
        )
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode="bilinear", align_corners=False
        )
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode="bilinear", align_corners=False
        )
        out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

        return out


class ResidualBlock(nn.Module):
    """Residual block without batch normalization"""

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SRNet(nn.Module):
    """Reconstruction & Upsampling network"""

    def __init__(self, in_nc=3, out_nc=3, nf=16, nb=10, upsample_func=None, scale=2):
        super(SRNet, self).__init__()

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.conv_up_cheap = nn.Sequential(
            nn.PixelShuffle(scale), nn.ReLU(inplace=True)
        )

        # output conv.
        self.conv_out = nn.Conv2d(4, out_nc, 3, 1, 1, bias=True)

        # upsampling function
        self.upsample_func = upsample_func

    def forward(self, lr_curr, hr_prev_tran):
        """lr_curr: the current lr data in shape nchw
        hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """
        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        out = self.resblocks(out)
        out = self.conv_up_cheap(out)
        out = self.conv_out(out)
        # out += self.upsample_func(lr_curr)
        return out


class FRNet(nn.Module):
    """Frame-recurrent network proposed in https://arxiv.org/abs/1801.04590"""

    def __init__(self, in_nc=3, out_nc=3, nb=10, scale=2):
        super(FRNet, self).__init__()

        self.scale = scale

        self.upsample_func = BicubicUpsample(self.scale)

        # define fnet & srnet
        self.fnet = FNet(in_nc)
        self.srnet = SRNet(
            in_nc, out_nc, 4 * scale * scale, nb, self.upsample_func, scale=scale
        )

        # setup params
        self.lr_prev = None
        self.hr_prev = None

        
    def backward_warp(self, x, flow, mode="bilinear", padding_mode="border"):
        """Backward warp `x` according to `flow`

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
        """
        n, c, h, w = x.shape

        # create mesh grid
        iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
        iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
        grid = torch.cat([iu, iv], 1).to(flow.device)

        # normalize flow to [-1, 1]
        flow = torch.cat(
            [flow[:, 0:1, ...] / ((w - 1.0) / 2.0), flow[:, 1:2, ...] / ((h - 1.0) / 2.0)],
            dim=1,
        )

        # add flow to grid and reshape to nhw2
        grid = (grid + flow).permute(0, 2, 3, 1)

        # bilinear sampling
        # Note: `align_corners` is set to `True` by default in PyTorch version
        #        lower than 1.4.0
        if int("".join(torch.__version__.split(".")[:2])) >= 14:
            output = F.grid_sample(
                x, grid, mode=mode, padding_mode=padding_mode, align_corners=True
            )
        else:
            output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

        return output

    def forward_core(self, lr_curr, lr_prev, hr_prev):
        """
        Parameters:
            :param lr_curr: the current lr data in shape nchw
            :param lr_prev: the previous lr data in shape nchw
            :param hr_prev: the previous hr data in shape nc(4h)(4w)
        """

        # estimate lr flow (lr_curr -> lr_prev)
        lr_flow = self.fnet(lr_curr, lr_prev)

        # pad_h = lr_curr.size(2) - lr_curr.size(2) // 8 * 8
        # pad_w = lr_curr.size(3) - lr_curr.size(3) // 8 * 8
        # lr_flow = F.pad(lr_flow, (0, pad_w, 0, pad_h), "reflect")


        # upsample lr flow
        hr_flow = self.scale * self.upsample_func(lr_flow)

        # warp hr_prev
        hr_prev_warp = self.backward_warp(hr_prev, hr_flow)

        # compute hr_curr
        hr_curr = self.srnet(lr_curr, space_to_depth(hr_prev_warp, self.scale))

        return hr_curr

    def forward_sequence(self, lr_data):
        """
        Parameters:
            :param lr_data: lr data in shape ntchw
        """

        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # calculate optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_flow = self.fnet(lr_curr, lr_prev)  # n*(t-1),2,h,w


        # upsample lr flows
        hr_flow = self.scale * self.upsample_func(lr_flow)
        hr_flow = hr_flow.view(n, (t - 1), 2, hr_h, hr_w)

        # compute the first hr data
        hr_data = []
        hr_prev = self.srnet(
            lr_data[:, 0, ...],
            torch.zeros(
                n,
                (self.scale**2) * c,
                lr_h,
                lr_w,
                dtype=torch.float32,
                device=lr_data.device,
            ),
        )
        hr_data.append(hr_prev)

        # compute the remaining hr data
        for i in range(1, t):
            # warp hr_prev
            hr_prev_warp = self.backward_warp(hr_prev, hr_flow[:, i - 1, ...])

            # compute hr_curr
            hr_curr = self.srnet(
                lr_data[:, i, ...], space_to_depth(hr_prev_warp, self.scale)
            )

            # save and update
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w

        return hr_data

    def forward(self, lr_curr):
        """
        Parameters:
            :param lr_curr: lr data in shape 1chw
        """
        if self.lr_prev is None:
            _, c, h, w = lr_curr.shape
            self.lr_prev = lr_curr
            self.hr_prev = self.srnet(
                lr_curr,
                torch.zeros(
                    1,
                    (self.scale**2) * c,
                    h,
                    w,
                    dtype=torch.float32,
                    device=lr_curr.device,
                ),
            )
        hr_curr = self.forward_core(lr_curr, self.lr_prev, self.hr_prev)
        self.lr_prev, self.hr_prev = lr_curr, hr_curr

        return hr_curr

class WrapperRGB(nn.Module):
    def __init__(self):
        super(WrapperRGB, self).__init__()
        self.model = FRNet()
    
    def forward(self, img):
        img = torch.permute(img, (2,0,1))  # h,w,3 -> 3,h,w  uint8
        img = img.unsqueeze(0)  # 3,h,w -> 1,3,h,w  uint8
        img = img.float() / 255
        img = self.model(img)
        img = float32_to_uint8(img)
        img = img.squeeze(0)  # 1,3,h,w -> 3,h,w
        img = torch.permute(img, (1,2,0))  # 3,h,w -> h,w,3
        return img

class WrapperBGRA(nn.Module):
    def __init__(self):
        super(WrapperBGRA, self).__init__()
        self.model = FRNet()
    
    def forward(self, img):
        img = img[:, :, :3]  # bgra -> bgr
        img = torch.permute(img, (2,0,1))  # h,w,3 -> 3,h,w  uint8
        img = img.unsqueeze(0)  # 3,h,w -> 1,3,h,w  uint8
        img = img.float() / 255
        img = self.model(img)
        img = float32_to_uint8(img)
        img = img.squeeze(0)  # 1,3,h,w -> 3,h,w
        alpha = torch.ones((1, 2160, 3840)) * 255
        alpha = alpha.to(torch.uint8)
        img = torch.cat([img, alpha], dim=0)  # bgr -> bgra
        img = torch.permute(img, (1,2,0))  # 4,h,w -> h,w,4
        return img

def export(model, path=".", name="SRNet.onnx"):
    model.eval().to("cpu")
    m = WrapperRGB()
    m.model = model
    input = torch.Tensor(1080, 1920, 3).to(torch.uint8).to("cpu")
    # input_seq = torch.Tensor(2,2,3,540,960).to("cuda")
    m.eval()
    torch.onnx.export(m,
        input,
        os.path.join(path, name),
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=16,
    )
    m = onnx.load(os.path.join(path, name))
    onnx.checker.check_model(m)

def export_bgra(model, path=".", name="SRNet_bgra.onnx"):
    model.eval().to("cpu")
    m = WrapperBGRA()
    m.model = model
    input = torch.Tensor(1080, 1920, 4).to(torch.uint8).to("cpu")
    m.eval()
    torch.onnx.export(m,
        input,
        os.path.join(path, name),
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=16,
    )
    m = onnx.load(os.path.join(path, name))
    onnx.checker.check_model(m)

# m = FRNet()
# export_bgra(m)
# exit()
# m.to("cuda")
# input_seq = torch.Tensor(4,15,3,256,256).to("cuda")
# o = m.forward_sequence(input_seq)
# print(o.shape)