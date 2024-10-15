import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose


class Resize(object):
    def __init__(
            self,
            width,
            height,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=1,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_AREA,
    ):
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)
        return y

    def get_size(self, width, height):
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")
        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")
        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
        sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)
        if self.__resize_target:
            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)
            if "mask" in sample:
                sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        return sample

class NormalizeImage(object):
    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std
        return sample

class PrepareForNet(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)
        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)
        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])
        return sample

def image2tensor(raw_image, input_size=518):
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='upper_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    h, w = raw_image.shape[:2]

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0)

    return image, (h, w)


def pre_process(img_path):
    # input_info = [((1, 3, 518, 518), "float32")]
    raw_img = cv2.imread(img_path)
    image, (h, w) = image2tensor(raw_img, 518)

    b, c, h_new, w_new = image.shape
    img_pad = torch.zeros((b, c, 518, 518))
    img_pad[:, :, :h_new, :w_new] = image
    return img_pad.numpy().flatten(), h_new, w_new, h, w


def post_process(array, h_new, w_new, h, w):
    # out = np.array(java_array).reshape((1, 518, 518))
    out = array.reshape((518, 518))
    out = out[:h_new, :w_new]
    # depth = F.interpolate(out[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
    # depth = depth.cpu().numpy()
    depth = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    depth_normalized = np.uint8(depth_normalized)[:, :, None]  # (h, w, 1)

    # Apply a colormap (optional, but helps visualize depth)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO) # (h, w, 3)

    def convert_to_argb(image):
        # Ensure the image is in RGB format
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Create an alpha channel with full opacity (255)
        alpha_channel = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)

        # Split the image into separate R, G, B channels
        b_channel, g_channel, r_channel = cv2.split(image)

        # Merge the channels into ARGB format
        argb_image = cv2.merge((alpha_channel, r_channel, g_channel, b_channel))

        # Combine the channels into a single 32-bit ARGB value
        argb_packed = (
                (alpha_channel.astype(np.uint32) << 24) |  # Alpha channel (highest 8 bits)
                (r_channel.astype(np.uint32) << 16) |      # Red channel
                (g_channel.astype(np.uint32) << 8) |       # Green channel
                b_channel.astype(np.uint32)                # Blue channel (lowest 8 bits)
        )

        return argb_packed

    argb_image = convert_to_argb(depth_colormap)  # (h, w, 4)

    return argb_image.astype(np.int32).flatten(), argb_image.shape
    # return convert_to_argb(depth_normalized).astype(np.int32).flatten(), depth_normalized.shape  # (h, w, 1)


# def run_inference(img_path):
#     ### read image and scale to 518
#     input_info = [((1, 3, 518, 518), "float32")]
#     raw_img = cv2.imread(img_path)
#     image, (h, w) = image2tensor(raw_img, 518)
#
#     ### padding
#     b, c, h_new, w_new = image.shape
#     img_pad = torch.zeros((b, c, 518, 518))
#     img_pad[:, :, :h_new, :w_new] = image
#
#     #########################################################################
#     name = "compiled_model_android"
#     target = tvm.target.Target("llvm -mtriple=arm64-linux-android")
#     dev = tvm.device(str(target))
#     dtype = "float32"
#     #########################################################################
#
#     ### load and inference
#     lib = tvm.runtime.load_module(f"{name}.so")
#
#     m = graph_executor.GraphModule(lib["default"](dev))
#     dev_data = tvm.nd.array(img_pad.numpy(), dev)
#     m.set_input("input", dev_data)
#     m.run()
#     tvm_output = m.get_output(0)
#     out = torch.from_numpy(tvm_output.numpy())
#
#     ### postprocess
#     out = out[:, :h_new, :w_new]
#     depth = F.interpolate(out[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
#     depth = depth.cpu().numpy()
#
#     depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
#     # Convert to uint8
#     depth_normalized = np.uint8(depth_normalized)
#     # Apply a colormap (optional, but helps visualize depth)
#     depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
#
#     return depth_colormap
