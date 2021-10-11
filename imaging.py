import numpy as np  # array operations
import math         # basing math operations
import utility
import time         # measure runtime
import debayer
import sys          # float precision

# =============================================================
# class: ImageInfo
#   Helps set up necessary information/metadata of the image
# =============================================================
class ImageInfo:
    def __init__(self, name = "unknown", data = -1, is_show = False):
        self.name   = name
        self.data   = data
        self.size   = np.shape(self.data)
        self.is_show = is_show
        self.color_space = "unknown"
        self.bayer_pattern = "unknown"
        self.channel_gain = (1.0, 1.0, 1.0, 1.0)
        self.bit_depth = 0
        self.black_level = (0, 0, 0, 0)
        self.white_level = (1, 1, 1, 1)
        self.color_matrix = [[1., .0, .0],\
                             [.0, 1., .0],\
                             [.0, .0, 1.]] # xyz2cam
        self.min_value = np.min(self.data)
        self.max_value = np.max(self.data)
        self.data_type = self.data.dtype

        # Display image only isShow = True
        if (self.is_show):
            plt.imshow(self.data)
            plt.show()

    def set_data(self, data):
        # This function updates data and corresponding fields
        self.data = data
        self.size = np.shape(self.data)
        self.data_type = self.data.dtype
        self.min_value = np.min(self.data)
        self.max_value = np.max(self.data)

    def get_size(self):
        return self.size

    def get_width(self):
        return self.size[1]

    def get_height(self):
        return self.size[0]

    def get_depth(self):
        if np.ndim(self.data) > 2:
            return self.size[2]
        else:
            return 0

    def set_color_space(self, color_space):
        self.color_space = color_space

    def get_color_space(self):
        return self.color_space

    def set_channel_gain(self, channel_gain):
        self.channel_gain = channel_gain

    def get_channel_gain(self):
        return self.channel_gain

    def set_color_matrix(self, color_matrix):
        self.color_matrix = color_matrix

    def get_color_matrix(self):
        return self.color_matrix

    def set_bayer_pattern(self, bayer_pattern):
        self.bayer_pattern = bayer_pattern

    def get_bayer_pattern(self):
        return self.bayer_pattern

    def set_bit_depth(self, bit_depth):
        self.bit_depth = bit_depth

    def get_bit_depth(self):
        return self.bit_depth

    def set_black_level(self, black_level):
        self.black_level = black_level

    def get_black_level(self):
        return self.black_level

    def set_white_level(self, white_level):
        self.white_level = white_level

    def get_white_level(self):
        return self.white_level

    def get_min_value(self):
        return self.min_value

    def get_max_value(self):
        return self.max_value

    def get_data_type(self):
        return self.data_type

    def __str__(self):
        return "Image " + self.name + " info:" + \
                          "\n\tname:\t" + self.name + \
                          "\n\tsize:\t" + str(self.size) + \
                          "\n\tcolor space:\t" + self.color_space + \
                          "\n\tbayer pattern:\t" + self.bayer_pattern + \
                          "\n\tchannel gains:\t" + str(self.channel_gain) + \
                          "\n\tbit depth:\t" + str(self.bit_depth) + \
                          "\n\tdata type:\t" + str(self.data_type) + \
                          "\n\tblack level:\t" + str(self.black_level) + \
                          "\n\tminimum value:\t" + str(self.min_value) + \
                          "\n\tmaximum value:\t" + str(self.max_value)

# =============================================================
# class: demosaic
# =============================================================
class demosaic:
    def __init__(self, data, bayer_pattern="rggb", clip_range=[0, 65535], name="demosaic"):
        self.data = np.float32(data)
        self.bayer_pattern = bayer_pattern
        self.clip_range = clip_range
        self.name = name

    def mhc(self, timeshow=False):

        print("----------------------------------------------------")
        print("Running demosaicing using Malvar-He-Cutler algorithm...")

        return debayer.debayer_mhc(self.data, self.bayer_pattern, self.clip_range, timeshow)

    def post_process_local_color_ratio(self, beta):
        # Objective is to reduce high chroma jump
        # Beta is controlling parameter, higher gives more effect,
        # however, too high does not make any more change

        print("----------------------------------------------------")
        print("Demosaicing post process using local color ratio...")

        data = self.data

        # add beta with the data to prevent divide by zero
        data_beta = self.data + beta

        # convolution kernels
        # zeta1 averages the up, down, left, and right four values of a 3x3 window
        zeta1 = np.multiply([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]], .25)
        # zeta2 averages the four corner values of a 3x3 window
        zeta2 = np.multiply([[1., 0., 1.], [0., 0., 0.], [1., 0., 1.]], .25)

        # average of color ratio
        g_over_b = signal.convolve2d(np.divide(data_beta[:, :, 1], data_beta[:, :, 2]), zeta1, mode="same", boundary="symm")
        g_over_r = signal.convolve2d(np.divide(data_beta[:, :, 1], data_beta[:, :, 0]), zeta1, mode="same", boundary="symm")
        b_over_g_zeta2 = signal.convolve2d(np.divide(data_beta[:, :, 2], data_beta[:, :, 1]), zeta2, mode="same", boundary="symm")
        r_over_g_zeta2 = signal.convolve2d(np.divide(data_beta[:, :, 0], data_beta[:, :, 1]), zeta2, mode="same", boundary="symm")
        b_over_g_zeta1 = signal.convolve2d(np.divide(data_beta[:, :, 2], data_beta[:, :, 1]), zeta1, mode="same", boundary="symm")
        r_over_g_zeta1 = signal.convolve2d(np.divide(data_beta[:, :, 0], data_beta[:, :, 1]), zeta1, mode="same", boundary="symm")

        # G at B locations and G at R locations
        if self.bayer_pattern == "rggb":
            # G at B locations
            data[1::2, 1::2, 1] = -beta + np.multiply(data_beta[1::2, 1::2, 2], g_over_b[1::2, 1::2])
            # G at R locations
            data[::2, ::2, 1] = -beta + np.multiply(data_beta[::2, ::2, 0], g_over_r[::2, ::2])
            # B at R locations
            data[::2, ::2, 2] = -beta + np.multiply(data_beta[::2, ::2, 1], b_over_g_zeta2[::2, ::2])
            # R at B locations
            data[1::2, 1::2, 0] = -beta + np.multiply(data_beta[1::2, 1::2, 1], r_over_g_zeta2[1::2, 1::2])
            # B at G locations
            data[::2, 1::2, 2] = -beta + np.multiply(data_beta[::2, 1::2, 1], b_over_g_zeta1[::2, 1::2])
            data[1::2, ::2, 2] = -beta + np.multiply(data_beta[1::2, ::2, 1], b_over_g_zeta1[1::2, ::2])
            # R at G locations
            data[::2, 1::2, 0] = -beta + np.multiply(data_beta[::2, 1::2, 1], r_over_g_zeta1[::2, 1::2])
            data[1::2, ::2, 0] = -beta + np.multiply(data_beta[1::2, ::2, 1], r_over_g_zeta1[1::2, ::2])

        elif self.bayer_pattern == "grbg":
            # G at B locations
            data[1::2, ::2, 1] = -beta + np.multiply(data_beta[1::2, ::2, 2], g_over_b[1::2, 1::2])
            # G at R locations
            data[::2, 1::2, 1] = -beta + np.multiply(data_beta[::2, 1::2, 0], g_over_r[::2, 1::2])
            # B at R locations
            data[::2, 1::2, 2] = -beta + np.multiply(data_beta[::2, 1::2, 1], b_over_g_zeta2[::2, 1::2])
            # R at B locations
            data[1::2, ::2, 0] = -beta + np.multiply(data_beta[1::2, ::2, 1], r_over_g_zeta2[1::2, ::2])
            # B at G locations
            data[::2, ::2, 2] = -beta + np.multiply(data_beta[::2, ::2, 1], b_over_g_zeta1[::2, ::2])
            data[1::2, 1::2, 2] = -beta + np.multiply(data_beta[1::2, 1::2, 1], b_over_g_zeta1[1::2, 1::2])
            # R at G locations
            data[::2, ::2, 0] = -beta + np.multiply(data_beta[::2, ::2, 1], r_over_g_zeta1[::2, ::2])
            data[1::2, 1::2, 0] = -beta + np.multiply(data_beta[1::2, 1::2, 1], r_over_g_zeta1[1::2, 1::2])

        elif self.bayer_pattern == "gbrg":
            # G at B locations
            data[::2, 1::2, 1] = -beta + np.multiply(data_beta[::2, 1::2, 2], g_over_b[::2, 1::2])
            # G at R locations
            data[1::2, ::2, 1] = -beta + np.multiply(data_beta[1::2, ::2, 0], g_over_r[1::2, ::2])
            # B at R locations
            data[1::2, ::2, 2] = -beta + np.multiply(data_beta[1::2, ::2, 1], b_over_g_zeta2[1::2, ::2])
            # R at B locations
            data[::2, 1::2, 0] = -beta + np.multiply(data_beta[::2, 1::2, 1], r_over_g_zeta2[::2, 1::2])
            # B at G locations
            data[::2, ::2, 2] = -beta + np.multiply(data_beta[::2, ::2, 1], b_over_g_zeta1[::2, ::2])
            data[1::2, 1::2, 2] = -beta + np.multiply(data_beta[1::2, 1::2, 1], b_over_g_zeta1[1::2, 1::2])
            # R at G locations
            data[::2, ::2, 0] = -beta + np.multiply(data_beta[::2, ::2, 1], r_over_g_zeta1[::2, ::2])
            data[1::2, 1::2, 0] = -beta + np.multiply(data_beta[1::2, 1::2, 1], r_over_g_zeta1[1::2, 1::2])

        elif self.bayer_pattern == "bggr":
            # G at B locations
            data[::2, ::2, 1] = -beta + np.multiply(data_beta[::2, ::2, 2], g_over_b[::2, ::2])
            # G at R locations
            data[1::2, 1::2, 1] = -beta + np.multiply(data_beta[1::2, 1::2, 0], g_over_r[1::2, 1::2])
            # B at R locations
            data[1::2, 1::2, 2] = -beta + np.multiply(data_beta[1::2, 1::2, 1], b_over_g_zeta2[1::2, 1::2])
            # R at B locations
            data[::2, ::2, 0] = -beta + np.multiply(data_beta[::2, ::2, 1], r_over_g_zeta2[::2, ::2])
            # B at G locations
            data[::2, 1::2, 2] = -beta + np.multiply(data_beta[::2, 1::2, 1], b_over_g_zeta1[::2, 1::2])
            data[1::2, ::2, 2] = -beta + np.multiply(data_beta[1::2, ::2, 1], b_over_g_zeta1[1::2, ::2])
            # R at G locations
            data[::2, 1::2, 0] = -beta + np.multiply(data_beta[::2, 1::2, 1], r_over_g_zeta1[::2, 1::2])
            data[1::2, ::2, 0] = -beta + np.multiply(data_beta[1::2, ::2, 1], r_over_g_zeta1[1::2, ::2])


        return np.clip(data, self.clip_range[0], self.clip_range[1])


    def directionally_weighted_gradient_based_interpolation(self):
        # Reference:
        # http://www.arl.army.mil/arlreports/2010/ARL-TR-5061.pdf

        print("----------------------------------------------------")
        print("Running demosaicing using directionally weighted gradient based interpolation...")

        # Fill up the green channel
        G = debayer.fill_channel_directional_weight(self.data, self.bayer_pattern)

        B, R = debayer.fill_br_locations(self.data, G, self.bayer_pattern)

        width, height = utility.helpers(self.data).get_width_height()
        output = np.empty((height, width, 3), dtype=np.float32)
        output[:, :, 0] = R
        output[:, :, 1] = G
        output[:, :, 2] = B

        return np.clip(output, self.clip_range[0], self.clip_range[1])


    def post_process_median_filter(self, edge_detect_kernel_size=3, edge_threshold=0, median_filter_kernel_size=3, clip_range=[0, 65535]):
        # Objective is to reduce the zipper effect around the edges
        # Inputs:
        #   edge_detect_kernel_size: the neighborhood size used to detect edges
        #   edge_threshold: the threshold value above which (compared against)
        #                   the gradient_magnitude to declare if it is an edge
        #   median_filter_kernel_size: the neighborhood size used to perform
        #                               median filter operation
        #   clip_range: used for scaling in edge_detection
        #
        # Output:
        #   output: median filtered output around the edges
        #   edge_location: a debug image to see where the edges were detected
        #                   based on the threshold


        # detect edge locations
        edge_location = utility.edge_detection(self.data).sobel(edge_detect_kernel_size, "is_edge", edge_threshold, clip_range)

        # allocate space for output
        output = np.empty(np.shape(self.data), dtype=np.float32)

        if (np.ndim(self.data) > 2):

            for i in range(0, np.shape(self.data)[2]):
                output[:, :, i] = utility.helpers(self.data[:, :, i]).edge_wise_median(median_filter_kernel_size, edge_location[:, :, i])

        elif (np.ndim(self.data) == 2):
            output = utility.helpers(self.data).edge_wise_median(median_filter_kernel_size, edge_location)

        return output, edge_location

    def __str__(self):
        return self.name