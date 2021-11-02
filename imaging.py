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

# =============================================================
# function: cal_Kfactor
#   calculate K-factor
#   im_frame: a frame of RGB images (e.g 20 frames)
#   integration_time_cam: the integration time of the camera which depends on the camera model
#   G_channel_avg_cam: the average green value of the camera
#   expo_gain: the exposure gain
#   expo_time: the exposure time
# =============================================================
def cal_Kfactor(im_frame, frame_type="raw", integration_time_cam=0.01306536, G_channel_avg_cam=65, expo_gain=1.23, expo_time=0.0100178):
    G = 0

    if frame_type == "raw":
        for i in range(0, im_frame.shape[0]):
            temp = utility.helpers(im_frame[i].data).scale(im_frame[i].get_bit_depth(),16)
            (R, G1, G2, B) = utility.helpers(temp).bayer_channel_separation(im_frame[0].bayer_pattern)

            G = G + ((np.sum(G1) + np.sum(G2))*255)/(65535*im_frame[i].get_width()*im_frame[i].get_height())

        G_channel_avg_calib = G/im_frame.shape[0]

    elif frame_type == "png":
        for i in range(0, im_frame.shape[0]):
            for j in range(0, im_frame.shape[1]-1):
                for k in (0, im_frame.shape[2]-1):
                    G = G + im_frame[i][j][k][1]

        G_channel_avg_calib = G/(im_frame.shape[0]*im_frame[1]*im_frame[2])

    else:
        print("unsupported image type")

        return
    
    integration_time_calib = expo_gain * expo_time
    Kfactor = 2.8 * (integration_time_cam / G_channel_avg_cam) / (integration_time_calib / G_channel_avg_calib)

    return Kfactor

# def cal_Kfactor(integration_time_cam=0.01306536, G_channel_avg_cam=65, expo_gain=1.23, expo_time=0.0100178, G_channel_avg_calib=100):
#     integration_time_calib = expo_gain * expo_time
#     Kfactor = 2.8 * (integration_time_cam / G_channel_avg_cam) / (integration_time_calib / G_channel_avg_calib)

#     return Kfactor

# =============================================================
# function: In_or_Out
#   decide wether an image is shot indoor or outdoor, the function will return "in", "out" or "other" as strings
#   Kfactor: the K-factor
#   intepration_time: the intepration time of the camera
#   SensorG: the sensor gain of the camera
# =============================================================
def In_or_Out(Kfactor, intepration_time, SensorG):
    fGExp = SensorG * intepration_time * Kfactor
    rho_out = 0.9 * (-ln(fGExp)-ln(0.04)) + 0.5
    if rho_out > 1:
        return "out"

    elif rho_out <= 0.5:
        return "in"

    else:
        return "other"


# =============================================================
# function: black_level_correction
#   subtracts the black level channel wise
# =============================================================
def black_level_correction(raw, black_level, white_level, clip_range):

    print("----------------------------------------------------")
    print("Running black level correction...")

    # make float32 in case if it was not
    black_level = np.float32(black_level)
    white_level = np.float32(white_level)
    raw = np.float32(raw)

    # create new data so that original raw data do not change
    data = np.zeros(raw.shape)

    # bring data in range 0 to 1
    data[::2, ::2]   = (raw[::2, ::2] - black_level[0]) / (white_level[0] - black_level[0])
    data[::2, 1::2]  = (raw[::2, 1::2] - black_level[1]) / (white_level[1] - black_level[1])
    data[1::2, ::2]  = (raw[1::2, ::2] - black_level[2]) / (white_level[2] - black_level[2])
    data[1::2, 1::2] = (raw[1::2, 1::2]- black_level[3]) / (white_level[3] - black_level[3])

    # bring within the bit depth range
    data = data * clip_range[1]

    # clip within the range
    data = np.clip(data, clip_range[0], clip_range[1]) # upper level not necessary
    data = np.float32(data)

    return data


# =============================================================
# class: lens_shading_correction
#   Correct the lens shading / vignetting
# =============================================================
class lens_shading_correction:
    def __init__(self, data, name="lens_shading_correction"):
        # convert to float32 in case it was not
        self.data = np.float32(data)
        self.name = name

    def flat_field_compensation(self, dark_current_image, flat_field_image):
        # dark_current_image:
        #       is captured from the camera with cap on
        #       and fully dark condition, several images captured and
        #       temporally averaged
        # flat_field_image:
        #       is found by capturing an image of a flat field test chart
        #       with certain lighting condition
        # Note: flat_field_compensation is memory intensive procedure because
        #       both the dark_current_image and flat_field_image need to be
        #       saved in memory beforehand
        print("----------------------------------------------------")
        print("Running lens shading correction with flat field compensation...")

        # convert to float32 in case it was not
        dark_current_image = np.float32(dark_current_image)
        flat_field_image = np.float32(flat_field_image)
        temp = flat_field_image - dark_current_image
        return np.average(temp) * np.divide((self.data - dark_current_image), temp)

    def approximate_mathematical_compensation(self, params, clip_min=0, clip_max=65535):
        # parms:
        #       parameters of a parabolic model y = a*(x-b)^2 + c
        #       For example, params = [0.01759, -28.37, -13.36]
        # Note: approximate_mathematical_compensation require less memory
        print("----------------------------------------------------")
        print("Running lens shading correction with approximate mathematical compensation...")
        width, height = utility.helpers(self.data).get_width_height()

        center_pixel_pos = [height/2, width/2]
        max_distance = utility.distance_euclid(center_pixel_pos, [height, width])

        # allocate memory for output
        temp = np.empty((height, width), dtype=np.float32)

        for i in range(0, height):
            for j in range(0, width):
                distance = utility.distance_euclid(center_pixel_pos, [i, j]) / max_distance
                # parabolic model
                gain = params[0] * (distance - params[1])**2 + params[2]
                temp[i, j] = self.data[i, j] * gain

        temp = np.clip(temp, clip_min, clip_max)
        return temp

    def __str__(self):
        return "lens shading correction. There are two methods: " + \
                "\n (1) flat_field_compensation: requires dark_current_image and flat_field_image" + \
                "\n (2) approximate_mathematical_compensation:"


# =============================================================
# function: bad_pixel_correction
#   correct for the bad (dead, stuck, or hot) pixels
# =============================================================
def bad_pixel_correction(data, neighborhood_size):

    print("----------------------------------------------------")
    print("Running bad pixel correction...")

    if ((neighborhood_size % 2) == 0):
        print("neighborhood_size shoud be odd number, recommended value 3")
        return data

    # convert to float32 in case they were not
    # Being consistent in data format to be float32
    data = np.float32(data)

    # Separate out the quarter resolution images
    D = {} # Empty dictionary
    D[0] = data[::2, ::2]
    D[1] = data[::2, 1::2]
    D[2] = data[1::2, ::2]
    D[3] = data[1::2, 1::2]

    # number of pixels to be padded at the borders
    no_of_pixel_pad = math.floor(neighborhood_size / 2.)

    for idx in range(0, len(D)): # perform same operation for each quarter

        # display progress
        print("bad pixel correction: Quarter " + str(idx+1) + " of 4")

        img = D[idx]
        width, height = utility.helpers(img).get_width_height()

        # pad pixels at the borders
        img = np.pad(img, \
                     (no_of_pixel_pad, no_of_pixel_pad),\
                     'reflect') # reflect would not repeat the border value

        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # save the middle pixel value
                mid_pixel_val = img[i, j]

                # extract the neighborhood
                neighborhood = img[i - no_of_pixel_pad : i + no_of_pixel_pad+1,\
                                   j - no_of_pixel_pad : j + no_of_pixel_pad+1]

                # set the center pixels value same as the left pixel
                # Does not matter replace with right or left pixel
                # is used to replace the center pixels value
                neighborhood[no_of_pixel_pad, no_of_pixel_pad] = neighborhood[no_of_pixel_pad, no_of_pixel_pad-1]

                min_neighborhood = np.min(neighborhood)
                max_neighborhood = np.max(neighborhood)

                if (mid_pixel_val < min_neighborhood):
                    img[i,j] = min_neighborhood
                elif (mid_pixel_val > max_neighborhood):
                    img[i,j] = max_neighborhood
                else:
                    img[i,j] = mid_pixel_val

        # Put the corrected image to the dictionary
        D[idx] = img[no_of_pixel_pad : height + no_of_pixel_pad,\
                     no_of_pixel_pad : width + no_of_pixel_pad]

    # Regrouping the data
    data[::2, ::2]   = D[0]
    data[::2, 1::2]  = D[1]
    data[1::2, ::2]  = D[2]
    data[1::2, 1::2] = D[3]

    return data


# =============================================================
# class: lens_shading_correction
#   Correct the lens shading / vignetting
# =============================================================
class bayer_denoising:
    def __init__(self, data, name="bayer_denoising"):
        # convert to float32 in case it was not
        self.data = np.float32(data)
        self.name = name

    def utilize_hvs_behavior(self, bayer_pattern, initial_noise_level, hvs_min, hvs_max, threshold_red_blue, clip_range):
        # Objective: bayer denoising
        # Inputs:
        #   bayer_pattern:  rggb, gbrg, grbg, bggr
        #   initial_noise_level:
        # Output:
        #   denoised bayer raw output
        # Source: Based on paper titled "Noise Reduction for CFA Image Sensors
        #   Exploiting HVS Behaviour," by Angelo Bosco, Sebastiano Battiato,
        #   Arcangelo Bruna and Rosetta Rizzo
        #   Sensors 2009, 9, 1692-1713; doi:10.3390/s90301692

        print("----------------------------------------------------")
        print("Running bayer denoising utilizing hvs behavior...")

        # copy the self.data to raw and we will only work on raw
        # to make sure no change happen to self.data
        raw = self.data
        raw = np.clip(raw, clip_range[0], clip_range[1])
        width, height = utility.helpers(raw).get_width_height()

        # First make the bayer_pattern rggb
        # The algorithm is written only for rggb pattern, thus convert all other
        # pattern to rggb. Furthermore, this shuffling does not affect the
        # algorithm output
        if (bayer_pattern != "rggb"):
            raw = utility.helpers(self.data).shuffle_bayer_pattern(bayer_pattern, "rggb")

        # fixed neighborhood_size
        neighborhood_size = 5 # we are keeping this fixed
                              # bigger size such as 9 can be declared
                              # however, the code need to be changed then

        # pad two pixels at the border
        no_of_pixel_pad = math.floor(neighborhood_size / 2)   # number of pixels to pad

        raw = np.pad(raw, \
                     (no_of_pixel_pad, no_of_pixel_pad),\
                     'reflect') # reflect would not repeat the border value

        # allocating space for denoised output
        denoised_out = np.empty((height, width), dtype=np.float32)

        texture_degree_debug = np.empty((height, width), dtype=np.float32)
        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # center pixel
                center_pixel = raw[i, j]

                # signal analyzer block
                half_max = clip_range[1] / 2
                if (center_pixel <= half_max):
                    hvs_weight = -(((hvs_max - hvs_min) * center_pixel) / half_max) + hvs_max
                else:
                    hvs_weight = (((center_pixel - clip_range[1]) * (hvs_max - hvs_min))/(clip_range[1] - half_max)) + hvs_max

                # noise level estimator previous value
                if (j < no_of_pixel_pad+2):
                    noise_level_previous_red   = initial_noise_level
                    noise_level_previous_blue  = initial_noise_level
                    noise_level_previous_green = initial_noise_level
                else:
                    noise_level_previous_green = noise_level_current_green
                    if ((i % 2) == 0): # red
                        noise_level_previous_red = noise_level_current_red
                    elif ((i % 2) != 0): # blue
                        noise_level_previous_blue = noise_level_current_blue

                # Processings depending on Green or Red/Blue
                # Red
                if (((i % 2) == 0) and ((j % 2) == 0)):
                    # get neighborhood
                    neighborhood = [raw[i-2, j-2], raw[i-2, j], raw[i-2, j+2],\
                                    raw[i, j-2], raw[i, j+2],\
                                    raw[i+2, j-2], raw[i+2, j], raw[i+2, j+2]]

                    # absolute difference from the center pixel
                    d =  np.abs(neighborhood - center_pixel)

                    # maximum and minimum difference
                    d_max = np.max(d)
                    d_min = np.min(d)

                    # calculate texture_threshold
                    texture_threshold = hvs_weight + noise_level_previous_red

                    # texture degree analyzer
                    if (d_max <= threshold_red_blue):
                        texture_degree = 1.
                    elif ((d_max > threshold_red_blue) and (d_max <= texture_threshold)):
                        texture_degree = -((d_max - threshold_red_blue) / (texture_threshold - threshold_red_blue)) + 1.
                    elif (d_max > texture_threshold):
                        texture_degree = 0.

                    # noise level estimator update
                    noise_level_current_red = texture_degree * d_max + (1 - texture_degree) * noise_level_previous_red

                # Blue
                elif (((i % 2) != 0) and ((j % 2) != 0)):

                    # get neighborhood
                    neighborhood = [raw[i-2, j-2], raw[i-2, j], raw[i-2, j+2],\
                                    raw[i, j-2], raw[i, j+2],\
                                    raw[i+2, j-2], raw[i+2, j], raw[i+2, j+2]]

                    # absolute difference from the center pixel
                    d =  np.abs(neighborhood - center_pixel)

                    # maximum and minimum difference
                    d_max = np.max(d)
                    d_min = np.min(d)

                    # calculate texture_threshold
                    texture_threshold = hvs_weight + noise_level_previous_blue

                    # texture degree analyzer
                    if (d_max <= threshold_red_blue):
                        texture_degree = 1.
                    elif ((d_max > threshold_red_blue) and (d_max <= texture_threshold)):
                        texture_degree = -((d_max - threshold_red_blue) / (texture_threshold - threshold_red_blue)) + 1.
                    elif (d_max > texture_threshold):
                        texture_degree = 0.

                    # noise level estimator update
                    noise_level_current_blue = texture_degree * d_max + (1 - texture_degree) * noise_level_previous_blue

                # Green
                elif ((((i % 2) == 0) and ((j % 2) != 0)) or (((i % 2) != 0) and ((j % 2) == 0))):

                    neighborhood = [raw[i-2, j-2], raw[i-2, j], raw[i-2, j+2],\
                                    raw[i-1, j-1], raw[i-1, j+1],\
                                    raw[i, j-2], raw[i, j+2],\
                                    raw[i+1, j-1], raw[i+1, j+1],\
                                    raw[i+2, j-2], raw[i+2, j], raw[i+2, j+2]]

                    # difference from the center pixel
                    d = np.abs(neighborhood - center_pixel)

                    # maximum and minimum difference
                    d_max = np.max(d)
                    d_min = np.min(d)

                    # calculate texture_threshold
                    texture_threshold = hvs_weight + noise_level_previous_green

                    # texture degree analyzer
                    if (d_max == 0):
                        texture_degree = 1
                    elif ((d_max > 0) and (d_max <= texture_threshold)):
                        texture_degree = -(d_max / texture_threshold) + 1.
                    elif (d_max > texture_threshold):
                        texture_degree = 0

                    # noise level estimator update
                    noise_level_current_green = texture_degree * d_max + (1 - texture_degree) * noise_level_previous_green

                # similarity threshold calculation
                if (texture_degree == 1):
                    threshold_low = threshold_high = d_max
                elif (texture_degree == 0):
                    threshold_low = d_min
                    threshold_high = (d_max + d_min) / 2
                elif ((texture_degree > 0) and (texture_degree < 1)):
                    threshold_high = (d_max + ((d_max + d_min) / 2)) / 2
                    threshold_low = (d_min + threshold_high) / 2

                # weight computation
                weight = np.empty(np.size(d), dtype=np.float32)
                pf = 0.
                for w_i in range(0, np.size(d)):
                    if (d[w_i] <= threshold_low):
                        weight[w_i] = 1.
                    elif (d[w_i] > threshold_high):
                        weight[w_i] = 0.
                    elif ((d[w_i] > threshold_low) and (d[w_i] < threshold_high)):
                        weight[w_i] = 1. + ((d[w_i] - threshold_low) / (threshold_low - threshold_high))

                    pf += weight[w_i] * neighborhood[w_i] + (1. - weight[w_i]) * center_pixel

                denoised_out[i - no_of_pixel_pad, j-no_of_pixel_pad] = pf / np.size(d)
                # texture_degree_debug is a debug output
                texture_degree_debug[i - no_of_pixel_pad, j-no_of_pixel_pad] = texture_degree

        if (bayer_pattern != "rggb"):
            denoised_out = utility.shuffle_bayer_pattern(denoised_out, "rggb", bayer_pattern)

        return np.clip(denoised_out, clip_range[0], clip_range[1]), texture_degree_debug

    def __str__(self):
        return self.name


# =============================================================
# class: color_correction
#   Correct the color in linaer domain
# =============================================================
class color_correction:
    def __init__(self, data, color_matrix, color_space="srgb", illuminant="d65", name="color correction", clip_range=[0, 65535]):
        # Inputs:
        #   data:   linear rgb image before nonlinearity/gamma
        #   xyz2cam: 3x3 matrix found from the camera metedata, specifically
        #            color matrix 2 from the metadata
        #   color_space: output color space
        #   illuminance: the illuminant of the lighting condition
        #   name: name of the class
        self.data = np.float32(data)
        self.xyz2cam = np.float32(color_matrix)
        self.color_space = color_space
        self.illuminant = illuminant
        self.name = name
        self.clip_range = clip_range

    def get_rgb2xyz(self):
        # Objective: get the rgb2xyz matrix dependin on the output color space
        #            and the illuminant
        # Source: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
        if (self.color_space == "srgb"):
            if (self.illuminant == "d65"):
                return [[.4124564,  .3575761,  .1804375],\
                        [.2126729,  .7151522,  .0721750],\
                        [.0193339,  .1191920,  .9503041]]
            elif (self.illuminant == "d50"):
                return [[.4360747,  .3850649,  .1430804],\
                        [.2225045,  .7168786,  .0606169],\
                        [.0139322,  .0971045,  .7141733]]
            else:
                print("for now, color_space must be d65 or d50")
                return

        elif (self.color_space == "adobe-rgb-1998"):
            if (self.illuminant == "d65"):
                return [[.5767309,  .1855540,  .1881852],\
                        [.2973769,  .6273491,  .0752741],\
                        [.0270343,  .0706872,  .9911085]]
            elif (self.illuminant == "d50"):
                return [[.6097559,  .2052401,  .1492240],\
                        [.3111242,  .6256560,  .0632197],\
                        [.0194811,  .0608902,  .7448387]]
            else:
                print("for now, illuminant must be d65 or d50")
                return
        else:
            print("for now, color_space must be srgb or adobe-rgb-1998")
            return

    def calculate_cam2rgb(self):
        # Objective: Calculates the color correction matrix

        # matric multiplication
        rgb2cam = np.dot(self.xyz2cam, self.get_rgb2xyz())

        # make sum of each row to be 1.0, necessary to preserve white balance
        # basically divice each value by its row wise sum
        rgb2cam = np.divide(rgb2cam, np.reshape(np.sum(rgb2cam, 1), [3, 1]))

        # - inverse the matrix to get cam2rgb.
        # - cam2rgb should also have the characteristic that sum of each row
        # equal to 1.0 to preserve white balance
        # - check if rgb2cam is invertible by checking the condition of
        # rgb2cam. If rgb2cam is singular it will give a warning and
        # return an identiry matrix
        if (np.linalg.cond(rgb2cam) < (1 / sys.float_info.epsilon)):
            return np.linalg.inv(rgb2cam) # this is cam2rgb / color correction matrix
        else:
            print("Warning! matrix not invertible.")
            return np.identity(3, dtype=np.float32)

    def apply_cmatrix(self):
        # Objective: Apply the color correction matrix (cam2rgb)

        print("----------------------------------------------------")
        print("running color correction...")

        # check if data is 3 dimensional
        if (np.ndim(self.data) != 3):
            print("data need to be three dimensional")
            return

        # get the color correction matrix
        cam2rgb = self.calculate_cam2rgb()

        # get width and height
        width, height = utility.helpers(self.data).get_width_height()

        # apply the matrix
        R = self.data[:, :, 0]
        G = self.data[:, :, 1]
        B = self.data[:, :, 2]

        color_corrected = np.empty((height, width, 3), dtype=np.float32)
        color_corrected[:, :, 0] = R * cam2rgb[0, 0] + G * cam2rgb[0, 1] + B * cam2rgb[0, 2]
        color_corrected[:, :, 1] = R * cam2rgb[1, 0] + G * cam2rgb[1, 1] + B * cam2rgb[1, 2]
        color_corrected[:, :, 2] = R * cam2rgb[2, 0] + G * cam2rgb[2, 1] + B * cam2rgb[2, 2]

        return np.clip(color_corrected, self.clip_range[0], self.clip_range[1])

    def __str__(self):
        return self.name


# =============================================================
# class: chromatic_aberration_correction
#   removes artifacts similar to result from chromatic
#   aberration
# =============================================================
class chromatic_aberration_correction:
    def __init__(self, data, name="chromatic aberration correction"):
        self.data = np.float32(data)
        self.name = name

    def purple_fringe_removal(self, nsr_threshold, cr_threshold, clip_range=[0, 65535]):
        # --------------------------------------------------------------
        # nsr_threshold: near saturated region threshold (in percentage)
        # cr_threshold: candidate region threshold
        # --------------------------------------------------------------

        width, height = utility.helpers(self.data).get_width_height()

        r = self.data[:, :, 0]
        g = self.data[:, :, 1]
        b = self.data[:, :, 2]

        ## Detection of purple fringe
        # near saturated region detection
        nsr_threshold = clip_range[1] * nsr_threshold / 100
        temp = (r + g + b) / 3
        temp = np.asarray(temp)
        mask = temp > nsr_threshold
        nsr = np.zeros((height, width), dtype=np.int)
        nsr[mask] = 1

        # candidate region detection
        temp = r - b
        temp1 = b - g
        temp = np.asarray(temp)
        temp1 = np.asarray(temp1)
        mask = (temp < cr_threshold) & (temp1 > cr_threshold)
        cr = np.zeros((height, width), dtype=np.int)
        cr[mask] = 1

        # quantization
        qr = utility.helpers(r).nonuniform_quantization()
        qg = utility.helpers(g).nonuniform_quantization()
        qb = utility.helpers(b).nonuniform_quantization()

        g_qr = utility.edge_detection(qr).sobel(5, "gradient_magnitude")
        g_qg = utility.edge_detection(qg).sobel(5, "gradient_magnitude")
        g_qb = utility.edge_detection(qb).sobel(5, "gradient_magnitude")

        g_qr = np.asarray(g_qr)
        g_qg = np.asarray(g_qg)
        g_qb = np.asarray(g_qb)

        # bgm: binary gradient magnitude
        bgm = np.zeros((height, width), dtype=np.float32)
        mask = (g_qr != 0) | (g_qg != 0) | (g_qb != 0)
        bgm[mask] = 1

        fringe_map = np.multiply(np.multiply(nsr, cr), bgm)
        fring_map = np.asarray(fringe_map)
        mask = (fringe_map == 1)

        r1 = r
        g1 = g
        b1 = b
        r1[mask] = g1[mask] = b1[mask] = (r[mask] + g[mask] + b[mask]) / 3.

        output = np.empty(np.shape(self.data), dtype=np.float32)
        output[:, :, 0] = r1
        output[:, :, 1] = g1
        output[:, :, 2] = b1

        return np.float32(output)


    def __str__(self):
        return self.name



# =============================================================
# function: gamma_correction
#   perform gamma correction to make the image more suitable for human eyes
#   data: image data
#   gamma: the gamma value depends on different displayer
#   bitdepth: the image bit depth
#   colour space: the colour space of the image
# =============================================================
def gamma_correction(data, gamma=1.0, bitdepth=8, colour_space="grey"):

    #if the image is grey-scale
    if colour_space == "grey":

        temp = np.float32(data)

        #scale the image intensity to (0,1)
        temp = temp/(2**bitdepth - 1)

        #invert gamma
        invgamma = 1 / gamma

        #perform gamma correction
        im = (data**invgamma) * (2**bitdepth - 1)

    #if the image is a rgb image
    elif colour_space == "rgb":

        #convert the image to grey-scale
        temp = utility.color_conversion(data).rgb2gray

        #scale the image intensity to (0,1)
        temp = temp/(2**bitdepth - 1)

        #invert gamma
        invgamma = 1 / gamma

        #perform gamma correction
        im = (data**invgamma) * (2**bitdepth - 1)

    else:
        print("unsupported colour space")

    return im


# =============================================================
# class: distortion_correction
#   correct the distortion
# =============================================================
class distortion_correction:
    def __init__(self, data, name="distortion correction"):
        self.data = np.float32(data)
        self.name = name


    def empirical_correction(self, correction_type="pincushion-1", strength=0.1, zoom_type="crop", clip_range=[0, 65535]):
        #------------------------------------------------------
        # Objective:
        #   correct geometric distortion with the assumption that the distortion
        #   is symmetric and the center is at the center of of the image
        # Input:
        #   correction_type:    which type of correction needed to be carried
        #                       out, choose one the four:
        #                       pincushion-1, pincushion-2, barrel-1, barrel-2
        #                       1 and 2 are difference between the power
        #                       over the radius
        #
        #   strength:           should be equal or greater than 0.
        #                       0 means no correction will be done.
        #                       if negative value were applied correction_type
        #                       will be reversed. Thus,>=0 value expected.
        #
        #   zoom_type:          either "fit" or "crop"
        #                       fit will return image with full content
        #                       in the whole area
        #                       crop will return image will 0 values outsise
        #                       the border
        #
        #   clip_range:         to clip the final image within the range
        #------------------------------------------------------

        if (strength < 0):
            print("Warning! strength should be equal of greater than 0.")
            return self.data

        print("----------------------------------------------------")
        print("Running distortion correction by empirical method...")

        # get half_width and half_height, assume this is the center
        width, height = utility.helpers(self.data).get_width_height()
        half_width = width / 2
        half_height = height / 2

        # create a meshgrid of points
        xi, yi = np.meshgrid(np.linspace(-half_width, half_width, width),\
                             np.linspace(-half_height, half_height, height))

        # cartesian to polar coordinate
        r = np.sqrt(xi**2 + yi**2)
        theta = np.arctan2(yi, xi)

        # maximum radius
        R = math.sqrt(width**2 + height**2)

        # make r within range 0~1
        r = r / R

        # apply the radius to the desired transformation
        s = utility.special_function(r).distortion_function(correction_type, strength)

        # select a scaling_parameter based on zoon_type and k value
        if ((correction_type=="barrel-1") or (correction_type=="barrel-2")):
            if (zoom_type == "fit"):
                scaling_parameter = r[0, 0] / s[0, 0]
            elif (zoom_type == "crop"):
                scaling_parameter = 1. / (1. + strength * (np.min([half_width, half_height])/R)**2)
        elif ((correction_type=="pincushion-1") or (correction_type=="pincushion-2")):
            if (zoom_type == "fit"):
                scaling_parameter = 1. / (1. + strength * (np.min([half_width, half_height])/R)**2)
            elif (zoom_type == "crop"):
                scaling_parameter = r[0, 0] / s[0, 0]

        # multiply by scaling_parameter and un-normalize
        s = s * scaling_parameter * R

        # convert back to cartesian coordinate and add back the center coordinate
        xt = np.multiply(s, np.cos(theta))
        yt = np.multiply(s, np.sin(theta))

        # interpolation
        if np.ndim(self.data == 3):

            output = np.empty(np.shape(self.data), dtype=np.float32)

            output[:, :, 0] = utility.helpers(self.data[:, :, 0]).bilinear_interpolation(xt + half_width, yt + half_height)
            output[:, :, 1] = utility.helpers(self.data[:, :, 1]).bilinear_interpolation(xt + half_width, yt + half_height)
            output[:, :, 2] = utility.helpers(self.data[:, :, 2]).bilinear_interpolation(xt + half_width, yt + half_height)

        elif np.ndim(self.data == 2):

            output = utility.helpers(self.data).bilinear_interpolation(xt + half_width, yt + half_height)

        return np.clip(output, clip_range[0], clip_range[1])


    def __str__(self):
        return self.name

        

