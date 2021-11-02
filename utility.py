import numpy as np
import math
import png

# =============================================================
# function: imsave
#   save image in image formats
#   data:   is the image data
#   output_dtype: output data type
#   input_dtype: input data type
#   is_scale: is scaling needed to go from input data type to output data type
# =============================================================
def imsave(data, output_name, output_dtype="uint8", input_dtype="uint8", is_scale=False):

    dtype_dictionary = {"uint8" : np.uint8(data), "uint16" : np.uint16(data),\
                        "uint32" : np.uint32(data), "uint64" : np.uint64(data),\
                        "int8" : np.int8(data), "int16" : np.int16(data),\
                        "int32" : np.int32(data), "int64" : np.int64(data),\
                        "float16" : np.float16(data), "float32" : np.float32(data),\
                        "float64" : np.float64(data)}

    min_val_dictionary = {"uint8" : 0, "uint16" : 0,\
                          "uint32" : 0, "uint64" : 0,\
                          "int8" : -128, "int16" : -32768,\
                          "int32" : -2147483648, "int64" : -9223372036854775808}

    max_val_dictionary = {"uint8" : 255, "uint16" : 65535,\
                          "uint32" : 4294967295, "uint64" : 18446744073709551615,\
                          "int8" : 127, "int16" : 32767,\
                          "int32" : 2147483647, "int64" : 9223372036854775807}

    # scale the data in case scaling is necessary to go from input_dtype
    # to output_dtype
    if (is_scale):

        # convert data into float32
        data = np.float32(data)

        # Get minimum and maximum value of the input and output data types
        in_min  = min_val_dictionary[input_dtype]
        in_max  = max_val_dictionary[input_dtype]
        out_min = min_val_dictionary[output_dtype]
        out_max = max_val_dictionary[output_dtype]

        # clip the input data in the input_dtype range
        data = np.clip(data, in_min, in_max)

        # scale the data
        data = out_min + (data - in_min) * (out_max - out_min) / (in_max - in_min)

        # clip scaled data in output_dtype range
        data = np.clip(data, out_min, out_max)

    # convert the data into the output_dtype
    data = dtype_dictionary[output_dtype]

    # output image type: raw, png, jpeg
    output_file_type = output_name[-3:]

    # save files depending on output_file_type
    if (output_file_type == "raw"):
        pass # will be added later
        return

    elif (output_file_type == "png"):

        # png will only save uint8 or uint16
        if ((output_dtype == "uint16") or (output_dtype == "uint8")):
            if (output_dtype == "uint16"):
                output_bitdepth = 16
            elif (output_dtype == "uint8"):
                output_bitdepth = 8

            pass
        else:
            print("For png output, output_dtype must be uint8 or uint16")
            return

        with open(output_name, "wb") as f:
            # rgb image
            if (np.ndim(data) == 3):
                # create the png writer
                writer = png.Writer(width=data.shape[1], height=data.shape[0],\
                                    bitdepth = output_bitdepth,\
                                    greyscale = False)
                # convert data to the python lists expected by the png Writer
                data2list = data.reshape(-1, data.shape[1]*data.shape[2]).tolist()
                # write in the file
                writer.write(f, data2list)

            # greyscale image
            elif (np.ndim(data) == 2):
                # create the png writer
                writer = png.Writer(width=data.shape[1], height=data.shape[0],\
                                    bitdepth = output_bitdepth,\
                                    greyscale = True)
                # convert data to the python lists expected by the png Writer
                data2list = data.tolist()
                # write in the file
                writer.write(f, data2list)

    elif (output_file_type == "jpg"):
        pass # will be added later
        return

    else:
        print("output_name should contain extensions of .raw, .png, or .jpg")
        return

# =============================================================
# function: read_raw_image
#   read raw image and return an array of image pixel values
#   path: the image file path
#   bitdepth: the bit depth of the image
#   LSB_supp: if the raw image is not 16-bit or 8-bit and the pixel value is supplemented with 0s in Least Significant Bits, this parameter should be True
# =============================================================
def read_raw_image(path, bitdepth=16, LSB_supp=False):
    #read raw8 image 
    if bitdepth == 8:
        im = np.fromfile(path, dtype="uint8", sep="")
    
    #read raw10, raw12, raw14 or raw16 images
    elif bitdepth > 8 and bitdepth <= 16:
        im = np.fromfile(path, dtype="uint16", sep="")
        if LSB_supp == True:
            im = im/math.pow(2,(16-bitdepth))

    #if the bit depth value is none of above
    else:
        print("Unsupported bit depth value for a raw image file")
        return
    
    return im

# =============================================================
# function: read_raw_image
#   read raw image and return an array of image pixel values
#   path: the image file path
#   bitdepth: the bit depth of the image
# =============================================================
def read_pgm_image(path, bitdepth=16):
    #read 16-bit pgm image
    if bitdepth == 16:
        #open the file and read the first line of headers to determine the pgm standard
        pgmf = open(path, 'rb')
        line_1 = pgmf.readline()

        #if the image is a P5 standard pgm image
        if line_1[:2] == b'P5':
            #read the second line of headers and get the height and width of the image
            line_2 = pgmf.readline()
            (width, height) = (line_2.decode()).split()
            (width, height) = (int(width), int(height))

            for n in range(0,6):
                pgmf.readline()

            #read the third line of headers and get the maximum pixel value
            line_3 = pgmf.readline()
            depth = int(line_3.decode())

            #return error if the colour depth is too large
            assert depth <= 65535

            #read the image and store it into a numpy array
            im = []
            for i in range(height*width):
                high_bits = int.from_bytes(pgmf.read(1), byteorder='little')
                im.append(255*high_bits+int.from_bytes(pgmf.read(1), byteorder='little'))
            pgmf.close()
            im = np.array(im)

        #if the image is a P2 standard image
        elif line_1.strip() == 'P2':
            im = []
            #store the image and discard headers
            lines = pgmf.readlines()
            for i in lines:
                im.extend([int(c) for c in i.split()])
            im = np.array(im[3:])
        
        else:
            print("unsupported pgm type")
            return

    #read 8-bit pgm image
    elif bitdepth == 8:
        #open the file and read the first line of headers to determine the pgm standard
        pgmf = open(path, 'rb')
        line_1 = pgmf.readline()

        #if the image is a P5 standard pgm image
        if line_1[:2] == b'P5':
            #read the second line of headers and get the height and width of the image
            line_2 = pgmf.readline()
            (width, height) = (line_2.decode()).split()
            (width, height) = (int(width), int(height))

            #read the third line of headers and get the maximum pixel value
            line_3 = pgmf.readline()

            while line_3[:1] == b'#':
                line_3 = pgmf.readline()

            depth = int(line_3.decode())

            #return error if the colour depth is too large
            assert depth <= 255

            #read the image and store it into a numpy array
            im = []
            for i in range(height*width):
                im.append(int.from_bytes(pgmf.read(1), byteorder='little'))
            pgmf.close()
            im = np.array(im)
        
        #if the image is a P2 standard image
        elif line_1.strip() == 'P2':
            im = []
            #store the image and discard headers
            lines = pgmf.readlines()
            for i in lines:
                im.extend([int(c) for c in i.split()])
            im = np.array(im[3:])

        else:
            print("unsupported pgm type")
            return

    else:
        print("unsupported bit depth for a pgm image")
        return

    return im

# =============================================================
# class: helpers
#   a class of useful helper functions
# =============================================================
class helpers:
    def __init__(self, data=None, name="helper"):
        self.data = np.float32(data)
        self.name = name
        
    def get_width_height(self):
        #------------------------------------------------------
        # returns width, height
        # We assume data be in height x width x number of channel x frames format
        #------------------------------------------------------
        if (np.ndim(self.data) > 1):
            size = np.shape(self.data)
            width = size[1]
            height = size[0]
            return width, height
        else:
            print("Error! data dimension must be 2 or greater")

    def scale(self, depth_in, depth_out):
        #------------------------------------------------------
        #scale the image to a desired colour depth
        #------------------------------------------------------
        data = self.data*math.pow(2,depth_out)/math.pow(2,depth_in)
        return data

    def bayer_channel_separation(self, pattern):
        #------------------------------------------------------
        # function: bayer_channel_separation
        #   Objective: Outputs four channels of the bayer pattern
        #   Input:
        #       data:   the bayer data
        #       pattern:    rggb, grbg, gbrg, or bggr
        #   Output:
        #       R, G1, G2, B (Quarter resolution images)
        #------------------------------------------------------
        if (pattern == "rggb"):
            R = self.data[::2, ::2]
            G1 = self.data[::2, 1::2]
            G2 = self.data[1::2, ::2]
            B = self.data[1::2, 1::2]
        elif (pattern == "grbg"):
            G1 = self.data[::2, ::2]
            R = self.data[::2, 1::2]
            B = self.data[1::2, ::2]
            G2 = self.data[1::2, 1::2]
        elif (pattern == "gbrg"):
            G1 = self.data[::2, ::2]
            B = self.data[::2, 1::2]
            R = self.data[1::2, ::2]
            G2 = self.data[1::2, 1::2]
        elif (pattern == "bggr"):
            B = self.data[::2, ::2]
            G1 = self.data[::2, 1::2]
            G2 = self.data[1::2, ::2]
            R = self.data[1::2, 1::2]
        else:
            print("pattern must be one of these: rggb, grbg, gbrg, bggr")
            return

        return R, G1, G2, B


    def bayer_channel_integration(self, R, G1, G2, B, pattern):
        #------------------------------------------------------
        # function: bayer_channel_integration
        #   Objective: combine data into a raw according to pattern
        #   Input:
        #       R, G1, G2, B:   the four separate channels (Quarter resolution)
        #       pattern:    rggb, grbg, gbrg, or bggr
        #   Output:
        #       data (Full resolution image)
        #------------------------------------------------------
        size = np.shape(R)
        data = np.empty((size[0]*2, size[1]*2), dtype=np.float32)
        if (pattern == "rggb"):
            data[::2, ::2] = R
            data[::2, 1::2] = G1
            data[1::2, ::2] = G2
            data[1::2, 1::2] = B
        elif (pattern == "grbg"):
            data[::2, ::2] = G1
            data[::2, 1::2] = R
            data[1::2, ::2] = B
            data[1::2, 1::2] = G2
        elif (pattern == "gbrg"):
            data[::2, ::2] = G1
            data[::2, 1::2] = B
            data[1::2, ::2] = R
            data[1::2, 1::2] = G2
        elif (pattern == "bggr"):
            data[::2, ::2] = B
            data[::2, 1::2] = G1
            data[1::2, ::2] = G2
            data[1::2, 1::2] = R
        else:
            print("pattern must be one of these: rggb, grbg, gbrg, bggr")
            return

        return data


    def shuffle_bayer_pattern(self, input_pattern, output_pattern):
        #------------------------------------------------------
        # function: shuffle_bayer_pattern
        #   convert from one bayer pattern to another
        #------------------------------------------------------

        # Get separate channels
        R, G1, G2, B = self.bayer_channel_separation(input_pattern)

        # return integrated data
        return self.bayer_channel_integration(R, G1, G2, B, output_pattern)

# =============================================================
# class: color_conversion
#   color conversion from one color space to another
# =============================================================
class color_conversion:
    def __init__(self, data, name="color conversion"):
        self.data = np.float32(data)
        self.name = name

    def rgb2gray(self):
        return 0.299 * self.data[:, :, 0] +\
               0.587 * self.data[:, :, 1] +\
               0.114 * self.data[:, :, 2]

    def rgb2ycc(self, rule="bt601"):

        # map to select kr and kb
        kr_kb_dict = {"bt601" : [0.299, 0.114],\
                      "bt709" : [0.2126, 0.0722],\
                      "bt2020" : [0.2627, 0.0593]}

        kr = kr_kb_dict[rule][0]
        kb = kr_kb_dict[rule][1]
        kg = 1 - (kr + kb)

        output = np.empty(np.shape(self.data), dtype=np.float32)
        output[:, :, 0] = kr * self.data[:, :, 0] + \
                          kg * self.data[:, :, 1] + \
                          kb * self.data[:, :, 2]
        output[:, :, 1] = 0.5 * ((self.data[:, :, 2] - output[:, :, 0]) / (1 - kb))
        output[:, :, 2] = 0.5 * ((self.data[:, :, 0] - output[:, :, 0]) / (1 - kr))

        return output

    def ycc2rgb(self, rule="bt601"):

        # map to select kr and kb
        kr_kb_dict = {"bt601" : [0.299, 0.114],\
                      "bt709" : [0.2126, 0.0722],\
                      "bt2020" : [0.2627, 0.0593]}

        kr = kr_kb_dict[rule][0]
        kb = kr_kb_dict[rule][1]
        kg = 1 - (kr + kb)

        output = np.empty(np.shape(self.data), dtype=np.float32)
        output[:, :, 0] = 2. * self.data[:, :, 2] * (1 - kr) + self.data[:, :, 0]
        output[:, :, 2] = 2. * self.data[:, :, 1] * (1 - kb) + self.data[:, :, 0]
        output[:, :, 1] = (self.data[:, :, 0] - kr * output[:, :, 0] - kb * output[:, :, 2]) / kg

        return output

    def rgb2xyz(self, color_space="srgb", clip_range=[0, 65535]):
        # input rgb in range clip_range
        # output xyz is in range 0 to 1

        if (color_space == "srgb"):

            # degamma / linearization
            data = helpers(self.data).degamma_srgb(clip_range)
            data = np.float32(data)
            data = np.divide(data, clip_range[1])

            # matrix multiplication`
            output = np.empty(np.shape(self.data), dtype=np.float32)
            output[:, :, 0] = data[:, :, 0] * 0.4124 + data[:, :, 1] * 0.3576 + data[:, :, 2] * 0.1805
            output[:, :, 1] = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
            output[:, :, 2] = data[:, :, 0] * 0.0193 + data[:, :, 1] * 0.1192 + data[:, :, 2] * 0.9505

        elif (color_space == "adobe-rgb-1998"):

            # degamma / linearization
            data = helpers(self.data).degamma_adobe_rgb_1998(clip_range)
            data = np.float32(data)
            data = np.divide(data, clip_range[1])

            # matrix multiplication
            output = np.empty(np.shape(self.data), dtype=np.float32)
            output[:, :, 0] = data[:, :, 0] * 0.5767309 + data[:, :, 1] * 0.1855540 + data[:, :, 2] * 0.1881852
            output[:, :, 1] = data[:, :, 0] * 0.2973769 + data[:, :, 1] * 0.6273491 + data[:, :, 2] * 0.0752741
            output[:, :, 2] = data[:, :, 0] * 0.0270343 + data[:, :, 1] * 0.0706872 + data[:, :, 2] * 0.9911085

        elif (color_space == "linear"):

            # matrix multiplication`
            output = np.empty(np.shape(self.data), dtype=np.float32)
            data = np.float32(self.data)
            data = np.divide(data, clip_range[1])
            output[:, :, 0] = data[:, :, 0] * 0.4124 + data[:, :, 1] * 0.3576 + data[:, :, 2] * 0.1805
            output[:, :, 1] = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
            output[:, :, 2] = data[:, :, 0] * 0.0193 + data[:, :, 1] * 0.1192 + data[:, :, 2] * 0.9505

        else:
            print("Warning! color_space must be srgb or adobe-rgb-1998.")
            return

        return output


    def xyz2rgb(self, color_space="srgb", clip_range=[0, 65535]):
        # input xyz is in range 0 to 1
        # output rgb in clip_range

        # allocate space for output
        output = np.empty(np.shape(self.data), dtype=np.float32)

        if (color_space == "srgb"):

            # matrix multiplication
            output[:, :, 0] = self.data[:, :, 0] *  3.2406 + self.data[:, :, 1] * -1.5372 + self.data[:, :, 2] * -0.4986
            output[:, :, 1] = self.data[:, :, 0] * -0.9689 + self.data[:, :, 1] *  1.8758 + self.data[:, :, 2] *  0.0415
            output[:, :, 2] = self.data[:, :, 0] *  0.0557 + self.data[:, :, 1] * -0.2040 + self.data[:, :, 2] *  1.0570

            # gamma to retain nonlinearity
            output = helpers(output * clip_range[1]).gamma_srgb(clip_range)


        elif (color_space == "adobe-rgb-1998"):

            # matrix multiplication
            output[:, :, 0] = self.data[:, :, 0] *  2.0413690 + self.data[:, :, 1] * -0.5649464 + self.data[:, :, 2] * -0.3446944
            output[:, :, 1] = self.data[:, :, 0] * -0.9692660 + self.data[:, :, 1] *  1.8760108 + self.data[:, :, 2] *  0.0415560
            output[:, :, 2] = self.data[:, :, 0] *  0.0134474 + self.data[:, :, 1] * -0.1183897 + self.data[:, :, 2] *  1.0154096

            # gamma to retain nonlinearity
            output = helpers(output * clip_range[1]).gamma_adobe_rgb_1998(clip_range)


        elif (color_space == "linear"):

            # matrix multiplication
            output[:, :, 0] = self.data[:, :, 0] *  3.2406 + self.data[:, :, 1] * -1.5372 + self.data[:, :, 2] * -0.4986
            output[:, :, 1] = self.data[:, :, 0] * -0.9689 + self.data[:, :, 1] *  1.8758 + self.data[:, :, 2] *  0.0415
            output[:, :, 2] = self.data[:, :, 0] *  0.0557 + self.data[:, :, 1] * -0.2040 + self.data[:, :, 2] *  1.0570

            # gamma to retain nonlinearity
            output = output * clip_range[1]

        else:
            print("Warning! color_space must be srgb or adobe-rgb-1998.")
            return

        return output


    def xyz2lab(self, cie_version="1931", illuminant="d65"):

        xyz_reference = helpers().get_xyz_reference(cie_version, illuminant)

        data = self.data
        data[:, :, 0] = data[:, :, 0] / xyz_reference[0]
        data[:, :, 1] = data[:, :, 1] / xyz_reference[1]
        data[:, :, 2] = data[:, :, 2] / xyz_reference[2]

        data = np.asarray(data)

        # if data[x, y, c] > 0.008856, data[x, y, c] = data[x, y, c] ^ (1/3)
        # else, data[x, y, c] = 7.787 * data[x, y, c] + 16/116
        mask = data > 0.008856
        data[mask] **= 1./3.
        data[np.invert(mask)] *= 7.787
        data[np.invert(mask)] += 16./116.

        data = np.float32(data)
        output = np.empty(np.shape(self.data), dtype=np.float32)
        output[:, :, 0] = 116. * data[:, :, 1] - 16.
        output[:, :, 1] = 500. * (data[:, :, 0] - data[:, :, 1])
        output[:, :, 2] = 200. * (data[:, :, 1] - data[:, :, 2])

        return output


    def lab2xyz(self, cie_version="1931", illuminant="d65"):

        output = np.empty(np.shape(self.data), dtype=np.float32)

        output[:, :, 1] = (self.data[:, :, 0] + 16.) / 116.
        output[:, :, 0] = (self.data[:, :, 1] / 500.) + output[:, :, 1]
        output[:, :, 2] = output[:, :, 1] - (self.data[:, :, 2] / 200.)

        # if output[x, y, c] > 0.008856, output[x, y, c] ^ 3
        # else, output[x, y, c] = ( output[x, y, c] - 16/116 ) / 7.787
        output = np.asarray(output)
        mask = output > 0.008856
        output[mask] **= 3.
        output[np.invert(mask)] -= 16/116
        output[np.invert(mask)] /= 7.787

        xyz_reference = helpers().get_xyz_reference(cie_version, illuminant)

        output = np.float32(output)
        output[:, :, 0] = output[:, :, 0] * xyz_reference[0]
        output[:, :, 1] = output[:, :, 1] * xyz_reference[1]
        output[:, :, 2] = output[:, :, 2] * xyz_reference[2]

        return output

    def lab2lch(self):

        output = np.empty(np.shape(self.data), dtype=np.float32)

        output[:, :, 0] = self.data[:, :, 0] # L transfers directly
        output[:, :, 1] = np.power(np.power(self.data[:, :, 1], 2) + np.power(self.data[:, :, 2], 2), 0.5)
        output[:, :, 2] = np.arctan2(self.data[:, :, 2], self.data[:, :, 1]) * 180 / np.pi

        return output

    def lch2lab(self):

        output = np.empty(np.shape(self.data), dtype=np.float32)

        output[:, :, 0] = self.data[:, :, 0] # L transfers directly
        output[:, :, 1] = np.multiply(np.cos(self.data[:, :, 2] * np.pi / 180), self.data[:, :, 1])
        output[:, :, 2] = np.multiply(np.sin(self.data[:, :, 2] * np.pi / 180), self.data[:, :, 1])

        return output

    def rgb2yuv(self):

        output = np.empty(np.shape(self.data), dtype=np.float32)

        output[:, :, 0] = 0.298 * self.data[:, :, 0] + \
                          0.612 * self.data[:, :, 1] + \
                          0.117 * self.data[:, :, 2]
        output[:, :, 1] = -0.168 * self.data[:, :, 0] - \
                          0.330 * self.data[:, :, 1] + \
                          0.498 * self.data[:, :, 2] + \
                          128
        output[:, :, 2] = 0.449 * self.data[:, :, 0] - \
                          0.435 * self.data[:, :, 1] - \
                          0.083 * self.data[:, :, 2] + \
                          128
        
        return output

    def __str__(self):
        return self.name