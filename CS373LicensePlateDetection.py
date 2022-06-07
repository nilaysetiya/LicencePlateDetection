import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png
THRESHOLD = 150

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

def computeRGBTGreyScale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):

    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            r = pixel_array_r[i][j]
            g = pixel_array_g[i][j]
            b = pixel_array_b[i][j]
            greyscale_pixel_array[i][j] = round(0.299 * r + 0.587 * g + 0.114 * b)
    
    return greyscale_pixel_array

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    
    flow = pixel_array[0][0]
    fhigh = flow
    for i in range(image_height):
        for j in range(image_width):
            if (pixel_array[i][j] > fhigh):
                fhigh = pixel_array[i][j]
            
            if (pixel_array[i][j] < flow):
                flow = pixel_array[i][j]
    
    gmin = 0
    gmax = 255
    if (fhigh - flow) != 0:
        zero_div = 0
        divisor = (gmax - gmin) / (fhigh - flow)
    else:
        zero_div = 1
    
    for i in range(image_height):
        for j in range(image_width):
            if zero_div == 0:
                sout = (pixel_array[i][j] - flow) * divisor + gmin
                
                if (sout < gmin):
                    pixel_array[i][j] = round(gmin)
                elif (sout > gmax):
                    pixel_array[i][j] = round(gmax)
                else:
                    pixel_array[i][j] = round(sout)
            else:
                pixel_array[i][j] = round(0)
    
    return pixel_array

def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):

    mean_array = createInitializedGreyscalePixelArray(image_width, image_height)
    sd_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            if (i == 0) or (j == 0) or (i == image_height - 1) or (j == image_width - 1) or (i == 1) or (j == 1) or (i == image_height - 2) or (j == image_width - 2): # border
                mean_array[i][j] = 0.0
            else: # not border
                sum1 = pixel_array[i-2][j-2] + pixel_array[i-2][j-1] + pixel_array[i-2][j] + pixel_array[i-2][j+1] + pixel_array[i-2][j+2] 
                sum2 = pixel_array[i-1][j-2] + pixel_array[i-1][j-1] + pixel_array[i-1][j] + pixel_array[i-1][j+1] + pixel_array[i-1][j+2]
                sum3 = pixel_array[i][j-2] + pixel_array[i][j-1] + pixel_array[i][j] + pixel_array[i][j+1] + pixel_array[i][j+2]
                sum4 = pixel_array[i+1][j-2] + pixel_array[i+1][j-1] + pixel_array[i+1][j] + pixel_array[i+1][j+1] + pixel_array[i+1][j+2]
                sum5 = pixel_array[i+2][j-2] + pixel_array[i+2][j-1] + pixel_array[i+2][j] + pixel_array[i+2][j+1] + pixel_array[i+2][j+2]

                mean = (sum1 + sum2 + sum3 + sum4 + sum5) / 25.0
                if (mean < 0):
                    mean_array[i][j] = -1.0 * mean
                else:
                    mean_array[i][j] = mean
    
    for i in range(image_height):
        for j in range(image_width):
            if (i == 0) or (j == 0) or (i == image_height - 1) or (j == image_width - 1) or (i == 1) or (j == 1) or (i == image_height - 2) or (j == image_width - 2): # border
                sd_array[i][j] = 0.0
            else: # not border
                count = 0
                for k in range(5):
                    for l in range(5):
                        count = count + (pixel_array[i-2+k][j-2+l] - mean_array[i][j])**2
                
                sd_array[i][j] = math.sqrt(count / 25.0)
    
    return sd_array

def thresholdBinaryImage(pixel_array, image_width, image_height):

    threshold_array = createInitializedGreyscalePixelArray(image_width=image_width, image_height=image_height)

    for i in range(image_height):
        for j in range(image_width):
            if (pixel_array[i][j] < THRESHOLD):
                threshold_array[i][j] = 0
            else:
                threshold_array[i][j] = 255
    
    return threshold_array

def computeDilation(pixel_array, image_width, image_height):
    px_array = [[0 for i in range(image_width + 2)] for j in range(image_height + 2)]
    dilated_array = [[0 for i in range(image_width)] for j in range(image_height)]

    for i in range(1, image_height+1):
        for j in range(1, image_width+1):
            px_array[i][j] = pixel_array[i-1][j-1]
    
    for i in range(1, image_height+1):
        for j in range(1, image_width+1):
            fit = 0
            for k in range(i-1, i+2):
                for l in range(j-1, j+2):
                    if (px_array[k][l] != 0):
                        fit = 255
            dilated_array[i-1][j-1] = fit
    
    return dilated_array

def computeErosion(pixel_array, image_width, image_height):
    eroded_array = [[0 for i in range(image_width)] for j in range(image_height)]

    for i in range(1, image_height-1):
        for j in range(1, image_width-1):
            fit = 255
            for k in range(i-1, i+2):
                for l in range(j-1, j+2):
                    if (pixel_array[k][l] == 0):
                        fit = 0
            eroded_array[i][j] = fit
    
    return eroded_array

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    q = Queue()
    comp_array = [[0 for i in range(image_width)] for j in range(image_height)]
    current_label = 1
    d = {}
    for i in range(image_height):
        for j in range(image_width):
            if comp_array[i][j] == 0 and pixel_array[i][j] != 0:
                coordinate = [i, j]
                q.enqueue(coordinate)
                current_value = 0
                while not (q.isEmpty()):
                    coordinate = q.dequeue()
                    if comp_array[coordinate[0]][coordinate[1]] == current_label:
                        continue
                    comp_array[coordinate[0]][coordinate[1]] = current_label
                    current_value += 1
                    
                    if (coordinate[1] != 0):
                        if (pixel_array[coordinate[0]][coordinate[1]-1] != 0) and (comp_array[coordinate[0]][coordinate[1]-1] == 0):
                            q.enqueue([coordinate[0], coordinate[1]-1])
                    if (coordinate[1] != image_width-1):
                        if (pixel_array[coordinate[0]][coordinate[1]+1] != 0) and (comp_array[coordinate[0]][coordinate[1]+1] == 0):
                            q.enqueue([coordinate[0], coordinate[1]+1])
                    if (coordinate[0] != 0):
                        if (pixel_array[coordinate[0]-1][coordinate[1]] != 0) and (comp_array[coordinate[0]-1][coordinate[1]] == 0):
                            q.enqueue([coordinate[0]-1, coordinate[1]])
                    if (coordinate[0] != image_height-1):
                        if (pixel_array[coordinate[0]+1][coordinate[1]] != 0) and (comp_array[coordinate[0]+1][coordinate[1]] == 0):
                            q.enqueue([coordinate[0]+1, coordinate[1]])
                d[current_label] = current_value
                current_label += 1
    return (comp_array, d)

# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate6.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    
    # Convert to greyscale
    greyscale_array = computeRGBTGreyScale(pixel_array_r=px_array_r, pixel_array_g=px_array_g, pixel_array_b=px_array_b, image_width=image_width, image_height=image_height)

    # Quantize
    scale_array = scaleTo0And255AndQuantize(pixel_array=greyscale_array, image_width=image_width, image_height=image_height)

    # Contrast stretching - standard deviation with 5x5 pixel neighbourhood
    contrast_array = computeStandardDeviationImage5x5(pixel_array=scale_array, image_width=image_width, image_height=image_height)

    # Quantize
    scale_array1 = scaleTo0And255AndQuantize(pixel_array=contrast_array, image_width=image_width, image_height=image_height)

    # Thresholding for Segmentation
    binary_image_array = thresholdBinaryImage(pixel_array=scale_array1, image_width=image_width, image_height=image_height)

    # Morphological Operations
    morph_array = computeDilation(pixel_array=binary_image_array, image_width=image_width, image_height=image_height)
    for i in range(3):
        morph_array = computeDilation(pixel_array=morph_array, image_width=image_width, image_height=image_height)
    
    for i in range(4):
        morph_array = computeErosion(pixel_array=morph_array, image_width=image_width, image_height=image_height)
    
    # Connected Component Analysis
    (connected_array, label_dict) = computeConnectedComponentLabeling(pixel_array=morph_array, image_width=image_width, image_height=image_height)
    
    while(True):
        max_value = -1
        current_key = -1
        for key in label_dict:
            if label_dict[key] > max_value:
                max_value = label_dict[key]
                current_key = key
        
        min_x_found = False
        min_y_found = False
        min_x = -1
        max_x = -1
        min_y = -1
        max_y = -1
        for i in range(image_height):
            for j in range(image_width):
                if (connected_array[i][j] == current_key):
                    if (not min_x_found):
                        min_x_found = True
                        min_x = j
                    
                    if (not min_y_found):
                        min_y_found = True
                        min_y = i

                    if (j > max_x):
                        max_x = j
                    
                    if (i > max_y):
                        max_y = i
                    
                    if (j < min_x):
                        min_x = j

                    if (i < min_y):
                        min_y = i

        try:
            aspect_ratio = (float(max_x)-(float(min_x)))/(float(max_y)-float(min_y))
        except ZeroDivisionError as e:
            continue

        if (aspect_ratio > 1.5 and aspect_ratio < 5):
            break
        elif (len(label_dict) == 0):
            print("dict empty")
            break
        else:
            label_dict.pop(current_key)

    px_array = px_array_r

    bbox_min_x = min_x
    bbox_max_x = max_x
    bbox_min_y = min_y
    bbox_max_y = max_y


    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)



    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()