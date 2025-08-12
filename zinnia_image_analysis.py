from shiny import App, ui, render
import os
import cv2
import copy
import numpy as np
import pandas as pd
import math
import keras
from PIL import Image, ImageOps  # Install pillow instead of PIL
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from itertools import compress
from scipy.stats import mode
from scipy.spatial import cKDTree, distance_matrix
from scipy.spatial.distance import cdist

"""Automatically detect color cards.

Algorithm written by mtwatso2-eng (github). Updated and implemented into PlantCV by Haley Schuhl.
"""
import os
import cv2
import math
import numpy as np

def _is_square(contour, min_size):
    """Determine if a contour is square or not.

    Parameters
    ----------
    contour : list
        OpenCV contour.

    Returns
    -------
    bool
        True if the contour is square, False otherwise.
    """
    return (cv2.contourArea(contour) > min_size and
            max(cv2.minAreaRect(contour)[1]) / min(cv2.minAreaRect(contour)[1]) < 1.27 and
            (cv2.contourArea(contour) / np.prod(cv2.minAreaRect(contour)[1])) > 0.8)


def _get_contour_sizes(contours):
    """Get the shape and size of all contours.

    Parameters
    ----------
    contours : list
        List of OpenCV contours.

    Returns
    -------
    list
        Contour areas, widths, and heights.
    """
    # Initialize chip shape lists
    marea, mwidth, mheight = [], [], []
    # Loop over our contours and size data about them
    for cnt in contours:
        marea.append(cv2.contourArea(cnt))
        _, wh, _ = cv2.minAreaRect(cnt)  # Rotated rectangle
        mwidth.append(wh[0])
        mheight.append(wh[1])
    return marea, mwidth, mheight


def _draw_color_chips(rgb_img, new_centers, radius):
    """Create labeled mask and debug image of color chips.

    Parameters
    ----------
    rgb_img : numpy.ndarray
        Input RGB image data containing a color card.
    new_centers : numpy.array
        Chip centers after transformation.
    radius : int
        Radius of circles to draw on the color chips.

    Returns
    -------
    list
        Labeled mask and debug image.
    """
    # Create blank img for drawing the labeled color card mask
    labeled_mask = np.zeros(rgb_img.shape[0:2])
    debug_img = np.copy(rgb_img)

    # Loop over the new chip centers and draw them on the RGB image and labeled mask
    for i, pt in enumerate(new_centers):
        cv2.circle(labeled_mask, new_centers[i], radius, (i + 1) * 10, -1)
        cv2.circle(debug_img, new_centers[i], radius, (255, 255, 0), -1)
    return labeled_mask, debug_img


def detect_color_card(rgb_img, label=None, **kwargs):

    # Get keyword arguments and set defaults if not set
    min_size = kwargs.get("min_size", 1000)  # Minimum size for _is_square chip filtering
    radius = kwargs.get("radius", 20)  # Radius of circles to draw on the color chips
    adaptive_method = kwargs.get("adaptive_method", 1)  # cv2.adaptiveThreshold method
    block_size = kwargs.get("block_size", 51)  # cv2.adaptiveThreshold block size

    # Hard code since we don't currently support other color cards
    nrows = 6
    ncols = 4

    # Convert to grayscale, threshold, and findContours
    imgray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(imgray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(gaussian, 255, adaptive_method,
                                   cv2.THRESH_BINARY_INV, block_size, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours, keep only square-shaped ones
    filtered_contours = [contour for contour in contours if _is_square(contour, min_size)]
    # Calculate median area of square contours
    target_square_area = np.median([cv2.contourArea(cnt) for cnt in filtered_contours])
    # Filter contours again, keep only those within 20% of median area
    filtered_contours = [contour for contour in filtered_contours if
                         (0.8 < (cv2.contourArea(contour) / target_square_area) < 1.2)]

    # Initialize chip shape lists
    marea, mwidth, mheight = _get_contour_sizes(filtered_contours)

    # Create dataframe for easy summary stats
    chip_size = np.median(marea)
    chip_height = np.median(mheight)
    chip_width = np.median(mwidth)

    # Concatenate all contours into one array and find the minimum area rectangle
    rect = np.concatenate([[np.array(cv2.minAreaRect(i)[0]).astype(int)] for i in filtered_contours])
    rect = cv2.minAreaRect(rect)
    # Get the corners of the rectangle
    corners = np.array(np.intp(cv2.boxPoints(rect)))
    # Determine which corner most likely contains the white chip
    white_index = np.argmin([np.mean(math.dist(rgb_img[corner[1], corner[0], :], (255, 255, 255))) for corner in corners])
    corners = corners[np.argsort([math.dist(corner, corners[white_index]) for corner in corners])[[0, 1, 3, 2]]]
    # Increment amount is arbitrary, cell distances rescaled during perspective transform
    increment = 100
    centers = [[int(0 + i * increment), int(0 + j * increment)] for j in range(nrows) for i in range(ncols)]

    # Find the minimum area rectangle of the chip centers
    new_rect = cv2.minAreaRect(np.array(centers))
    # Get the corners of the rectangle
    box_points = cv2.boxPoints(new_rect).astype("float32")
    # Calculate the perspective transform matrix from the minimum area rectangle
    m_transform = cv2.getPerspectiveTransform(box_points, corners.astype("float32"))
    # Transform the chip centers using the perspective transform matrix
    new_centers = cv2.transform(np.array([centers]), m_transform)[0][:, 0:2]

    # Create labeled mask and debug image of color chips
    labeled_mask, debug_img = _draw_color_chips(rgb_img, new_centers, radius)

    return chip_size, labeled_mask

def apply_transformation_matrix(source_img, target_img, transformation_matrix):

    # split transformation_matrix
    red, green, blue, _, _, _, _, _, _ = np.split(transformation_matrix, 9, 1)

    source_dtype = source_img.dtype
    # normalization value as max number if the type is unsigned int
    max_val = 1.0
    if source_dtype.kind == 'u':
        max_val = np.iinfo(source_dtype).max
    # convert img to float to avoid integer overflow, normalize between 0-1
    source_flt = source_img.astype(np.float64)/max_val
    # find linear, square, and cubic values of source_img color channels
    source_b, source_g, source_r = cv2.split(source_flt)
    source_b2 = np.square(source_b)
    source_b3 = np.power(source_b, 3)
    source_g2 = np.square(source_g)
    source_g3 = np.power(source_g, 3)
    source_r2 = np.square(source_r)
    source_r3 = np.power(source_r, 3)

    # apply linear model to source color channels
    b = 0 + source_r * blue[0] + source_g * blue[1] + source_b * blue[2] + source_r2 * blue[3] + source_g2 * blue[
        4] + source_b2 * blue[5] + source_r3 * blue[6] + source_g3 * blue[7] + source_b3 * blue[8]
    g = 0 + source_r * green[0] + source_g * green[1] + source_b * green[2] + source_r2 * green[3] + source_g2 * green[
        4] + source_b2 * green[5] + source_r3 * green[6] + source_g3 * green[7] + source_b3 * green[8]
    r = 0 + source_r * red[0] + source_g * red[1] + source_b * red[2] + source_r2 * red[3] + source_g2 * red[
        4] + source_b2 * red[5] + source_r3 * red[6] + source_g3 * red[7] + source_b3 * red[8]

    # merge corrected color channels onto source_image
    bgr = [b, g, r]
    corrected_img = cv2.merge(bgr)

    # return values of the image to the original range
    corrected_img = max_val*np.clip(corrected_img, 0, 1)
    # cast back to original dtype (if uint the value defaults to the closest smaller integer)
    corrected_img = corrected_img.astype(source_dtype)

    # return corrected_img
    return corrected_img

def calc_transformation_matrix(matrix_m, matrix_b):
    t_r, t_r2, t_r3, t_g, t_g2, t_g3, t_b, t_b2, t_b3 = np.split(matrix_b, 9, 1)

    # multiply each 22x1 matrix from target color space by matrix_m
    red = np.matmul(matrix_m, t_r)
    green = np.matmul(matrix_m, t_g)
    blue = np.matmul(matrix_m, t_b)

    red2 = np.matmul(matrix_m, t_r2)
    green2 = np.matmul(matrix_m, t_g2)
    blue2 = np.matmul(matrix_m, t_b2)

    red3 = np.matmul(matrix_m, t_r3)
    green3 = np.matmul(matrix_m, t_g3)
    blue3 = np.matmul(matrix_m, t_b3)

    # concatenate each product column into 9X9 transformation matrix
    transformation_matrix = np.concatenate((red, green, blue, red2, green2, blue2, red3, green3, blue3), 1)

    # find determinant of transformation matrix
    t_det = np.linalg.det(transformation_matrix)

    return 1-t_det, transformation_matrix


def get_color_matrix(rgb_img, mask):
    img_dtype = rgb_img.dtype
    # normalization value as max number if the type is unsigned int
    max_val = 1.0
    if img_dtype.kind == 'u':
        max_val = np.iinfo(img_dtype).max

    # convert to float and normalize to work with values between 0-1
    rgb_img = rgb_img.astype(np.float64)/max_val

    # create empty color_matrix
    color_matrix = np.zeros((len(np.unique(mask))-1, 4))

    # create headers
    headers = ["chip_number", "r_avg", "g_avg", "b_avg"]

    # declare row_counter variable and initialize to 0
    row_counter = 0

    # for each unique color chip calculate each average RGB value
    for i in np.unique(mask):
        if i != 0:
            chip = rgb_img[np.where(mask == i)]
            color_matrix[row_counter][0] = i
            color_matrix[row_counter][1] = np.mean(chip[:, 2])
            color_matrix[row_counter][2] = np.mean(chip[:, 1])
            color_matrix[row_counter][3] = np.mean(chip[:, 0])
            row_counter += 1

    return headers, color_matrix

def get_matrix_m(target_matrix, source_matrix):
    # if the number of chips in source_img match the number of chips in target_matrix
    if np.shape(target_matrix) == np.shape(source_matrix):
        _, t_r, t_g, t_b = np.split(target_matrix, 4, 1)
        _, s_r, s_g, s_b = np.split(source_matrix, 4, 1)
    else:
        combined_matrix = np.zeros((np.ma.size(source_matrix, 0), 7))
        row_count = 0
        for r in range(0, np.ma.size(target_matrix, 0)):
            for i in range(0, np.ma.size(source_matrix, 0)):
                if target_matrix[r][0] == source_matrix[i][0]:
                    combined_matrix[row_count][0] = target_matrix[r][0]
                    combined_matrix[row_count][1] = target_matrix[r][1]
                    combined_matrix[row_count][2] = target_matrix[r][2]
                    combined_matrix[row_count][3] = target_matrix[r][3]
                    combined_matrix[row_count][4] = source_matrix[i][1]
                    combined_matrix[row_count][5] = source_matrix[i][2]
                    combined_matrix[row_count][6] = source_matrix[i][3]
                    row_count += 1
        _, t_r, t_g, t_b, s_r, s_g, s_b = np.split(combined_matrix, 7, 1)
    t_r2 = np.square(t_r)
    t_r3 = np.power(t_r, 3)
    t_g2 = np.square(t_g)
    t_g3 = np.power(t_g, 3)
    t_b2 = np.square(t_b)
    t_b3 = np.power(t_b, 3)
    s_r2 = np.square(s_r)
    s_r3 = np.power(s_r, 3)
    s_g2 = np.square(s_g)
    s_g3 = np.power(s_g, 3)
    s_b2 = np.square(s_b)
    s_b3 = np.power(s_b, 3)

    # create matrix_a
    matrix_a = np.concatenate((s_r, s_g, s_b, s_r2, s_g2, s_b2, s_r3, s_g3, s_b3), 1)
    # create matrix_m
    matrix_m = np.linalg.solve(np.matmul(matrix_a.T, matrix_a), matrix_a.T)
    # create matrix_b
    matrix_b = np.concatenate((t_r, t_r2, t_r3, t_g, t_g2, t_g3, t_b, t_b2, t_b3), 1)
    return matrix_a, matrix_m, matrix_b

def correct_color(target_img, target_mask, source_img, source_mask):

    # get color matrices for target and source images
    _, target_matrix = get_color_matrix(target_img, target_mask)
    _, source_matrix = get_color_matrix(source_img, source_mask)

    # get matrix_m
    _, matrix_m, matrix_b = get_matrix_m(target_matrix=target_matrix, source_matrix=source_matrix)
    # calculate transformation_matrix and save
    _, transformation_matrix = calc_transformation_matrix(matrix_m, matrix_b)

    # apply transformation
    corrected_img = apply_transformation_matrix(source_img, target_img, transformation_matrix)

    return corrected_img

def cropSquareFromContour(c, img, rotateByHeuristic = False):

    rect = cv2.minAreaRect(c)
    rect = list(rect)
    orientation = np.argmax(rect[1])
    rect[1] = (max(rect[1]), max(rect[1]))

    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width = int(max(rect[1]))
    height = int(max(rect[1]))

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    if(rotateByHeuristic and orientation == 0):
      warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped

def meanColor(img, excluded_color):
  img = img.reshape(np.prod(img.shape[0:2]), 3)
  img = img[np.any(img != excluded_color, axis=1)]
  return img.mean(0).astype("uint8")

referenceImage = cv2.imread("referenceImage.JPG")
_, referenceColorCardMask = detect_color_card(rgb_img=referenceImage)

def colorCorrect(image, referenceImage = referenceImage):
  _, cc_mask = detect_color_card(rgb_img=image)
  corrected_img = correct_color(target_img=referenceImage, target_mask=referenceColorCardMask, source_img=image, source_mask=cc_mask)
  return corrected_img

def getColorCardChipLength(image):
  chipSize, _ = detect_color_card(rgb_img=image)
  return np.sqrt(chipSize)

flowerClassifierModel = keras.layers.TFSMLayer("flower-not flower classifier/model.savedmodel", call_endpoint = 'serving_default')
flowerClassifierClassNames = open("flower-not flower classifier/labels.txt", "r").read().splitlines()
def classifyFlower(cv2Image):
  img = cv2Image
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = Image.fromarray(img)
  width_height_tuple = (224, 224)
  width, height = img.size
  target_width, target_height = width_height_tuple
  crop_height = (width * target_height) // target_width
  crop_width = (height * target_width) // target_height
  crop_height = min(height, crop_height)
  crop_width = min(width, crop_width)
  crop_box_hstart = (height - crop_height) // 2
  crop_box_wstart = (width - crop_width) // 2
  crop_box_wend = crop_box_wstart + crop_width
  crop_box_hend = crop_box_hstart + crop_height
  crop_box = [
      crop_box_wstart, crop_box_hstart, crop_box_wend,
      crop_box_hend
  ]
  img = img.resize(width_height_tuple, Image.NEAREST, box=crop_box)
  my_image = np.asarray(img)
  my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
  my_image = my_image / 255.
  prediction = flowerClassifierModel(my_image)
  prediction = prediction[list(prediction.keys())[0]].numpy()[0]
  return(prediction)

stemClassifierModel = keras.layers.TFSMLayer("stem-not stem classifier/model.savedmodel", call_endpoint = 'serving_default')
stemClassifierClassNames = open("stem-not stem classifier/labels.txt", "r").read().splitlines()

def classifyStem(cv2Image):
  img = cv2Image
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = Image.fromarray(img)
  width_height_tuple = (224, 224)
  width, height = img.size
  target_width, target_height = width_height_tuple
  crop_height = (width * target_height) // target_width
  crop_width = (height * target_width) // target_height
  crop_height = min(height, crop_height)
  crop_width = min(width, crop_width)
  crop_box_hstart = (height - crop_height) // 2
  crop_box_wstart = (width - crop_width) // 2
  crop_box_wend = crop_box_wstart + crop_width
  crop_box_hend = crop_box_hstart + crop_height
  crop_box = [
      crop_box_wstart, crop_box_hstart, crop_box_wend,
      crop_box_hend
  ]
  img = img.resize(width_height_tuple, Image.NEAREST, box=crop_box)
  my_image = np.asarray(img)
  my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
  my_image = my_image / 255.
  prediction = stemClassifierModel(my_image)
  prediction = prediction[list(prediction.keys())[0]].numpy()[0]
  return(prediction)

def thresholdImage(image, scaling_factor = 2, color = [120, 50], adjustment = 0.5):
  thres = cv2.resize(image, None, fx = 1/scaling_factor, fy = 1/scaling_factor)
  thres = cv2.cvtColor(thres, cv2.COLOR_BGR2Lab)
  blueDistances = cdist(thres[:,:,[1,2]].reshape(thres.shape[0] * thres.shape[1], 2), [color]) # [120, 50]
  thres = blueDistances.reshape(thres.shape[0], thres.shape[1], 1).astype("uint8")
  thres = cv2.adaptiveThreshold(thres, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5001, 2)
  thres = cv2.resize(thres, None, fx = scaling_factor, fy = scaling_factor)
  return thres

def countImage(image, imageName=None):
    phenotypes = pd.DataFrame(columns=['diameter', 'meanColorR', 'meanColorG', 'meanColorB', 'stemLength'])
    colorCardChipLengthCm = 1.81
    colorCardChipLengthPixels = getColorCardChipLength(image)
    image = colorCorrect(image)
    thres = thresholdImage(image)

    kernel = np.ones((9, 9), np.uint8)
    thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flowerContours = []
    stemContours = []
    for i, contour in enumerate(contours):
            rect = cv2.minAreaRect(contour)
            rectArea = rect[1][0] * rect[1][1]
            if(rectArea > 100 and rectArea < (thres.shape[0] * thres.shape[1] * 0.1)):
                canvas = np.zeros(thres.shape).astype("uint8")
                canvas = cv2.drawContours(canvas, [contour], -1, 255, cv2.FILLED)
                mask = cropSquareFromContour(contour, canvas, rotateByHeuristic = True)
                segmentation = cropSquareFromContour(contour, image, rotateByHeuristic = True)
                segmentation[(255 - mask).astype("bool"), :] = [0,0,0]
                # segmentation[(255 - cropSquareFromContour(contour, thres)).astype("bool"), :] = [0,0,0]
                # segmentation = cv2.resize(segmentation, (224,224))
                confidence = classifyFlower(cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB))
                if confidence[flowerClassifierClassNames.index('flower')] > 0.4:
                  flowerContours.append(contour)
                  flowerDiameter = np.max(rect[1]) * colorCardChipLengthCm / colorCardChipLengthPixels
                  R, G, B = meanColor(cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB), [0,0,0])
                  phenotypes.loc[len(phenotypes.index)] = [flowerDiameter, R, G, B, None]
                else:
                  confidence = classifyStem(cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB))
                  if confidence[stemClassifierClassNames.index('stem')] > 0.4:
                      stemContours.append(contour)
                      stemLenghth = (cv2.arcLength(contour, closed = True) / 2) * colorCardChipLengthCm / colorCardChipLengthPixels
                      phenotypes.loc[len(phenotypes.index)] = [None, None, None, None, stemLenghth]

    for flowerContour in flowerContours:
      image = cv2.drawContours(image, [flowerContour], -1, (0,255,0), 20)
    for stemContour in stemContours:
      image = cv2.drawContours(image, [stemContour], -1, (0,255,255), 20)
    
    # Add image name column to phenotypes dataframe if provided
    if not phenotypes.empty and imageName is not None:
        phenotypes.insert(0, 'imageName', imageName)
    
    return image, phenotypes
