import shutil
import os
import cv2
import numpy as np
import dicom
from dicom import dataelem
import pickle

have_PIL = True
try:
    import PIL.Image
    import PIL.ImageOps as ops
except ImportError:
    have_PIL = False

have_numpy = True
try:
    import numpy as np
except ImportError:
    have_numpy = False



def image_computing_edges(filename):
    filename = './images/' + filename
    img = cv2.imread(filename, 0)
    edges = cv2.Canny(img, 50, 25)

    cv2.imwrite(filename[:-4] + '_cv.png', edges)



def image_computing_watershed(filename, k=30, dist=0.1):
    #filename = '/home/hugo/PycharmProjects/IPaaS/images/' + filename
    print(filename)
    #img = cv2.imread(filename, 0)
    gray = cv2.imread(filename, 0)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((k, k), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)


    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, dist * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    #####attention here
    height, width = gray.shape
    img = np.zeros((height,width,3), np.uint8)
    cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB, img)

    #invert img
    #img = cv2.bitwise_not(img)

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    nf = filename[:-4] + '_cv.png'
    cv2.imwrite(nf, img)

    return nf


def get_LUT_value(data, window, level):
    """Apply the RGB Look-Up Table for the given
       data and window/level value."""
    if not have_numpy:
        raise ImportError("Numpy is not available."
                          "See http://numpy.scipy.org/"
                          "to download and install")

    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) /
                         (window - 1) + 0.5) * (255 - 0)])


def get_PIL_image(dataset):
    """Get Image object from Python Imaging Library(PIL)"""
    if not have_PIL:
        raise ImportError("Python Imaging Library is not available. "
                          "See http://www.pythonware.com/products/pil/ "
                          "to download and install")

    if ('PixelData' not in dataset):
        raise TypeError("Cannot show image -- DICOM dataset does not have "
                        "pixel data")
    # can only apply LUT if these window info exists
    if ('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):
        bits = dataset.BitsAllocated
        samples = dataset.SamplesPerPixel
        if bits == 8 and samples == 1:
            mode = "L"
        elif bits == 8 and samples == 3:
            mode = "RGB"
        elif bits == 16:
            # not sure about this -- PIL source says is 'experimental'
            # and no documentation. Also, should bytes swap depending
            # on endian of file and system??
            mode = "I;16"
        else:
            raise TypeError("Don't know PIL mode for %d BitsAllocated "
                            "and %d SamplesPerPixel" % (bits, samples))

        # PIL size = (width, height)
        size = (dataset.Columns, dataset.Rows)

        # Recommended to specify all details
        # by http://www.pythonware.com/library/pil/handbook/image.htm
        im = PIL.Image.frombuffer(mode, size, dataset.PixelData,
                                  "raw", mode, 0, 1)

    else:
        image = get_LUT_value(dataset.pixel_array, dataset.WindowWidth,
                              dataset.WindowCenter)
        # Convert mode to L since LUT has only 256 values:
        #   http://www.pythonware.com/library/pil/handbook/image.htm
        im = PIL.Image.fromarray(image).convert('L')

    return im


def show_PIL(dataset):
    """Display an image using the Python Imaging Library (PIL)"""
    im = get_PIL_image(dataset)
    im.show()

def save_PIL(filename, dataset):
    nf = filename[:-8]+'.png'
    im = get_PIL_image(dataset)

    im = im.convert(mode='I')
    im.save(nf)
    return nf


def save_segmentation(dcm_filename, seg_img_filename):
    seg_dcm_fl = dcm_filename[:-8] + '_seg.dcm'

    img = cv2.imread(seg_img_filename)
    i = pickle.dumps(img)

    ds = dicom.read_file(dcm_filename)

    de = dataelem.DataElement(0x00690069, 'UN', i)
    ds.add(de)
    dicom.write_file(seg_dcm_fl, ds)

    return seg_dcm_fl

def read_seg(dcm_filename):
    ds = dicom.read_file(dcm_filename)
    img = ds[0x00690069].value
    i = pickle.loads(img)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    newf = dcm_filename+ '_convert.png'
    cv2.imwrite(newf, i)

def read_dicom(f):
    return dicom.read_file(f, force=True)

#f = "./images/mammo_new.dcm"
#ds = dicom.read_file(f, force=True)

#show_PIL(ds)
#nf=save_PIL(f, ds)
#nf='./images/mammo_pil_tmp.png'
#print(nf)
#nnf = image_computing_watershed(nf)
#seg_dcm = save_segmentation(f, nnf)
#read_seg(seg_dcm)
