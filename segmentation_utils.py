import cv2
import numpy as np
from pathlib import Path


def findSignificantContour(edgeImg):
    image, contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
    # From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])

    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour


def filterOutSaltPepperNoise(edgeImg):
    # Get rid of salt & pepper noise.
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        # get those pixels that gets zeroed out
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 3)


def segment_object(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    blurred_float = blurred.astype(np.float32) / 255.0

    # get edges
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    edges_8u = np.asarray(edges, np.uint8)
    filterOutSaltPepperNoise(edges_8u)

    # get contour
    contour = findSignificantContour(edges_8u)

    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    # mark inital mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    # run grabcut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv2.grabCut(image, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # create mask again
    mask2 = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')

    contour2 = findSignificantContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv2.fillPoly(mask3, [contour2], 255)

    # blended alpha cut-out
    mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
    alpha = mask4.astype(float) * 1.1  # making blend stronger
    alpha[mask3 > 0] = 255
    alpha[alpha > 255] = 255
    alpha = alpha.astype(float)

    foreground = np.copy(image).astype(float)
    foreground[mask4 == 0] = 0

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha / 255.0
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # bounding rect for foreground
    mask4_gray = cv2.cvtColor(mask4, cv2.COLOR_RGB2GRAY)
    x, y, w, h = cv2.boundingRect(mask4_gray)
    return foreground[y:y+h, x:x+w, :]

def filter_and_clean_directory(input_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True,parents=True)
    count=200
    for index, image_path in enumerate(Path(input_dir).iterdir()):
        orig_image = cv2.imread(str(image_path))
        if orig_image is None:
            print('Failed to read image')
            continue
        try:
            image = segment_object(orig_image)
            cv2.imwrite(str(output_dir / image_path.name), image)
        except cv2.error as e:
            print(f'Failed to write segmented image {image_path.name}\n{e}')
        if index % count == 0:
            print("completed {} images".format(index))
