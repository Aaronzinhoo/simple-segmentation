import cv2
import pytesseract # in env_color
from pathlib import Path

from remove_background import blackout_background, extract_contour, get_largest_bbox

def is_text_image(image, text_threshold=15):
    res = pytesseract.image_to_string(image,lang='eng+jpn')
    if len(res.replace('\n','').replace(' ', '')) > text_threshold:
        return True
    return False

def filter_and_clean_directory(input_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True,parents=True)
    count=200
    for index, image_path in enumerate(Path(input_dir).iterdir()):
        image = cv2.imread(str(image_path))
        try:
            if not is_text_image(image):
                try:
                    image = blackout_background(image)
                    image = extract_contour(image)
                    image = get_largest_bbox(image)
                except Exception as e:  # may catch cv2 assertion if image not correct type
                    print(e, image_path.name, "could not clean image")
                    continue
                else:
                    cv2.imwrite(str(output_dir / image_path.name), image)

        except Exception as e:
            print(e, image_path, "could not find image")
            continue
        if index % count == 0:
            print("completed {} images".format(index))
