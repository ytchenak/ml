import cv2
import glob

ORIGINAL_IMG_PATH = "test_images\\urban100\\"
DOWNSCALED_IMG_PATH = "test_images\\urban100x4\\"
for img_path in glob.glob(f"{ORIGINAL_IMG_PATH}*.png"):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    lr_img = cv2.resize(img, (w//4, h//4), interpolation=cv2.INTER_AREA)
    out_path = DOWNSCALED_IMG_PATH + img_path.split("\\")[-1].split(".")[0] + ".png"
    cv2.imwrite(out_path, lr_img)
