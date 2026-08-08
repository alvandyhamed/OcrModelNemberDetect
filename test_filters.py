import cv2
import numpy as np

def remove_grid(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Binarize (Otsu)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Grid dot filter: Remove small dot-grid noise using morphological opening
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel_small)
    
    # 3. Connect text stroke pixels
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(cleaned, kernel_connect, iterations=1)
    
    # Invert back to black-on-white text
    output = cv2.bitwise_not(dilated)
    return output

if __name__ == "__main__":
    import glob
    for f in sorted(glob.glob("*.png")):
        res = remove_grid(f)
        out_name = "clean_" + f
        cv2.imwrite(out_name, res)
        print(f"Saved {out_name}")
