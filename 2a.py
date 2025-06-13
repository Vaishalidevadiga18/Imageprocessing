import cv2
import numpy as np

def transform_image(image, tx=0, ty=0, angle=0, scale_x=1.0, scale_y=1.0):
    rows, cols = image.shape[:2]
    trans_mat = np.float32([[1, 0, tx], [0, 1, ty]])
    rot_center = (cols // 2, rows // 2)
    rot_mat = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
    
    translated = cv2.warpAffine(image, trans_mat, (cols, rows))
    rotated = cv2.warpAffine(image, rot_mat, (cols, rows))
    scaled = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    
    return translated, rotated, scaled

def main():
    path = input("Enter image path: ")
    img = cv2.imread(path)
    if img is None:
        print("Error loading image"); return

    tx = int(input("Translation x: "))
    ty = int(input("Translation y: "))
    angle = float(input("Rotation angle: "))
    sx = float(input("Scale x: "))
    sy = float(input("Scale y: "))

    t_img, r_img, s_img = transform_image(img, tx, ty, angle, sx, sy)

    cv2.imshow('Original', img)
    cv2.imshow('Translated', t_img)
    cv2.imshow('Rotated', r_img)
    cv2.imshow('Scaled', s_img)

    if input("Save images? (yes/no): ").strip().lower() == 'yes':
        cv2.imwrite('translated.jpg', t_img)
        cv2.imwrite('rotated.jpg', r_img)
        cv2.imwrite('scaled.jpg', s_img)
        print("Images saved.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
