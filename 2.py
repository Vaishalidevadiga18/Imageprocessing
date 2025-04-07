import cv2
import numpy as np

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print("Error: Could not load image")
        exit()
    return image

def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

def rotate_image(image, angle, scale=1.0):
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def scale_image(image, scale_x, scale_y):
    scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    return scaled_image

def main():
    image_path = input("Enter the image path: ")
    image = load_image(image_path)

    tx = int(input("Enter translation in x direction: "))
    ty = int(input("Enter translation in y direction: "))
    angle = float(input("Enter rotation angle: "))
    scale_x = float(input("Enter scaling factor for x: "))
    scale_y = float(input("Enter scaling factor for y: "))

    translated = translate_image(image, tx, ty)
    rotated = rotate_image(image, angle, 1.0)
    scaled = scale_image(image, scale_x, scale_y)

    cv2.imshow('Original Image', image)
    cv2.imshow('Translated Image', translated)
    cv2.imshow('Rotated Image', rotated)
    cv2.imshow('Scaled Image', scaled)

    save_option = input('Do you want to save the transformed images? (yes/no): ').strip().lower()
    if save_option == 'yes':
        cv2.imwrite('translated_image.jpg', translated)
        cv2.imwrite('rotated_image.jpg', rotated)
        cv2.imwrite('scaled_image.jpg', scaled)
        print('Images saved successfully!')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
