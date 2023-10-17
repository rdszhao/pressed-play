import numpy as np
import requests
import cv2

def download_image_bytes(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_bytes = response.content
        return image_bytes
    except Exception as e:
        print(f"error downloading image: {e}")
        return None

def decode_image(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def encode_image(image):
    try:
        _, encoded_image = cv2.imencode(".jpg", image)
        encoded_bytes = encoded_image.tobytes()
        return encoded_bytes
    except Exception as e:
        print(f"error encoding image: {e}")
        return None

def random_rotation(image):
    angle = np.random.uniform(-10, 10)
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows))

def random_horizontal_flip(image):
    return cv2.flip(image, 1)

def random_vertical_flip(image):
    return cv2.flip(image, 0)

def random_zoom_and_crop(image):
    zoom_factor = np.random.uniform(0.8, 1.2)
    zoomed_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    rows, cols, _ = zoomed_image.shape
    cropped_image = zoomed_image[rows // 8: -rows // 8, cols // 8: -cols // 8]
    return cv2.resize(cropped_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

def random_brightness_and_contrast(image):
    brightness_factor = np.random.uniform(0.7, 1.3)
    contrast_factor = np.random.uniform(0.7, 1.3)
    image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

def random_color_saturation_and_hue(image):
    saturation_factor = np.random.uniform(0.7, 1.3)
    hue_shift = np.random.uniform(-10, 10)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def random_gaussian_noise(image):
    noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def random_gaussian_blur(image):
    ksize = int(np.random.uniform(1, 5)) * 2 + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def random_sharpen(image):
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(image, -1, kernel)

distortion_functions = [
    random_rotation,
    random_horizontal_flip,
    random_vertical_flip,
    random_zoom_and_crop,
    random_brightness_and_contrast,
    random_color_saturation_and_hue,
    random_gaussian_noise,
    random_gaussian_blur,
    random_sharpen,
]

def apply_random_distortions(image):
    distorted_image = image.copy()
    transformation_mask = np.random.rand(len(distortion_functions)) > 0.5
    for distortion, apply_distortion in zip(distortion_functions, transformation_mask):
        if apply_distortion:
            distorted_image = distortion(distorted_image)
    return distorted_image

def distort_encode(image_bytes):
    decoded_image = decode_image(image_bytes)
    distorted_image = apply_random_distortions(decoded_image)
    encoded_image = encode_image(distorted_image)
    return encoded_image