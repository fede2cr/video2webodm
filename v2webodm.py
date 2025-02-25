'''
It grabs a video file and transforms it into images with GPS metadata for WebODM
'''

import argparse
import cv2
import pysrt
from PIL import Image
import piexif
from skimage.metrics import structural_similarity as ssim
from skimage import io, img_as_float
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Transform drone videos into images for WebODM")

    # Adding arguments
    parser.add_argument('-i', '--input', type=str, help='Video file for processing', required=True)
    parser.add_argument('-s', '--start', type=int, help='Start of video in seconds', required=True)
    parser.add_argument('-e', '--end', type=int, help='End of video in seconds', required=True)
    parser.add_argument('-o', '--overlap', type=int, help='Overlap from one image to the next, in percent', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()
    if args.verbose:
        subtitles = args.input.split('.')[0] + '.SRT'
        print(f"Input video is {args.input}, subtitles is {subtitles}, starting at {args.start} and ending at {args.end} with {args.overlap}% of overlap")
    print(f"Procesing {args.input} with {args.overlap}% of overlap.")
    process_video(args.input, args.start, args.end, args.overlap)


def process_video(video, start, end, overlap):
    '''
    It uses cv2 to split a video into images with GPS metadata
    '''
    video_capture = cv2.VideoCapture(video)
    # Overlap in seconds starts at 5, and will be adjusted depending on overlap percentage
    overlap_seconds = 3

    if not video_capture.isOpened():
        print("Error: Could not open video.")
        exit()
    current_time = start
    prev_image = None
    subtitle_file = video.split('.')[0] + '.SRT'

    while current_time < end:
        video_capture.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        success, image = video_capture.read()
        if not success:
            print("Error: Could not read video frame.")
            break
        cv2.imwrite(f"image_{current_time}.jpg", image)
        latitude, longitude = get_gps_metadata(subtitle_file, current_time)
        gps_to_exif(f"image_{current_time}.jpg", f"image_{current_time}-gps.jpg", latitude, longitude)
        if prev_image is not None:
            current_overlap = images_overlap(prev_image, "image_"+str(current_time)+".jpg")
            #current_overlap = 70
            # current_overlap can be +- 5% of the desired overlap
            print(f"Overlap: {overlap}, Current Overlap: {current_overlap}")
            if current_overlap < (overlap + 10):
                print(f"Error: Images do not overlap enough. Lowering overlap to {overlap_seconds} seconds.")
                if overlap_seconds < 0.1:
                    print("Error: Overlap is too low. Exiting.")
                    break
                overlap_seconds -= 0.5
            if current_overlap > (overlap - 10):
                print(f"Overlap is to high. Increasing overlap by seconds 0.5s")
                overlap_seconds += 0.5
            print(f"Overlap between {current_time - overlap_seconds} and {current_time} is {current_overlap}%")
        prev_image = "image_"+str(current_time)+".jpg"
        current_time += overlap_seconds

    print("Video processed successfully.")
    video_capture.release()
    cv2.destroyAllWindows()

def images_overlap(image1_file, image2_file):
    '''
    It checks if two images overlap by a certain percentage
    '''
    print(f"Comparing {image1_file} and {image2_file}")
   
    # Load the images
    image1 = img_as_float(io.imread(image1_file))
    image2 = img_as_float(io.imread(image2_file))

    img1_float32 = np.float32(image1)
    img2_float32 = np.float32(image2)
    before_gray = cv2.cvtColor(img1_float32, cv2.COLOR_BGRA2GRAY)
    after_gray = cv2.cvtColor(img2_float32, cv2.COLOR_BGRA2GRAY)

    # Compute SSIM
#    ssim_index, ssim_image = ssim(image1, image2, full=True, channel_axis=None)
    ssim_index, ssim_image = ssim(before_gray, after_gray, full=True, data_range=after_gray.max() - after_gray.min())

    print(f'SSIM Index: {ssim_index}')
    return ssim_index * 1000


"""
     image1 = cv2.imread(image1_file, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_file, cv2.IMREAD_GRAYSCALE)

    # Resize images to the same size
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    # Compute SSIM
    similarity_index, _ = ssim(image1, image2, full=True)

    # Estimate overlap percentage
    overlap_percentage = similarity_index * 100
    print(f'Overlap Percentage: {overlap_percentage:.2f}%')
 """

def get_gps_metadata(subtitle_file, seconds):
    '''
    It reads a subtitle file and returns the GPS metadata for a given second
    as a tuple (latitude, longitude)
    '''
    subs = pysrt.open(subtitle_file)
    subs.slice(starts_after={'hours': 0, 'minutes': 0, 'seconds': seconds})
    latitude, longitude = float(subs.text.split("[")[7][10:19]), float(subs.text.split("[")[8][10:21])
    return (latitude, longitude)


def convert_to_rational(number):
    """Convert a number to rational format for EXIF."""
    return (int(number * 10000), 10000)


def gps_to_exif(image, output_image, latitude, longitude):
    '''
    It writes the GPS metadata to an image
    '''
    image = Image.open(image)
    
    # Prepare GPS data
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: 'N' if latitude >= 0 else 'S',
        piexif.GPSIFD.GPSLatitude: [convert_to_rational(abs(latitude)), (0, 1), (0, 1)],
        piexif.GPSIFD.GPSLongitudeRef: 'E' if longitude >= 0 else 'W',
        piexif.GPSIFD.GPSLongitude: [convert_to_rational(abs(longitude)), (0, 1), (0, 1)],
    }

    # Get existing EXIF data
    if "exif" in image.info:
        exif_dict = piexif.load(image.info["exif"])
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}
    exif_dict['GPS'] = gps_ifd

    # Convert EXIF data to bytes
    exif_bytes = piexif.dump(exif_dict)
    image.save(output_image, exif=exif_bytes)

if __name__ == "__main__":
    main()