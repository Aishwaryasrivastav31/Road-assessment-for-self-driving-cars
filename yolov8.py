import cv2
import glob
import os

# Define paths for images and annotations
image_path = 'archive/train/images/*.jpg'
annotation_path = 'archive/train/annotations/*.txt'
output_path = 'archive/train/visualized/'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Process each image and its corresponding annotation file
for image_file in glob.glob(image_path):
    # Generate corresponding annotation file path
    annotation_file = image_file.replace('images', 'annotations').replace('.jpg', '.txt')

    # Check if annotation file exists
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found for {image_file}: {annotation_file}")
        continue

    # Read the image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error reading image: {image_file}")
        continue
    
    height, width, _ = image.shape

    # Read annotations and draw bounding boxes
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipping line due to incorrect format: {line.strip()}")
                continue
            
            class_id, x_center, y_center, box_width, box_height = map(float, parts)

            # Convert YOLO format to pixel coordinates
            x_center_pixel = int(x_center * width)
            y_center_pixel = int(y_center * height)
            box_width_pixel = int(box_width * width)
            box_height_pixel = int(box_height * height)

            # Calculate top-left and bottom-right coordinates
            top_left_x = int(x_center_pixel - box_width_pixel / 2)
            top_left_y = int(y_center_pixel - box_height_pixel / 2)
            bottom_right_x = int(x_center_pixel + box_width_pixel / 2)
            bottom_right_y = int(y_center_pixel + box_height_pixel / 2)

            # Draw the bounding box on the image
            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 2)  # Blue box

    # Save the visualized image
    output_file = os.path.join(output_path, os.path.basename(image_file).replace('.jpg', '.png'))
    cv2.imwrite(output_file, image)
    print(f"Saved visualized image with bounding boxes: {output_file}")

print("Bounding boxes drawn and saved successfully.")
