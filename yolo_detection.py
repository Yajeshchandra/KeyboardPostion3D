import cv2
import numpy as np
from ultralytics import YOLO

def detect_objects_in_image(image_path, model_path='yolov8x.pt', scale_factor=0.5):
    """
    Detect objects in an image and return the annotated image with all detections visualized.
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model weights
        scale_factor: Factor by which to scale down the image
        
    Returns:
        annotated_image: Image with detection visualization
    """
    # Load model
    model = YOLO(model_path)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Scale down the image
    original_size = image.shape[:2]
    scaled_size = (int(original_size[1] * scale_factor), int(original_size[0] * scale_factor))
    image = cv2.resize(image, scaled_size, interpolation=cv2.INTER_AREA)
    
    # Run inference
    results = model(image, verbose=False)[0]
    
    # Annotate all detections
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = list(map(int, det[:4])) + [det[4], int(det[5])]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add class and confidence score
        text = f"Class {cls}: {conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def main():
    # Specify your image path
    image_path = '26.jpg_binarized.jpg'
    
    try:
        # Detect objects and get annotated image
        annotated_image = detect_objects_in_image(image_path, scale_factor=0.5)
        
        # Show result
        cv2.imshow('Object Detection', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
