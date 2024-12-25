import numpy as np
import cv2
import json

class KeyboardDetector:
    def __init__(self):
        """Initialize with the provided keyboard measurements"""
        self.TOTAL_LENGTH_MM = 354.076  # Length in mm
        self.TOTAL_WIDTH_MM = 123.444   # Width in mm
        
        # Load key positions
        with open('final_points.json', 'r') as f:
            self.key_positions = {
            key: np.array(pos, dtype=np.float32)
            for key, pos in json.load(f).items()
            }
        
        # Convert key positions to 3D coordinates (Z=0 for planar surface)
        self.object_points = []
        self.key_names = []
        for key, pos in self.key_positions.items():
            self.object_points.append([pos[0], pos[1], 0])
            self.key_names.append(key)
        self.object_points = np.array(self.object_points, dtype=np.float32)

    def apply_transform(self, points, matrix):
        """Apply transformation matrix to points"""
        # Convert to homogeneous coordinates
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        # Apply transform
        transformed = np.dot(homogeneous, matrix.T)
        # Convert back from homogeneous coordinates
        return transformed[:, :2] / transformed[:, 2:]

    def detect_keyboard(self, image, debug=False):
        """
        Detect keyboard in image and return key positions
        
        Args:
            image: Input BGR image
            debug: If True, return additional debug information
            
        Returns:
            dict of key_name: (x, y) pixel coordinates
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
            
        # Find the largest contour (assumed to be keyboard)
        keyboard_contour = max(contours, key=cv2.contourArea)
        
        # Get corner points
        rect = cv2.minAreaRect(keyboard_contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)
        
        # Sort points to ensure consistent order: top-left, top-right, bottom-right, bottom-left
        box_sorted = np.zeros_like(box)
        sum_pts = box.sum(axis=1)
        diff_pts = np.diff(box, axis=1)
        
        box_sorted[0] = box[np.argmin(sum_pts)]      # Top-left
        box_sorted[2] = box[np.argmax(sum_pts)]      # Bottom-right
        box_sorted[1] = box[np.argmin(diff_pts)]     # Top-right
        box_sorted[3] = box[np.argmax(diff_pts)]     # Bottom-left
        
        # Define target points (rectangle with keyboard dimensions)
        target = np.array([
            [0, 0],
            [self.TOTAL_LENGTH_MM, 0],
            [self.TOTAL_LENGTH_MM, self.TOTAL_WIDTH_MM],
            [0, self.TOTAL_WIDTH_MM]
        ], dtype=np.float32)
        
        # Get perspective transform
        matrix = cv2.getPerspectiveTransform(target, box_sorted)
        
        # Transform all key positions
        key_points = np.array([pos for pos in self.key_positions.values()])
        transformed_points = self.apply_transform(key_points, matrix)
        
        # Create dictionary of transformed key positions
        detected_keys = {}
        for key, pos in zip(self.key_positions.keys(), transformed_points):
            detected_keys[key] = tuple(map(int, pos))
            
        if debug:
            return detected_keys, {
                'contour': keyboard_contour,
                'box': box_sorted,
                'transform': matrix
            }
            
        return detected_keys

    def draw_debug(self, image, debug_info):
        """Draw debug visualization"""
        result = image.copy()
        
        # Draw keyboard contour
        cv2.drawContours(result, [debug_info['contour']], 0, (0, 255, 0), 2)
        
        # Draw corner points
        for point in debug_info['box']:
            cv2.circle(result, tuple(map(int, point)), 5, (0, 0, 255), -1)
            
        return result

    def draw_keys(self, image, key_positions):
        """Draw detected key positions on image"""
        result = image.copy()
        
        # Draw each key position
        for key, pos in key_positions.items():
            cv2.circle(result, pos, 3, (0, 255, 0), -1)
            cv2.putText(result, key, (pos[0]+5, pos[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        return result

# Usage example
def main():
    # Create detector
    detector = KeyboardDetector()
    
    # Load image
    image = cv2.imread('26.jpg')
    
    # Detect keys with debug info
    result, debug_info = detector.detect_keyboard(image, debug=True)
    
    if result:
        # Draw debug visualization
        debug_view = detector.draw_debug(image, debug_info)
        cv2.imshow('Debug View', debug_view)
        
        # Draw key positions
        key_view = detector.draw_keys(image, result)
        cv2.imshow('Detected Keys', key_view)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to detect keyboard")

if __name__ == "__main__":
    main()