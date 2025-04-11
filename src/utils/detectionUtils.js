/**
 * Maps detection coordinates from the original image to the displayed image size
 * @param {HTMLImageElement} imgElement - The image element in the DOM
 * @param {Array} coordinates - Array of coordinates [x1, y1, x2, y2] or [x, y, w, h]
 * @returns {Array} - Scaled coordinates
 */
export const mapDetectionCoordinates = (imgElement, coordinates) => {
    if (!imgElement || !coordinates) {
      console.error('Invalid parameters for coordinate mapping', { imgElement, coordinates });
      return [0, 0, 0, 0];
    }
    
    // Log input parameters for debugging
    console.log('Mapping coordinates:', { 
      element: imgElement, 
      coordinates, 
      displayedWidth: imgElement.offsetWidth,
      displayedHeight: imgElement.offsetHeight,
      naturalWidth: imgElement.naturalWidth,
      naturalHeight: imgElement.naturalHeight
    });
  
    // Handle different coordinate formats
    let x1, y1, x2, y2;
    
    // Check if coordinates is an array
    if (Array.isArray(coordinates)) {
      if (coordinates.length >= 4) {
        // Format is [x1, y1, x2, y2]
        [x1, y1, x2, y2] = coordinates;
      } else {
        console.error('Coordinate array has incorrect length', coordinates);
        return [0, 0, 0, 0];
      }
    } else if (typeof coordinates === 'object') {
      // It might be an object with coordinate properties
      if ('xmin' in coordinates && 'ymin' in coordinates && 'xmax' in coordinates && 'ymax' in coordinates) {
        x1 = coordinates.xmin;
        y1 = coordinates.ymin;
        x2 = coordinates.xmax;
        y2 = coordinates.ymax;
      } else if ('x' in coordinates && 'y' in coordinates && 'width' in coordinates && 'height' in coordinates) {
        x1 = coordinates.x;
        y1 = coordinates.y;
        x2 = x1 + coordinates.width;
        y2 = y1 + coordinates.height;
      } else {
        console.error('Invalid coordinate object format', coordinates);
        return [0, 0, 0, 0];
      }
    } else {
      console.error('Coordinates must be an array or object', coordinates);
      return [0, 0, 0, 0];
    }
    
    // If the third and fourth values represent width and height, convert to x2, y2
    if (x2 < x1 || y2 < y1) {
      console.log('Converting width/height format to absolute coordinates');
      x2 = x1 + x2;
      y2 = y1 + y2;
    }
  
    const displayedWidth = imgElement.offsetWidth;
    const displayedHeight = imgElement.offsetHeight;
    
    // Ensure we don't divide by zero
    const naturalWidth = imgElement.naturalWidth || displayedWidth;
    const naturalHeight = imgElement.naturalHeight || displayedHeight;
    
    // Scale coordinates based on ratio of displayed size to natural size
    const scaleX = displayedWidth / naturalWidth;
    const scaleY = displayedHeight / naturalHeight;
    
    const scaledCoordinates = [
      x1 * scaleX,
      y1 * scaleY,
      x2 * scaleX,
      y2 * scaleY
    ];
    
    console.log('Scaled coordinates:', scaledCoordinates);
    
    return scaledCoordinates;
  };