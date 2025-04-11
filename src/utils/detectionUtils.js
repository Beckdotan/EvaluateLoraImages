/**
 * Maps detection coordinates from the original image to the displayed image size
 * @param {HTMLImageElement} imgElement - The image element in the DOM
 * @param {Array} coordinates - Array of coordinates [x1, y1, x2, y2] or [x, y, w, h]
 * @returns {Array} - Scaled coordinates
 */
export const mapDetectionCoordinates = (imgElement, coordinates) => {
    if (!imgElement || !coordinates || coordinates.length < 4) {
      console.error('Invalid parameters for coordinate mapping');
      return [0, 0, 0, 0];
    }
  
    // Check if coordinates are in format [x, y, width, height] and convert to [x1, y1, x2, y2]
    let [x1, y1, x2OrW, y2OrH] = coordinates;
    
    // If the third and fourth values are width and height, convert to x2, y2
    if (x2OrW < x1 || y2OrH < y1) {
      x2OrW = x1 + x2OrW;
      y2OrH = y1 + y2OrH;
    }
  
    const displayedWidth = imgElement.offsetWidth;
    const displayedHeight = imgElement.offsetHeight;
    
    // Scale coordinates based on ratio of displayed size to natural size
    const scaleX = displayedWidth / imgElement.naturalWidth;
    const scaleY = displayedHeight / imgElement.naturalHeight;
  
    return [
      x1 * scaleX,
      y1 * scaleY,
      x2OrW * scaleX,
      y2OrH * scaleY
    ];
  };