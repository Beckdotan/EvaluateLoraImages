export const mapDetectionCoordinates = (imgElement, originalCoords) => {
  const displayedWidth = imgElement.offsetWidth;
  const displayedHeight = imgElement.offsetHeight;
  const scaleX = displayedWidth / imgElement.naturalWidth;
  const scaleY = displayedHeight / imgElement.naturalHeight;

  return originalCoords.map((coord, index) => {
    return index % 2 === 0 
      ? coord * scaleX
      : coord * scaleY;
  });
};