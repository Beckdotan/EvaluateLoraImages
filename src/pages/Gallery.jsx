import { useState, useEffect, useRef } from 'react';
import { useLocation, Link } from 'react-router-dom';
import './Gallery.css';

function Gallery() {
  const location = useLocation();
  const { referenceImages, generatedImage, evaluationResult } = location.state || {};
  
  // State for detection toggles
  const [showFaces, setShowFaces] = useState(true);
  const [showHeads, setShowHeads] = useState(true);
  const [showBodies, setShowBodies] = useState(true);
  
  // State for selected image
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedType, setSelectedType] = useState(null); // 'reference' or 'generated'
  const [selectedIndex, setSelectedIndex] = useState(null);
  
  // Refs for canvas and image
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  // Basic validation
  if (!referenceImages || !generatedImage || !evaluationResult) {
    return (
      <div className="gallery-container">
        <h2>No images to display</h2>
        <p>Please upload images on the home page first.</p>
        <Link to="/" className="btn">Go to Upload Page</Link>
      </div>
    );
  }

  // Extract detection results
  let generatedDetections;
  let referenceDetections = [];

  // Check for different response formats and extract accordingly
  if (evaluationResult.generated_result) {
    // New API format
    generatedDetections = evaluationResult.generated_result || {};
    referenceDetections = evaluationResult.reference_results || [];
  } else {
    // Fallback - original format
    generatedDetections = {
      face: evaluationResult.face || [],
      head: evaluationResult.head || [],
      body: evaluationResult.body || []
    };
  }

  // Set default selected image if none is selected
  useEffect(() => {
    if (!selectedImage && generatedImage) {
      setSelectedImage(generatedImage.url);
      setSelectedType('generated');
      setSelectedIndex(0);
    }
  }, [selectedImage, generatedImage]);

  // Helper functions to get counts
  const countDetections = (detections, type) => {
    if (!detections || !detections[type]) return 0;
    return detections[type].length;
  };

  // Function to draw detection overlays on the canvas
  const drawDetections = () => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    
    if (!canvas || !image || !selectedType) {
      console.log("Missing requirements for drawing:", { canvas, image, selectedType });
      return;
    }
    
    console.log("Drawing detections for:", { type: selectedType, index: selectedIndex });
    
    const ctx = canvas.getContext('2d');
    
    // Get image dimensions and position within the container
    const imageRect = image.getBoundingClientRect();
    const containerRect = image.parentElement.getBoundingClientRect();
    
    // Calculate the actual displayed size of the image
    const displayedWidth = imageRect.width;
    const displayedHeight = imageRect.height;
    
    // Calculate offset within the container (for centering)
    const offsetX = (containerRect.width - displayedWidth) / 2;
    const offsetY = (containerRect.height - displayedHeight) / 2;
    
    // Set canvas size to match the container
    canvas.width = containerRect.width;
    canvas.height = containerRect.height;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Get the appropriate detections based on selected image
    const detections = selectedType === 'generated' 
      ? generatedDetections
      : referenceDetections[selectedIndex];
    
    if (!detections) {
      console.log("No detection data available for this image");
      return;
    }
    
    console.log("Detection data:", detections);
    
    // Scale factor for mapping coordinates from original image to displayed size
    const scaleX = displayedWidth / (image.naturalWidth || displayedWidth);
    const scaleY = displayedHeight / (image.naturalHeight || displayedHeight);
    
    console.log("Scale factors:", { scaleX, scaleY, offsetX, offsetY });
    
    // Draw face detections
    if (showFaces && detections.face && detections.face.length > 0) {
      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 3;
      
      detections.face.forEach(face => {
        if (face.coordinates) {
          console.log("Processing face coordinates:", face.coordinates);
          let [x1, y1, x2, y2] = face.coordinates;
          
          // Check if the format is [x, y, width, height] instead of [x1, y1, x2, y2]
          if (x2 < x1 || y2 < y1) {
            x2 = x1 + x2;
            y2 = y1 + y2;
          }
          
          // Scale coordinates to displayed image size and add offset
          const scaledX1 = (x1 * scaleX) + offsetX;
          const scaledY1 = (y1 * scaleY) + offsetY;
          const scaledWidth = (x2 - x1) * scaleX;
          const scaledHeight = (y2 - y1) * scaleY;
          
          console.log("Drawing face rectangle:", { 
            x: scaledX1, 
            y: scaledY1, 
            width: scaledWidth, 
            height: scaledHeight 
          });
          
          ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);
          
          // Add confidence label if available
          if (face.confidence) {
            ctx.fillStyle = '#FF0000';
            ctx.font = '14px Arial';
            ctx.fillText(`Face: ${Math.round(face.confidence * 100)}%`, scaledX1, scaledY1 - 5);
          }
        }
      });
    }
    
    // Draw head detections
    if (showHeads && detections.head && detections.head.length > 0) {
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      
      detections.head.forEach(head => {
        if (head.coordinates) {
          console.log("Processing head coordinates:", head.coordinates);
          let [x1, y1, x2, y2] = head.coordinates;
          
          // Check if the format is [x, y, width, height] instead of [x1, y1, x2, y2]
          if (x2 < x1 || y2 < y1) {
            x2 = x1 + x2;
            y2 = y1 + y2;
          }
          
          // Scale coordinates to displayed image size and add offset
          const scaledX1 = (x1 * scaleX) + offsetX;
          const scaledY1 = (y1 * scaleY) + offsetY;
          const scaledWidth = (x2 - x1) * scaleX;
          const scaledHeight = (y2 - y1) * scaleY;
          
          console.log("Drawing head rectangle:", { 
            x: scaledX1, 
            y: scaledY1, 
            width: scaledWidth, 
            height: scaledHeight 
          });
          
          ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);
        }
      });
    }
    
    // Draw body detections
    if (showBodies && detections.body && detections.body.length > 0) {
      ctx.strokeStyle = '#0000FF';
      ctx.lineWidth = 3;
      
      detections.body.forEach(body => {
        if (body.coordinates) {
          console.log("Processing body coordinates:", body.coordinates);
          let [x1, y1, x2, y2] = body.coordinates;
          
          // Check if the format is [x, y, width, height] instead of [x1, y1, x2, y2]
          if (x2 < x1 || y2 < y1) {
            x2 = x1 + x2;
            y2 = y1 + y2;
          }
          
          // Scale coordinates to displayed image size and add offset
          const scaledX1 = (x1 * scaleX) + offsetX;
          const scaledY1 = (y1 * scaleY) + offsetY;
          const scaledWidth = (x2 - x1) * scaleX;
          const scaledHeight = (y2 - y1) * scaleY;
          
          console.log("Drawing body rectangle:", { 
            x: scaledX1, 
            y: scaledY1, 
            width: scaledWidth, 
            height: scaledHeight 
          });
          
          ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);
        }
      });
    }
  };
  
  // Redraw detections when relevant states change
  useEffect(() => {
    drawDetections();
  }, [selectedImage, selectedType, selectedIndex, showFaces, showHeads, showBodies]);
  
  // Add a resize listener to redraw the canvas when the window size changes
  useEffect(() => {
    const handleResize = () => {
      console.log("Window resized, redrawing detections");
      drawDetections();
    };
    
    window.addEventListener('resize', handleResize);
    
    // Clean up
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [selectedImage, selectedType, selectedIndex, showFaces, showHeads, showBodies]);
  
  // Redraw when the image is loaded
  useEffect(() => {
    if (imageRef.current) {
      const handleImageLoad = () => {
        console.log("Image loaded, drawing detections");
        drawDetections();
      };
      
      if (imageRef.current.complete) {
        handleImageLoad();
      } else {
        imageRef.current.onload = handleImageLoad;
      }
    }
  }, [selectedImage]);

  // Helper function to handle thumbnail click
  const handleThumbnailClick = (url, type, index) => {
    setSelectedImage(url);
    setSelectedType(type);
    setSelectedIndex(index);
  };

  return (
    <div className="two-panel-gallery">
      {/* Left panel - Thumbnails */}
      <div className="thumbnails-panel">
        <div className="panel-section">
          <h3>Reference Images</h3>
          <div className="thumbnail-grid">
            {referenceImages.map((image, index) => (
              <div 
                key={`ref-${index}`} 
                className={`thumbnail ${selectedType === 'reference' && selectedIndex === index ? 'selected' : ''}`}
                onClick={() => handleThumbnailClick(image.url, 'reference', index)}
              >
                <img 
                  src={image.url} 
                  alt={`Reference ${index + 1}`} 
                />
                <div className="thumbnail-badge">
                  <span className="face-count">{countDetections(referenceDetections[index], 'face')}</span>
                  <span className="head-count">{countDetections(referenceDetections[index], 'head')}</span>
                  <span className="body-count">{countDetections(referenceDetections[index], 'body')}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="panel-section">
          <h3>Generated Image</h3>
          <div className="thumbnail-grid">
            <div 
              className={`thumbnail ${selectedType === 'generated' ? 'selected' : ''}`}
              onClick={() => handleThumbnailClick(generatedImage.url, 'generated', 0)}
            >
              <img 
                src={generatedImage.url} 
                alt="Generated" 
              />
              <div className="thumbnail-badge">
                <span className="face-count">{countDetections(generatedDetections, 'face')}</span>
                <span className="head-count">{countDetections(generatedDetections, 'head')}</span>
                <span className="body-count">{countDetections(generatedDetections, 'body')}</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="detection-controls">
          <h3>Detection Controls</h3>
          <div className="control-buttons">
            <button 
              className={`control-btn ${showFaces ? 'active' : ''}`}
              onClick={() => setShowFaces(!showFaces)}
            >
              <span className="btn-icon" style={{backgroundColor: '#FF0000'}}></span>
              <span>Faces</span>
            </button>
            
            <button 
              className={`control-btn ${showHeads ? 'active' : ''}`}
              onClick={() => setShowHeads(!showHeads)}
            >
              <span className="btn-icon" style={{backgroundColor: '#00FF00'}}></span>
              <span>Heads</span>
            </button>
            
            <button 
              className={`control-btn ${showBodies ? 'active' : ''}`}
              onClick={() => setShowBodies(!showBodies)}
            >
              <span className="btn-icon" style={{backgroundColor: '#0000FF'}}></span>
              <span>Bodies</span>
            </button>
          </div>
        </div>
        
        <div className="bottom-controls">
          <Link to="/" className="btn">Upload More Images</Link>
        </div>
      </div>
      
      {/* Right panel - Selected image with detection overlays */}
      <div className="details-panel">
        <div className="selected-image-container">
          {selectedImage && (
            <>
              <img 
                ref={imageRef}
                src={selectedImage} 
                alt="Selected" 
              />
              <canvas 
                ref={canvasRef}
                className="detection-overlay"
              />
            </>
          )}
        </div>
        
        <div className="image-info-bar">
          {selectedType && (
            <div className="image-info">
              <h3>{selectedType === 'generated' ? 'Generated Image' : `Reference Image ${selectedIndex + 1}`}</h3>
              <div className="detection-counts">
                <span className="detection-count face-count">
                  <span className="count-icon" style={{backgroundColor: '#FF0000'}}></span>
                  Faces: {selectedType === 'generated' 
                    ? countDetections(generatedDetections, 'face')
                    : countDetections(referenceDetections[selectedIndex], 'face')}
                </span>
                <span className="detection-count head-count">
                  <span className="count-icon" style={{backgroundColor: '#00FF00'}}></span>
                  Heads: {selectedType === 'generated'
                    ? countDetections(generatedDetections, 'head')
                    : countDetections(referenceDetections[selectedIndex], 'head')}
                </span>
                <span className="detection-count body-count">
                  <span className="count-icon" style={{backgroundColor: '#0000FF'}}></span>
                  Bodies: {selectedType === 'generated'
                    ? countDetections(generatedDetections, 'body')
                    : countDetections(referenceDetections[selectedIndex], 'body')}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Gallery;