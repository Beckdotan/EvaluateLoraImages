import { useState, useEffect, useRef } from 'react';
import { useLocation, Link } from 'react-router-dom';
import './Gallery.css';

function Gallery() {
  const location = useLocation();
  const { referenceImages, generatedImage, results } = location.state || {};
  
  // State for face detection toggle
  const [showFaces, setShowFaces] = useState(true);
  
  // State for selected image
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedResult, setSelectedResult] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [selectedType, setSelectedType] = useState(null); // 'reference' or 'generated'
  
  // Container ref for getting proper dimensions
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  // Basic validation
  if (!referenceImages || !generatedImage || !results) {
    return (
      <div className="error-container">
        <h2>No images to display</h2>
        <p>Please upload images on the home page first.</p>
        <Link to="/" className="btn">Go to Upload Page</Link>
      </div>
    );
  }

  // Separate results into reference and generated
  const referenceResults = results.filter(result => result.type === 'reference');
  const generatedResult = results.find(result => result.type === 'generated');

  // Set default selected image if none is selected
  useEffect(() => {
    if (!selectedImage && generatedResult) {
      // Default to the generated image
      setSelectedImage(generatedResult.image_base64);
      setSelectedResult(generatedResult);
      setSelectedType('generated');
      setSelectedIndex(0);
    }
  }, [selectedImage, generatedResult]);

  // Function to draw face detection overlays on the canvas
  const drawFaceDetections = () => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    const container = containerRef.current;
    
    if (!canvas || !image || !container || !selectedResult) {
      console.log("Missing requirements for drawing:", { canvas, image, container, selectedResult });
      return;
    }
    
    // Get container and image dimensions
    const containerRect = container.getBoundingClientRect();
    const imageRect = image.getBoundingClientRect();
    
    // Set canvas size to match the container
    canvas.width = containerRect.width;
    canvas.height = containerRect.height;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Don't draw if faces are hidden
    if (!showFaces || !selectedResult.faces || selectedResult.faces.length === 0) {
      return;
    }
    
    // Calculate scaling factors and offsets to map original image coordinates to displayed image
    const scale = {
      x: imageRect.width / image.naturalWidth,
      y: imageRect.height / image.naturalHeight
    };
    
    // Calculate offset (when image is centered in container)
    const offset = {
      x: (containerRect.width - imageRect.width) / 2,
      y: (containerRect.height - imageRect.height) / 2
    };
    
    console.log("Drawing parameters:", {
      containerSize: { width: containerRect.width, height: containerRect.height },
      imageDisplaySize: { width: imageRect.width, height: imageRect.height },
      imageNaturalSize: { width: image.naturalWidth, height: image.naturalHeight },
      scale,
      offset
    });
    
    // Draw face detections
    ctx.strokeStyle = '#FF0000';  // Red for faces
    ctx.lineWidth = 2;
    
    selectedResult.faces.forEach((face, index) => {
      if (face.coordinates && face.coordinates.length === 4) {
        const [x1, y1, x2, y2] = face.coordinates;
        
        // Scale and offset the coordinates
        const scaledX = x1 * scale.x + offset.x;
        const scaledY = y1 * scale.y + offset.y;
        const scaledWidth = (x2 - x1) * scale.x;
        const scaledHeight = (y2 - y1) * scale.y;
        
        console.log(`Face ${index} coordinates:`, {
          original: { x1, y1, x2, y2 },
          scaled: { x: scaledX, y: scaledY, width: scaledWidth, height: scaledHeight }
        });
        
        // Draw rectangle
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
        
        // Draw confidence score if available
        if (face.confidence) {
          ctx.fillStyle = '#FF0000';
          ctx.font = '12px Arial';
          ctx.fillText(
            `${Math.round(face.confidence * 100)}%`, 
            scaledX, 
            scaledY - 5
          );
        }
      }
    });
  };
  
  // Redraw detections when relevant states change
  useEffect(() => {
    drawFaceDetections();
  }, [selectedImage, selectedResult, showFaces]);
  
  // Add event listener for window resize
  useEffect(() => {
    const handleResize = () => {
      drawFaceDetections();
    };
    
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [selectedImage, selectedResult, showFaces]);
  
  // Handle image load
  useEffect(() => {
    if (imageRef.current) {
      const handleImageLoad = () => {
        console.log("Image loaded, drawing detections");
        // Give browser time to calculate proper dimensions
        setTimeout(drawFaceDetections, 100);
      };
      
      if (imageRef.current.complete) {
        handleImageLoad();
      } else {
        imageRef.current.onload = handleImageLoad;
      }
    }
  }, [selectedImage]);

  // Handle reference thumbnail click
  const handleReferenceClick = (resultIndex) => {
    const result = referenceResults[resultIndex];
    setSelectedImage(result.image_base64);
    setSelectedResult(result);
    setSelectedType('reference');
    setSelectedIndex(resultIndex);
  };

  // Handle generated image click
  const handleGeneratedClick = () => {
    if (generatedResult) {
      setSelectedImage(generatedResult.image_base64);
      setSelectedResult(generatedResult);
      setSelectedType('generated');
      setSelectedIndex(0);
    }
  };

  // Count faces for a result
  const countFaces = (result) => {
    return result && result.faces ? result.faces.length : 0;
  };

  return (
    <div className="two-panel-gallery">
      {/* Left panel - Thumbnails */}
      <div className="thumbnails-panel">
        {/* Reference Images Section */}
        <div className="panel-section">
          <h3>Reference Images</h3>
          <div className="thumbnail-grid">
            {referenceResults.map((result, index) => (
              <div 
                key={`ref-${index}`} 
                className={`thumbnail ${selectedType === 'reference' && selectedIndex === index ? 'selected' : ''}`}
                onClick={() => handleReferenceClick(index)}
              >
                <img 
                  src={result.thumbnail_base64 || result.image_base64} 
                  alt={`Reference ${index + 1}`} 
                />
                <div className="thumbnail-badge">
                  <span className="face-count">{countFaces(result)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Generated Image Section */}
        <div className="panel-section">
          <h3>Generated Image</h3>
          <div className="thumbnail-grid">
            {generatedResult && (
              <div 
                className={`thumbnail ${selectedType === 'generated' ? 'selected' : ''}`}
                onClick={handleGeneratedClick}
              >
                <img 
                  src={generatedResult.thumbnail_base64 || generatedResult.image_base64} 
                  alt="Generated" 
                />
                <div className="thumbnail-badge">
                  <span className="face-count">{countFaces(generatedResult)}</span>
                </div>
              </div>
            )}
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
              <span>Face Detection</span>
            </button>
          </div>
        </div>
        
        <div className="bottom-controls">
          <Link to="/" className="btn">Upload More Images</Link>
        </div>
      </div>
      
      {/* Right panel - Selected image with detection overlays */}
      <div className="details-panel">
        <div 
          ref={containerRef}
          className="selected-image-container"
        >
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
          {selectedResult && (
            <div className="image-info">
              <h3>
                {selectedType === 'generated' 
                  ? 'Generated Image' 
                  : `Reference Image ${selectedIndex + 1}`}
              </h3>
              <div className="detection-counts">
                <span className="detection-count">
                  <span className="count-icon" style={{backgroundColor: '#FF0000'}}></span>
                  Faces: {countFaces(selectedResult)}
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