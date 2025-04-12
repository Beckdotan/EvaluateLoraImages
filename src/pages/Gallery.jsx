import { useState, useEffect, useRef } from 'react';
import { useLocation, Link } from 'react-router-dom';
import './Gallery.css';

function Gallery() {
  const location = useLocation();
  const { referenceImages, generatedImage, results } = location.state || {};
  
  // Toggle states
  const [showFaces, setShowFaces] = useState(true);
  const [showBackgroundRemoved, setShowBackgroundRemoved] = useState(true);
  
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
      return;
    }
    
    // Get container dimensions
    const containerRect = container.getBoundingClientRect();
    
    // Set canvas size to match the container
    canvas.width = containerRect.width;
    canvas.height = containerRect.height;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Don't draw if faces are hidden
    if (!showFaces || !selectedResult.faces || selectedResult.faces.length === 0) {
      return;
    }
    
    // Get the dimensions of the displayed image
    const imageRect = image.getBoundingClientRect();
    
    // Original dimensions from the result data (the dimensions of the processed image)
    const originalWidth = selectedResult.width;
    const originalHeight = selectedResult.height;
    
    // Calculate scaling factors for the displayed image relative to original size
    const scaleX = imageRect.width / originalWidth;
    const scaleY = imageRect.height / originalHeight;
    
    // Calculate offset (when image is centered in container)
    const offsetX = (containerRect.width - imageRect.width) / 2;
    const offsetY = (containerRect.height - imageRect.height) / 2;
    
    // Draw face detections
    ctx.strokeStyle = '#FF0000';  // Red for faces
    ctx.lineWidth = 2;
    
    selectedResult.faces.forEach((face, index) => {
      if (face.coordinates && face.coordinates.length === 4) {
        const [x1, y1, x2, y2] = face.coordinates;
        
        // Scale coordinates to match displayed image size
        const scaledX = x1 * scaleX + offsetX;
        const scaledY = y1 * scaleY + offsetY;
        const scaledWidth = (x2 - x1) * scaleX;
        const scaledHeight = (y2 - y1) * scaleY;
        
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
  
  // Get the image to display based on toggles
  const getDisplayImage = () => {
    if (!selectedResult) return null;
    
    return showBackgroundRemoved 
      ? selectedResult.image_base64  // Background removed image
      : selectedResult.thumbnail_base64;  // Original processed image (with background)
  };
  
  // Redraw detections when relevant states change
  useEffect(() => {
    drawFaceDetections();
  }, [selectedImage, selectedResult, showFaces, showBackgroundRemoved]);
  
  // Add event listener for window resize
  useEffect(() => {
    const handleResize = () => {
      drawFaceDetections();
    };
    
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [selectedImage, selectedResult, showFaces, showBackgroundRemoved]);
  
  // Handle image load
  useEffect(() => {
    if (imageRef.current) {
      const handleImageLoad = () => {
        // Give browser time to calculate proper dimensions
        setTimeout(drawFaceDetections, 100);
      };
      
      if (imageRef.current.complete) {
        handleImageLoad();
      } else {
        imageRef.current.onload = handleImageLoad;
      }
    }
  }, [selectedImage, showBackgroundRemoved]);

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
                  src={result.thumbnail_base64} 
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
                  src={generatedResult.thumbnail_base64} 
                  alt="Generated" 
                />
                <div className="thumbnail-badge">
                  <span className="face-count">{countFaces(generatedResult)}</span>
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="bottom-controls">
          <Link to="/" className="btn">Upload More Images</Link>
        </div>
      </div>
      
      {/* Right panel - Selected image with detection overlays and controls */}
      <div className="details-panel">
        <div 
          ref={containerRef}
          className="selected-image-container"
        >
          {selectedResult && (
            <>
              <img 
                ref={imageRef}
                src={getDisplayImage()} 
                alt="Selected" 
              />
              <canvas 
                ref={canvasRef}
                className="detection-overlay"
              />
            </>
          )}
        </div>
        
        <div className="image-controls-bar">
          {selectedResult && (
            <>
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
                  <span className="image-dimensions">
                    {selectedResult.width} × {selectedResult.height} px
                  </span>
                </div>
              </div>
              
              <div className="detection-controls">
                <button 
                  className={`control-btn ${showFaces ? 'active' : ''}`}
                  onClick={() => setShowFaces(!showFaces)}
                >
                  <span className="btn-icon" style={{backgroundColor: '#FF0000'}}></span>
                  <span>Face Detection</span>
                </button>
                
                <button 
                  className={`control-btn ${showBackgroundRemoved ? 'active' : ''}`}
                  onClick={() => setShowBackgroundRemoved(!showBackgroundRemoved)}
                >
                  <span className="btn-icon" style={{backgroundColor: '#00BFFF'}}></span>
                  <span>Remove Background</span>
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default Gallery;