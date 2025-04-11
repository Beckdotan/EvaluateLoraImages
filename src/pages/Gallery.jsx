import { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import './Gallery.css';
import { mapDetectionCoordinates } from '../utils/detectionUtils';

function Gallery() {
  const location = useLocation();
  const { referenceImages, generatedImage, evaluationResult } = location.state || {};
  
  const [showFaces, setShowFaces] = useState(true);
  const [showHeads, setShowHeads] = useState(true);
  const [showBodies, setShowBodies] = useState(true);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  if (!referenceImages || !generatedImage || !evaluationResult) {
    return (
      <div className="gallery-container">
        <h2>No images to display</h2>
        <p>Please upload images on the home page first.</p>
      </div>
    );
  }

  // Extract the evaluation data from the server response
  const generatedDetections = evaluationResult.generated_result || {
    face: [],
    head: [],
    body: []
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imgRef.current;

    if (img && canvas) {
      // Wait for the image to be fully loaded
      const handleImageLoad = () => {
        canvas.width = img.offsetWidth;
        canvas.height = img.offsetHeight;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw face detections
        if (showFaces && generatedDetections.face && generatedDetections.face.length > 0) {
          generatedDetections.face.forEach(detection => {
            if (detection.coordinates) {
              const [x1, y1, x2, y2] = mapDetectionCoordinates(img, detection.coordinates);
              ctx.strokeStyle = '#FF0000';
              ctx.lineWidth = 2;
              ctx.strokeRect(x1, y1, x2-x1, y2-y1);
              
              // Add confidence text if available
              if (detection.confidence) {
                ctx.fillStyle = '#FF0000';
                ctx.font = '12px Arial';
                ctx.fillText(`${Math.round(detection.confidence * 100)}%`, x1, y1 - 5);
              }
            }
          });
        }

        // Draw head detections
        if (showHeads && generatedDetections.head && generatedDetections.head.length > 0) {
          generatedDetections.head.forEach(detection => {
            if (detection.coordinates) {
              const [x1, y1, x2, y2] = mapDetectionCoordinates(img, detection.coordinates);
              ctx.strokeStyle = '#00FF00';
              ctx.lineWidth = 2;
              ctx.strokeRect(x1, y1, x2-x1, y2-y1);
            }
          });
        }

        // Draw body detections
        if (showBodies && generatedDetections.body && generatedDetections.body.length > 0) {
          generatedDetections.body.forEach(detection => {
            if (detection.coordinates) {
              const [x1, y1, x2, y2] = mapDetectionCoordinates(img, detection.coordinates);
              ctx.strokeStyle = '#0000FF';
              ctx.lineWidth = 2;
              ctx.strokeRect(x1, y1, x2-x1, y2-y1);
            }
          });
        }
      };
      
      if (img.complete) {
        handleImageLoad();
      } else {
        img.onload = handleImageLoad;
      }
    }
  }, [showFaces, showHeads, showBodies, generatedDetections]);

  // Count the number of detections
  const faceCount = generatedDetections.face?.length || 0;
  const headCount = generatedDetections.head?.length || 0;
  const bodyCount = generatedDetections.body?.length || 0;

  return (
    <div className="gallery-container">
      <section className="reference-images">
        <h2>Reference Images</h2>
        <div className="image-grid">
          {referenceImages.map((image, index) => (
            <div key={index} className="image-card">
              <img src={image.url} alt={`Reference ${index + 1}`} />
            </div>
          ))}
        </div>
      </section>

      <section className="generated-image">
        <h2>Generated Image with Detections</h2>
        <div className="image-visualization">
          {generatedImage && (
            <div className="image-container">
              <img 
                ref={imgRef}
                src={generatedImage.url} 
                alt="Generated"
              />
              <canvas ref={canvasRef} className="detection-overlay" />
            </div>
          )}
        </div>
      </section>

      <section className="evaluation-controls">
        <h3>Detection Controls</h3>
        <div className="toggle-group">
          <button 
            className={`toggle-btn ${showFaces ? 'active' : ''}`}
            onClick={() => setShowFaces(!showFaces)}
          >
            {showFaces ? '✓' : '○'} Faces ({faceCount})
          </button>
          <button
            className={`toggle-btn ${showHeads ? 'active' : ''}`}
            onClick={() => setShowHeads(!showHeads)}
          >
            {showHeads ? '✓' : '○'} Heads ({headCount})
          </button>
          <button
            className={`toggle-btn ${showBodies ? 'active' : ''}`}
            onClick={() => setShowBodies(!showBodies)}
          >
            {showBodies ? '✓' : '○'} Bodies ({bodyCount})
          </button>
        </div>
      </section>

      <section className="detection-legend">
        <h3>Detection Legend</h3>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-color" style={{backgroundColor: '#FF0000'}}></span>
            <span>Face</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{backgroundColor: '#00FF00'}}></span>
            <span>Head</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{backgroundColor: '#0000FF'}}></span>
            <span>Body</span>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Gallery;