import { useState, useEffect, useRef } from 'react';
import './Gallery.css';
import { mapDetectionCoordinates } from '../utils/detectionUtils';

function Gallery({ referenceImages, generatedImage }) {
  const [showFaces, setShowFaces] = useState(true);
  const [showHeads, setShowHeads] = useState(true);
  const [showBodies, setShowBodies] = useState(true);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  // Sample detection data structure
  const [evaluationResult] = useState({
    face: [{ coordinates: [100, 100, 200, 200], confidence: 0.95 }],
    head: [{ coordinates: [80, 80, 220, 220] }],
    body: [{ coordinates: [50, 50, 250, 250] }]
  });

  if (!referenceImages || !generatedImage) {
    return <div>No images to display. Please upload images first.</div>;
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imgRef.current;

    if (img && canvas) {
      canvas.width = img.offsetWidth;
      canvas.height = img.offsetHeight;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw face detections
      showFaces && evaluationResult.face.forEach(detection => {
        const [x1, y1, x2, y2] = mapDetectionCoordinates(img, detection.coordinates);
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2-x1, y2-y1);
      });

      // Draw head detections
      showHeads && evaluationResult.head.forEach(detection => {
        const [x1, y1, x2, y2] = mapDetectionCoordinates(img, detection.coordinates);
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2-x1, y2-y1);
      });

      // Draw body detections
      showBodies && evaluationResult.body.forEach(detection => {
        const [x1, y1, x2, y2] = mapDetectionCoordinates(img, detection.coordinates);
        ctx.strokeStyle = '#0000FF';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2-x1, y2-y1);
      });
    }
  }, [showFaces, showHeads, showBodies, evaluationResult]);

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
        <h2>Generated Image</h2>
        <div className="image-visualization">
          {generatedImage && (
            <div className="image-container">
              <img 
                ref={imgRef}
                src={generatedImage.url} 
                alt="Generated"
                onLoad={() => {
                if (canvasRef.current && imgRef.current) {
                  canvasRef.current.width = imgRef.current.offsetWidth;
                  canvasRef.current.height = imgRef.current.offsetHeight;
                }
              }}
              />
              <canvas ref={canvasRef} className="detection-overlay" />
            </div>
          )}
        </div>
      </section>

      <section className="evaluation-controls">
        <div className="toggle-group">
          <button 
            className={`toggle-btn ${showFaces ? 'active' : ''}`}
            onClick={() => setShowFaces(!showFaces)}
          >
            Faces ({evaluationResult.face.length})
          </button>
          <button
            className={`toggle-btn ${showHeads ? 'active' : ''}`}
            onClick={() => setShowHeads(!showHeads)}
          >
            Heads ({evaluationResult.head.length})
          </button>
          <button
            className={`toggle-btn ${showBodies ? 'active' : ''}`}
            onClick={() => setShowBodies(!showBodies)}
          >
            Bodies ({evaluationResult.body.length})
          </button>
        </div>
      </section>
    </div>
  );
}

export default Gallery;