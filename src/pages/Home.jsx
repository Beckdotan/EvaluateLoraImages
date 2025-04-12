import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Dropzone from 'react-dropzone'
import heic2any from 'heic2any'
import './Home.css'

function Home() {
  const [referenceImages, setReferenceImages] = useState([])
  const [generatedImage, setGeneratedImage] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState('')
  const [isConverting, setIsConverting] = useState(false)
  const navigate = useNavigate()

  // Helper function to check if a file is HEIC
  const isHeicFile = (file) => {
    return file.type === 'image/heic' || 
           file.type === 'image/heif' || 
           file.name.toLowerCase().endsWith('.heic') || 
           file.name.toLowerCase().endsWith('.heif');
  }

  // Helper function to convert HEIC to JPEG
  const convertHeicToJpeg = async (file) => {
    try {
      const jpegBlob = await heic2any({
        blob: file,
        toType: 'image/jpeg',
        quality: 0.8
      });
      
      // If the result is an array (multiple images), take the first one
      const blob = Array.isArray(jpegBlob) ? jpegBlob[0] : jpegBlob;
      
      return new File([blob], file.name.replace(/\.(heic|heif)$/i, '.jpg'), {
        type: 'image/jpeg',
        lastModified: Date.now()
      });
    } catch (error) {
      console.error('HEIC conversion error:', error);
      throw new Error(`Failed to convert HEIC image: ${error.message}. Please try converting it to JPEG manually.`);
    }
  }

  // Function to detect and fix image orientation
  const fixImageOrientation = async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          // Create a canvas element
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          
          // Set proper canvas dimensions
          canvas.width = img.width;
          canvas.height = img.height;
          
          // Draw the image with correct orientation
          ctx.drawImage(img, 0, 0, img.width, img.height);
          
          // Convert canvas to blob
          canvas.toBlob((blob) => {
            const correctedFile = new File([blob], file.name, {
              type: 'image/jpeg',
              lastModified: Date.now()
            });
            resolve(correctedFile);
          }, 'image/jpeg', 0.95);
        };
        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = e.target.result;
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsDataURL(file);
    });
  }

  const handleReferenceImageDrop = async (acceptedFiles) => {
    setError('');
    setIsConverting(true);
    
    try {
      // Process files in parallel
      const newImages = await Promise.all(acceptedFiles.map(async (file) => {
        try {
          let processedFile = file;
          
          // Convert HEIC files to JPEG
          if (isHeicFile(file)) {
            processedFile = await convertHeicToJpeg(file);
          }
          
          // Fix orientation if needed
          try {
            processedFile = await fixImageOrientation(processedFile);
          } catch (orientationError) {
            console.warn('Could not fix orientation:', orientationError);
            // Continue with original file if orientation fix fails
          }
          
          // Create preview URL
          return Object.assign(processedFile, {
            preview: URL.createObjectURL(processedFile),
            isReference: true // Mark as reference image
          });
        } catch (error) {
          console.error(`Error processing file ${file.name}:`, error);
          setError(error.message);
          return null;
        }
      }));
      
      // Filter out any null values from failed conversions
      const validImages = newImages.filter(img => img !== null);
      setReferenceImages([...referenceImages, ...validImages]);
    } catch (error) {
      console.error('Error processing dropped files:', error);
      setError('Error processing images. Please try again with different files.');
    } finally {
      setIsConverting(false);
    }
  }

  const handleGeneratedImageDrop = async (acceptedFiles) => {
    setError('');
    
    if (acceptedFiles.length > 0) {
      setIsConverting(true);
      
      try {
        let file = acceptedFiles[0];
        
        // Convert HEIC file to JPEG if necessary
        if (isHeicFile(file)) {
          file = await convertHeicToJpeg(file);
        }
        
        // Fix orientation if needed
        try {
          file = await fixImageOrientation(file);
        } catch (orientationError) {
          console.warn('Could not fix orientation:', orientationError);
          // Continue with original file if orientation fix fails
        }
        
        setGeneratedImage(Object.assign(file, {
          preview: URL.createObjectURL(file),
          isGenerated: true // Mark as generated image
        }));
      } catch (error) {
        console.error('Error processing generated image:', error);
        setError(error.message);
      } finally {
        setIsConverting(false);
      }
    }
  }

  const removeReferenceImage = (index) => {
    const updatedImages = [...referenceImages]
    // Revoke the object URL to avoid memory leaks
    URL.revokeObjectURL(updatedImages[index].preview)
    updatedImages.splice(index, 1)
    setReferenceImages(updatedImages)
  }

  const removeGeneratedImage = () => {
    if (generatedImage) {
      URL.revokeObjectURL(generatedImage.preview)
      setGeneratedImage(null)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    // Validate inputs
    if (referenceImages.length === 0) {
      setError('Please upload at least one reference image')
      return
    }
    
    if (!generatedImage) {
      setError('Please upload a generated image')
      return
    }
    
    setIsUploading(true)
    setError('')
    
    try {
      // Create FormData
      const formData = new FormData()
      
      // Append all reference images first
      referenceImages.forEach((img) => {
        formData.append('files', img)
      })
      
      // Append generated image (should be the last file)
      formData.append('files', generatedImage)
      
      console.log('Uploading images for processing...');
      
      // Make API call to server
      const response = await fetch('http://localhost:8000/process', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(
          errorData.detail?.message || 
          (typeof errorData.detail === 'string' ? errorData.detail : 'Server returned an error')
        )
      }
      
      const data = await response.json()
      console.log('Received processing results:', data);
      
      if (!data.results || data.results.length === 0) {
        throw new Error('No results returned from the server');
      }
      
      // Navigate to gallery page with results
      navigate('/gallery', { 
        state: {
          referenceImages: referenceImages.map(img => ({ url: img.preview, name: img.name })),
          generatedImage: { url: generatedImage.preview, name: generatedImage.name },
          results: data.results
        }
      });
      
    } catch (err) {
      console.error('Upload error:', err)
      setError(`Failed to process images: ${err.message || 'Unknown error'}`)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="home-container">
      <h2>Background Removal & Face Detection</h2>
      <p>Upload images to remove backgrounds and detect faces.</p>
      
      {error && <div className="error-message">{error}</div>}
      
      <form onSubmit={handleSubmit}>
        <div className="dropzone-section">
          <h3>Reference Images</h3>
          <p>Upload 1-15 reference photos</p>
          
          <Dropzone 
            onDrop={handleReferenceImageDrop}
            accept={{
              'image/*': ['.jpeg', '.jpg', '.png', '.heic', '.heif']
            }}
            maxFiles={15}
            disabled={referenceImages.length >= 15 || isUploading || isConverting}
          >
            {({getRootProps, getInputProps}) => (
              <div 
                {...getRootProps()} 
                className={`dropzone ${referenceImages.length >= 5 || isConverting ? 'dropzone-disabled' : ''}`}
              >
                <input {...getInputProps()} />
                {isConverting ? (
                  <p>Converting images, please wait...</p>
                ) : (
                  <p>Drag & drop reference images here, or click to select files</p>
                )}
              </div>
            )}
          </Dropzone>
          
          {referenceImages.length > 0 && (
            <div className="preview-section">
              <h4>Reference Image Previews</h4>
              <div className="preview-grid">
                {referenceImages.map((file, index) => (
                  <div key={index} className="preview-item">
                    <img src={file.preview} alt={`Reference ${index + 1}`} />
                    <button 
                      type="button" 
                      className="remove-btn"
                      onClick={() => removeReferenceImage(index)}
                      disabled={isUploading || isConverting}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        <div className="dropzone-section">
          <h3>Generated Image</h3>
          <p>Upload the AI-generated image to analyze</p>
          
          <Dropzone 
            onDrop={handleGeneratedImageDrop}
            accept={{
              'image/*': ['.jpeg', '.jpg', '.png', '.heic', '.heif']
            }}
            maxFiles={1}
            disabled={isUploading || isConverting}
          >
            {({getRootProps, getInputProps}) => (
              <div {...getRootProps()} className="dropzone">
                <input {...getInputProps()} />
                {isConverting ? (
                  <p>Converting image, please wait...</p>
                ) : (
                  <p>Drag & drop generated image here, or click to select file</p>
                )}
              </div>
            )}
          </Dropzone>
          
          {generatedImage && (
            <div className="preview-section">
              <h4>Generated Image Preview</h4>
              <div className="preview-grid">
                <div className="preview-item">
                  <img src={generatedImage.preview} alt="Generated" />
                  <button 
                    type="button" 
                    className="remove-btn"
                    onClick={removeGeneratedImage}
                    disabled={isUploading || isConverting}
                  >
                    ×
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
        
        <button 
          type="submit" 
          className="upload-button"
          disabled={isUploading || isConverting || referenceImages.length === 0 || !generatedImage}
        >
          {isUploading ? 'Processing...' : 'Process Images'}
        </button>
      </form>
    </div>
  )
}

export default Home