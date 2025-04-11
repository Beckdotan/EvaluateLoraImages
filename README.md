# EvaluateLoraImages

EvaluateLoraImages is a web application for evaluating AI-generated photos created using Lora by comparing them with reference images of the original person.

## Features

- Upload multiple reference images (1-5) of the original person
- Upload a single AI-generated image for comparison
- Interactive drag-and-drop interface for image uploads
- Image preview functionality
- Evaluation results display

## Prerequisites

Before running the application, make sure you have the following installed:

- Node.js (version 14.0.0 or higher)
- npm (usually comes with Node.js)

## Installation

1. Clone the repository or download the source code
2. Open a terminal in the project directory
3. Install the dependencies by running:

```bash
npm install
```

## Running the Application

### Development Mode

To run the application in development mode with hot-reload:

```bash
npm run dev
```

The application will start and be available at `http://localhost:5173`

### Production Build

To create a production build:

```bash
npm run build
```

To preview the production build:

```bash
npm run preview
```

## Project Structure

- `/src` - Source code directory
  - `/components` - Reusable React components
  - `/pages` - Page components
  - `App.jsx` - Main application component
  - `main.jsx` - Application entry point

## Technologies Used

- React.js
- Vite
- React Router
- React Dropzone
- Axios
