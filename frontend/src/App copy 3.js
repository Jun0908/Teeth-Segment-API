import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [imageBase64, setImageBase64] = useState('');
  const [histograms, setHistograms] = useState([]);
  const [csvData, setCsvData] = useState([]);
  const [csvPlotData, setCsvPlotData] = useState([]);
  const [imageUrl, setImageUrl] = useState(null);

  const uploadImage = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setImageUrl(data.imageUrl);
    } catch (error) {
      console.error('Upload error', error);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      uploadImage(file);
    }
  };

  const fetchAndDisplayImage = async (endpoint) => {
    try {
      const response = await axios.get(`http://localhost:8000/${endpoint}`);
      setImageBase64(response.data.image_base64);
    } catch (error) {
      console.error(`Error fetching ${endpoint} image`, error);
    }
  };

  const fetchHistograms = async () => {
    try {
      const response = await axios.get('http://localhost:8000/generate_histograms');
      setHistograms(response.data);
    } catch (error) {
      console.error('Error fetching histograms', error);
    }
  };

  const fetchCSVData = async () => {
    try {
      const response = await axios.get('http://localhost:8000/download_csv/midpoints');
      setCsvData(response.data);
    } catch (error) {
      console.error('Error fetching CSV data:', error);
    }
  };

  const fetchPlotData = async () => {
    try {
      const response = await axios.get('http://localhost:8000/download_csv/final_plotted_points');
      setCsvPlotData(response.data);
    } catch (error) {
      console.error('Error fetching plot data:', error);
    }
  };

  const fetchAllData = async () => {
    try {
      await fetchAndDisplayImage('predict');
      await fetchAndDisplayImage('contours');
      await fetchAndDisplayImage('processed_imaged');
      await fetchHistograms();
      await fetchCSVData();
      await fetchPlotData();
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  return (
    <div>
      <h1>React Frontend for Dental X-Ray Analysis</h1>
      <h2>Image Upload</h2>
      <input type="file" onChange={handleFileChange} />
      {imageUrl && <img src={imageUrl} alt="Uploaded" />}
      
      <button onClick={() => fetchAndDisplayImage('predict')}>Get Predicted Image</button>
      <button onClick={() => fetchAndDisplayImage('contours')}>Get Contours Image</button>
      <button onClick={() => fetchAndDisplayImage('processed_imaged')}>Get Processed Image</button>
      
      {imageBase64 && (
        <div>
          <h2>Image Result</h2>
          <img src={`data:image/png;base64,${imageBase64}`} alt="Result" />
        </div>
      )}
      
      <div>
        <button onClick={fetchHistograms}>Get Histograms</button>
        {histograms.length > 0 && (
          <div>
            <h2>Histograms Data</h2>
            {histograms.map((hist, index) => (
              <div key={index}>
                <h3>Histogram {index + 1}</h3>
                <p>{hist.join(', ')}</p>
              </div>
            ))}
          </div>
        )}
        <button onClick={fetchAllData}>Fetch All Data</button>
        <table>
          <thead>
            <tr>
              {csvData[0] && Object.keys(csvData[0]).map((key) => <th key={key}>{key}</th>)}
            </tr>
          </thead>
          <tbody>
            {csvData.map((row, index) => (
              <tr key={index}>
                {Object.values(row).map((value, i) => <td key={i}>{value}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
  
      <div>
        <button onClick={fetchPlotData}>Get Plot Data</button>
        <table>
          <thead>
            <tr>
              {csvPlotData[0] && Object.keys(csvPlotData[0]).map((key) => <th key={key}>{key}</th>)}
            </tr>
          </thead>
          <tbody>
            {csvPlotData.map((row, index) => (
              <tr key={index}>
                {Object.values(row).map((value, i) => <td key={i}>{value}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;
