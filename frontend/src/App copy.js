import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [imageBase64, setImageBase64] = useState('');
  const [histograms, setHistograms] = useState([]);
  const [csvData, setCsvData] = useState([]);
  const [csvPlotData, setCsvPlotData] = useState([]);
  const [imageUrl, setImageUrl] = useState(null);

  // 画像アップロードのための関数
  const uploadImage = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://172.31.201.73:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      // 取得した画像のURLを状態に設定します。
      setImageUrl(data.imageUrl);
    } catch (error) {
      console.error('アップロードエラー', error);
    }
  };

  // ファイルが選択されたときに実行される関数
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      uploadImage(file);
    }
  };

  const fetchPredictedImage = async () => {
    try {
      const response = await axios.get('http://172.31.201.73:5000/predict');
      setImageBase64(response.data.image_base64);
    } catch (error) {
      console.error('Error fetching predicted image', error);
    }
  };

  const fetchContoursImage = async () => {
    try {
      const response = await axios.get('http://172.31.201.73:5000/contours');
      setImageBase64(response.data.image_base64);
    } catch (error) {
      console.error('Error fetching contours image', error);
    }
  };

  const fetchHistograms = async () => {
    try {
      const response = await axios.get('http://172.31.201.73:5000/histograms');
      setHistograms(response.data);
    } catch (error) {
      console.error('Error fetching histograms', error);
    }
  };

  const fetchProcessedImage = async () => {
    try {
      const response = await axios.get('http://172.31.201.73:5000/processed_imaged');
      setImageBase64(response.data.image_base64);
    } catch (error) {
      console.error('Error fetching processed image', error);
    }
  };

  useEffect(() => {
    fetchPredictedImage();
  }, []);

  
  const fetchData = async () => {
      try {
        const response = await axios.get('http://172.31.201.73:5000/download_csv/midpoints');
        setCsvData(response.data);
      } catch (error) {
        console.error('Error fetching CSV data:', error);
      }
  };

  const fetchCSVData = async () => {
    try {
      const response = await axios.get('http://172.31.201.73:5000/download_csv/final_plotted_points');
      setCsvPlotData(response.data);
    } catch (error) {
      console.error('Error fetching CSV data:', error);
    }
  };

  return (
    
    <div>
      <h1>React Frontend for Dental X-Ray Analysis</h1>
      <h2>Image Upload</h2>
      <input type="file" onChange={handleFileChange} />
      {imageUrl && <img src={imageUrl} alt="Uploaded" />}
      
      <button onClick={fetchPredictedImage}>Get Predicted Image</button>
      <button onClick={fetchContoursImage}>Get Contours Image</button>
      
      <button onClick={fetchProcessedImage}>Get Processed Image</button>
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
      <button onClick={fetchData}>Load midpoints</button> {/* データ取得をトリガーするボタンを追加 */}
      <table>
        <thead>
          <tr>
            {/* CSVのヘッダーを動的に生成 */}
            {csvData[0] && Object.keys(csvData[0]).map((key) => <th key={key}>{key}</th>)}
          </tr>
        </thead>
        <tbody>
          {/* CSVの各行をテーブルの行として表示 */}
          {csvData.map((row, index) => (
            <tr key={index}>
              {Object.values(row).map((value, i) => <td key={i}>{value}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>

    <div>
      <button onClick={fetchCSVData}>Load final_plotted_points </button> {/* データ取得をトリガーするボタンを追加 */}
      <table>
        <thead>
          <tr>
            {/* CSVのヘッダーを動的に生成 */}
            {csvPlotData[0] && Object.keys(csvPlotData[0]).map((key) => <th key={key}>{key}</th>)}
          </tr>
        </thead>
        <tbody>
          {/* CSVの各行をテーブルの行として表示 */}
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