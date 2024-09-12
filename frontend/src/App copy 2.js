import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [actionType, setActionType] = useState('');
  const [responseContent, setResponseContent] = useState('');

  // ファイルの選択状態を更新
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  // 選択されたアクションタイプに基づき、適切なリクエストを実行
  const executeAction = async () => {
    if (!file) {
      alert('Please upload a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    // actionTypeをFormDataに追加
    formData.append('action', actionType);

    try {
      // APIに画像ファイルとactionTypeを送信
      const response = await axios.post('http://localhost:5000/api', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResponseContent(JSON.stringify(response.data, null, 2));
    } catch (error) {
      console.error('Error executing action:', error);
      setResponseContent('Error executing action: ' + error.message);
    }
  };

  return (
    <div>
      <h1>React Frontend for Single Endpoint API</h1>

      {/* ファイルアップロードの入力フィールド */}
      <input type="file" onChange={handleFileChange} />
      
      {/* アクションタイプの選択 */}
      <select value={actionType} onChange={(e) => setActionType(e.target.value)}>
        <option value="">Select an Action</option>
        <option value="predict">Get Predicted Image</option>
        <option value="contours">Get Contours Image</option>
        <option value="histograms">Get Histograms</option>
        <option value="processed_image">Get Processed Image</option>
        <option value="download_csv">Download CSV</option>
        <option value="final_plotted_points">Get Final Plotted Points</option>
      </select>

      {/* アクション実行のトリガーボタン */}
      <button onClick={executeAction}>Execute</button>

      {/* APIレスポンスの表示 */}
      {responseContent && (
        <div>
          <h2>Response</h2>
          <pre>{responseContent}</pre>
        </div>
      )}
    </div>
  );
}

export default App;

