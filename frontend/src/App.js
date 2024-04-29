import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [date, setDate] = useState('');
    const [county, setCounty] = useState('');
    const [prediction, setPrediction] = useState('');

    const handleSubmit = async (event) => {
      event.preventDefault();
      const url = 'http://localhost:5002/predict'; // make sure this matches the correct port Flask server is running on
      const data = { date, county };
      const options = {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        data: JSON.stringify(data),
        url,
      };
      axios(options)
        .then(response => {
          setPrediction(response.data.prediction); // Update state to display the prediction
          console.log(response.data); // Optional: for debugging
        })
        .catch(error => console.error('Error:', error));
    };

    return (
        <div>
          <h1>Snow Day Closure Predictor</h1>
          <form onSubmit={handleSubmit}>
            <label>
              Date:
              <input type="date" value={date} onChange={e => setDate(e.target.value)} required />
            </label>
            <br />
            <label>
              County:
              <input type="text" value={county} onChange={e => setCounty(e.target.value)} required />
            </label>
            <br />

            <button type="submit">Predict</button>
          </form>
          {prediction && <p>{prediction}</p>}
        </div>
    );
}

export default App;
