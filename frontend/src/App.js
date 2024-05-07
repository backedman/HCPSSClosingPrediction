import React, { useState } from 'react';
import axios from 'axios';
import styles from "./App.module.css";

const App = () => {
  const [date, setDate] = useState('');
  const [prediction, setPrediction] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    const url = 'http://localhost:5000/predict';
    const data = {date};
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
        setPrediction(response.data.prediction);
      })
      .catch(error => console.error('Error:', error));
  };

  return (
    <div className={styles.cover}>
      <img className={styles.backgroundIcon} alt="" src="/background.svg" />
      <section className={styles.background} id="background" />
      <div className={styles.snowDayPredictorWrapper}>
        <h3 className={styles.snowDayPredictor} id="header">
          Snow Day Predictor
        </h3>
      </div>
      <div className={styles.resultsOutputtextWrapper}>
        <h3 className={styles.snowDayPredictor}>Results: {prediction}</h3>
      </div>
      <form onSubmit={handleSubmit} className={styles.form}>
        <label className={styles.snowDateWrapper}>
          <div className={styles.snowDate}>Snow Date</div>
          <input
            className={styles.coverChild}
            type="date"
            value={date}
            onChange={e => setDate(e.target.value)}
            required
          />
        </label>
        <button className={styles.inputWrapper} type="submit">
          <div className={styles.input}>Predict</div>
        </button>
      </form>
      <img className={styles.snow3Icon} alt="" src="/snow1Icon.png" />
      <img className={styles.cloudComputing1Icon} alt="" src="/cloudComputing1Icon.png" />
      <img className={styles.cloudComputing4Icon} alt="" src="/cloudComputing4Icon.png" />
      <img className={styles.cloudComputing2Icon} alt="" src="/cloudComputing2Icon.png" />
      <img className={styles.cloudComputing3Icon} alt="" src="/cloudComputing3Icon.png" />
      <img className={styles.snow1Icon} alt="" src="/snow1Icon.png" />
      <img className={styles.snow2Icon} alt="" src="/snow2Icon.png" />
    </div>
  );
};

export default App;

