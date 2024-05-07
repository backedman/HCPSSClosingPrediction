## Snow Day Prediction App

A very cool and awesome app 

React.js frontend,  Flask backend

### To run Flask server

First time: install required Flask libraries
```
cd backend
pip install -r requirements.txt 
```

To run the server
```
cd backend
flask run
```

The server is currently set to run on port 5002. If you want to change this, modify in the `app.py` file.

If the port is changed, make sure to change the port in `App.js` (in frontend/src) as well since this is the port the server is running on.

Server will be hosted locally at `http://localhost:5002` or `http://localhost:<port>` if modified.

### To run React.js App

```
cd frontend
npm start
```

App will be hosted locally at `http://localhost:3000`
