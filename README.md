
Explanation of Myntra Hackathon - Gujarat Tigers Project

Overview
The project is developed for the Myntra Hackathon by the Gujarat Tigers team. It focuses on creating a chatbot and trend forecasting system using NLP and machine learning techniques.

Main Components

1. `modelTrain` Class
This class is responsible for:
- Loading intents from a JSON file.
- Preprocessing data: tokenizing words, lemmatizing, and creating documents.
- Saving processed data (words and classes) to files using pickle.
- Preparing training data (input and output arrays).
- Creating and training a neural network model using TensorFlow/Keras.

Methods
- `loadIntents(intents_path)`: Loads intents from a specified JSON file.
- `preprocess_save_Data(intents)`: Processes and saves words and classes.
- `prepareTrainingData(words, classes)`: Prepares training data.
- `createModel(train_x, train_y, epochs, batch_size, save_path)`: Creates and trains a model.

2. `modelPredict` Class
This class handles:
- Loading the trained model.
- Cleaning up and tokenizing input sentences.
- Predicting the class of the input sentence.
- Generating responses based on the predicted class.
- Correcting spelling of the user input using TextBlob.

Methods
- `clean_up_sentence(sentence)`: Tokenizes and lemmatizes input sentences.
- `bow(sentence, words)`: Converts sentences into a bag-of-words representation.
- `predict_class(sentence, model, error_threshold)`: Predicts the class of the input sentence.
- `getResponse(ints, intents_json)`: Retrieves a response based on the predicted class.
- `chatbot_response(msg)`: Generates a response from the chatbot.
- `correct_spelling(sentence)`: Corrects spelling of the input sentence.

3. Flask App
The Flask app provides endpoints for interacting with the chatbot and forecasting trends.
- Endpoints**:
  - `/`: Renders the homepage.
  - `/chat`: Handles chatbot interactions.
  - `/forecast`: Fetches trend data, prepares it for forecasting, generates forecasts, and scrapes relevant images.

Important Functions
- `fetch_trends_data(keyword, timeframe)`: Fetches Google Trends data.
- `prepare_data_for_forecast(trends_data, keyword)`: Prepares data for forecasting.
- `forecast_trend(data, periods)`: Uses NeuralProphet to forecast trends.
- `plot_forecast(data, forecast, keyword)`: Plots and encodes the forecasted trend data.
- `scrape_images(keyword, num_images)`: Scrapes images from Google Images.

Running the Project
1. **Training the Model**:
   - Initialize and preprocess data using `modelTrain`.
   - Train and save the model.
2. **Running the Flask App**:
   - Start the Flask application to interact with the chatbot and forecasting system.

Additional Components
- Integration with PyNgrok for easy local testing.
- Image generation using Stable Diffusion and CLIP models.
