#FORECAST
from flask import Flask, request, render_template, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from neuralprophet import NeuralProphet
from pytrends.request import TrendReq
from pyngrok import ngrok
import time
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO as PILBytesIO

app = Flask(__name__, template_folder='templates', static_folder='static')
pytrends = TrendReq(hl='en-US', tz=360)

# Function to fetch Google Trends data for a given keyword
def fetch_trends_data(keyword, timeframe='today 5-y'):
    try:
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
        data = pytrends.interest_over_time()

        if data.empty:
            print("No data returned for this keyword.")
            return None

        if 'isPartial' in data.columns:
            data = data.drop(columns=['isPartial'])

        data.reset_index(inplace=True)
        return data

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Waiting for 10 seconds before retrying...")
        time.sleep(10)
        return fetch_trends_data(keyword, timeframe)

# Prepare the data for NeuralProphet
def prepare_data_for_forecast(trends_data, keyword):
    data = trends_data[['date', keyword]].dropna()
    data.columns = ['ds', 'y']
    return data

# Forecast function
def forecast_trend(data, periods=1825):  # 5 years = 1825 days
    model = NeuralProphet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(data, freq='D')
    future = model.make_future_dataframe(data, periods=periods)
    forecast = model.predict(future)
    return forecast

# Plotting function
def plot_forecast(data, forecast, keyword):
    plt.figure(figsize=(12, 8))
    plt.plot(data['ds'], data['y'], label='Historical', color='blue')
    plt.plot(forecast['ds'], forecast['yhat1'], label='Forecast', color='orange')

    if 'yhat1_lower' in forecast.columns and 'yhat1_upper' in forecast.columns:
        plt.fill_between(forecast['ds'], forecast['yhat1_lower'], forecast['yhat1_upper'], color='orange', alpha=0.2)

    plt.xlabel('Date')
    plt.ylabel('Interest Over Time')
    plt.title(f'Forecast for {keyword}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

# Function to scrape images from Google Images
def scrape_images(keyword, num_images=5):
    query = '+'.join(keyword.split())
    url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img', limit=num_images + 1)  # +1 to account for the first image which is usually the Google logo

    img_urls = []
    for img_tag in img_tags[1:]:  # Skip the first image
        img_url = img_tag.get('src')
        if img_url and img_url.startswith('http'):
            img_urls.append(img_url)
            if len(img_urls) >= num_images:
                break
    return img_urls

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    keyword = request.form['keyword']
    trends_data = fetch_trends_data(keyword)

    if trends_data is not None and keyword in trends_data.columns:
        data = prepare_data_for_forecast(trends_data, keyword)
        forecast = forecast_trend(data)
        plot_url = plot_forecast(data, forecast, keyword)

        # Scrape images
        image_urls = scrape_images(keyword)

        return jsonify({
            'plot_url': plot_url,
            'image_urls': image_urls
        })
    else:
        return jsonify({'error': f'No data found for the keyword: {keyword}'}), 404

if __name__ == '__main__':
    ngrok.set_auth_token("NGROK-TOKEN")
    public_url = ngrok.connect(5000)
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:5000\"".format(public_url))
    app.run()
