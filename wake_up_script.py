import os
import requests
from bs4 import BeautifulSoup

def wake_up_app(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        wake_up_button = soup.find('button', string=lambda text: 'Yes, get this app back up!' in text if text else False)
        
        if wake_up_button:
            print("App is sleeping. Attempting to wake it up...")
            wake_up_url = url + wake_up_button.get('data-url', '')
            requests.get(wake_up_url)
            print("Wake-up request sent.")
        else:
            print("App is already awake or wake-up button not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app_url = os.environ.get('APP_URL')
    if app_url:
        wake_up_app(app_url)
    else:
        print("APP_URL environment variable not set.")
