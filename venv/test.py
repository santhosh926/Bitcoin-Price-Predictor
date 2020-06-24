import requests
import json

# base URLs
globalURL = "https://api.coinmarketcap.com/v1/global/"
tickerURL = "https://api.coinmarketcap.com/v1/ticker/"

# get data
request = requests.get(globalURL)
data = request.json()

print()
choice = input("Enter a cryptocurrency to see its current price: ")

tickerURL += '/'+choice+'/'
last_price = 0

while True:
    request = requests.get(tickerURL)
    data = request.json()
    ticker = data[0]['symbol']
    price = data[0]['price_usd']
    if price != last_price:
        print(ticker + ":\t\t$" + price)
        last_price = price
        print()


