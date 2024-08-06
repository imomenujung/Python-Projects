# import relevant libraries
import requests 
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd

# lists to store data
all_titles = []
all_prices = []

# for loop to iterate over the range of page numbers you want to scrape from
# note: python includes the first argument and excludes the second argument in the range function
for page_number in range(1, 50):

    # define the url
    url = f"https://books.toscrape.com/catalogue/page-{page_number}.html"

    # send a request to get html code from that url
    response = requests.get(url, headers={"Accept": "text/html"}) 

    # parse the response
    parsed_response = BeautifulSoup(response.text, "html.parser") 

    titles = parsed_response.find_all("h3")
    prices  = parsed_response.find_all("p", class_= "price_color")

    # extract and clean titles and prices
    for title, price in zip(titles, prices):
        cleaned_title = ' '.join(title.a["title"].split())
        cleaned_price = ' '.join(price.text.split())
        all_titles.append(cleaned_title)
        all_prices.append(cleaned_price)

    # pause the program for 1 second between iterations of the loop
    sleep(1)

# create a DataFrame
books_df = pd.DataFrame({
    "Title": all_titles,
    "Price": all_prices
})

books_df.to_csv("books_data.csv", index=False)
