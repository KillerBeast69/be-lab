from bs4 import BeautifulSoup
import requests

url = "http://books.toscrape.com/index.html"
response = requests.get(url).text
soup = BeautifulSoup(response, "html.parser")

for product in soup.find_all("article", class_="product_pod"):
    title = product.h3.a["title"]
    price = product.find("p", class_="price_color").text
    link = product.h3.a["href"]
    print(f"Title: {title} | Price: {price} | Link: {link}")

    