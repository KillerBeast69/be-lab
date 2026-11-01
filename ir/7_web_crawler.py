import requests          # downloads a web page from a URL
from bs4 import BeautifulSoup  # parses HTML into a searchable tree
import json              # saves results to a file

# Download the page HTML from a real site built for practice scraping
url = "https://books.toscrape.com/"
response = requests.get(url)  # fetch the page (same as a browser visiting it)
html = response.text          # get the raw HTML string

# Parse the HTML into a tree so we can query tags by class/id
soup = BeautifulSoup(html, "html.parser")

products = []  # empty list to collect extracted product data
# Find every <article class="product_pod"> block — each one is one product card
for card in soup.find_all("article", class_="product_pod"):
    # Extract the product name from the <h3> tag inside this card
    name = card.find("h3").get_text(strip=True)
    # Extract the price from the <p> tag with class="price_color"
    price = card.find("p", class_="price_color").get_text(strip=True)
    # Extract the URL from the <a> tag's href attribute
    link = card.find("a")["href"]
    # Store the extracted fields as a dictionary entry
    products.append({"name": name, "price": price, "link": link})

# Print each product's name, price, and link
for p in products:
    print(f"{p['name']:25} | {p['price']:10} | {p['link']}")

# Save all results to a JSON file for submission
with open("products.json", "w") as f:
    json.dump(products, f, indent=2)