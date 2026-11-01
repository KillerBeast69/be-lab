import requests          # downloads web pages (like a browser fetching a URL)
from bs4 import BeautifulSoup  # parses HTML into a searchable tree

# ---- STEP 1: Download a single page ----
url = "https://books.toscrape.com/"   # a real site built for practice scraping
response = requests.get(url)          # fetch the page HTML
html = response.text                  # get the raw HTML as a string

# ---- STEP 2: Parse the HTML ----
soup = BeautifulSoup(html, "html.parser")
# html.parser = Python's built-in HTML reader
# soup is now a tree of all HTML tags — you can query it like a DOM

# ---- STEP 3: Find every product card ----
products = []                          # empty list to collect results
for card in soup.find_all("article", class_="product_pod"):
    # Each <article class="product_pod"> block = one product
    # find_all returns ALL matching elements (not just the first)

    name = card.find("h3").get_text(strip=True)
    # find("h3") = grab the first <h3> tag inside this card
    # get_text(strip=True) = extract the text inside, removing extra spaces

    price = card.find("p", class_="price_color").get_text(strip=True)
    # look for <p> with class="price_color" — that's where the price lives

    link = card.find("a")["href"]
    # grab the <a> tag's href attribute (the URL it points to)

    products.append({"name": name, "price": price, "link": link})
    # save this product's data into our list

# ---- STEP 4: Print / save results ----
for p in products:
    print(f"{p['name']:25} | {p['price']:10} | {p['link']}")

# Optional: save to a JSON file
import json
with open("products.json", "w") as f:
    json.dump(products, f, indent=2)