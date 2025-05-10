import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

base_url = "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/"
response = requests.get(base_url)
html_content = response.text

soup = BeautifulSoup(html_content, 'html.parser')

# Find all links starting with the base URL
links = soup.find_all('a', href=True)
urls = [urljoin(base_url, link['href']) for link in links if link['href'].startswith(base_url)]

# Include the base URL itself
urls.append(base_url)

with open('sports.txt', 'w', encoding='utf-8') as file:
    seen_lines = set()
    for url in urls:
        response = requests.get(url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        main_content = soup.find('main', class_='content--primary')
        if main_content:
            divs = main_content.find_all('div')
            for div in divs:
                text = div.get_text(separator=' ', strip=True)
                if text not in seen_lines:
                    file.write(text + '\n')
                    seen_lines.add(text)
