import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def fetch_events(url, retries=5, delay=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise Exception(f"Failed to load page {url} after {retries} attempts")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    events = []

    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string)
            title = data.get('name', '')
            date_start = data.get('startDate', '')
            if date_start:
                date_start = datetime.strptime(date_start, '%Y-%m-%dT%H:%M:%S%z')
                location_data = data.get('location', {}).get('address', {})
                location = f"{data.get('location', {}).get('name', '')}, {location_data.get('streetAddress', '')}, {location_data.get('addressLocality', '')}, {location_data.get('addressRegion', '')}, {location_data.get('postalCode', '')}"
                description = data.get('description', '')
                
                # Filter events happening after March 19th
                if date_start > datetime(2025, 3, 19, tzinfo=date_start.tzinfo):
                    events.append({
                        'date': date_start.strftime('%Y-%m-%d'),
                        'time': date_start.strftime('%H:%M:%S'),
                        'title': title,
                        'location': location,
                        'description': description
                    })
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
        except Exception as e:
            print(f"Failed to extract event: {e}")
    
    return events

def fetch_events_new_website(url, retries=5, delay=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise Exception(f"Failed to load page {url} after {retries} attempts")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    events = []

    # Extract JSON data from script tags
    for script in soup.find_all('script', type='application/ld+json'):
        data = json.loads(script.string)
        title = data.get('name', '')
        date_start = data.get('startDate', '')
        if date_start:
            date_start = datetime.strptime(date_start, '%Y-%m-%d')
            location = data.get('location', {}).get('address', '')
            description = data.get('description', '')
            
            # Filter events happening after March 19th
            if date_start > datetime(2025, 3, 19):
                events.append({
                    'date': date_start.strftime('%Y-%m-%d'),
                    'time': '',
                    'title': title,
                    'location': location,
                    'description': description
                })
    
    return events

def fetch_events_from_html(url, retries=5, delay=10):
    options = Options()
    options.headless = True
    service = Service('E:/chromedriver-win64/chromedriver.exe')  # Update this path to your chromedriver
    driver = webdriver.Chrome(service=service, options=options)
    
    for i in range(retries):
        try:
            driver.get(url)
            time.sleep(delay)  # Wait for JavaScript to load content
            break
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                driver.quit()
                raise Exception(f"Failed to load page {url} after {retries} attempts")
    
    # Click through all "Show 50 more" buttons
    while True:
        try:
            show_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'lw_cal_next'))
            )
            show_more_button.click()
            time.sleep(delay)  # Wait for more events to load
        except Exception as e:
            break
    
    # Scrape all events after loading all pages
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    
    events = []
    
    # Find all h3 tags and their corresponding div tags
    h3_tags = soup.find_all('h3')
    
    for h3_tag in h3_tags:
        event_date = h3_tag.get_text(strip=True)
        events_div = h3_tag.find_next_sibling('div')
        
        if events_div:
            for event_div in events_div.find_all('div', class_='lw_cal_event'):
                event_title = event_div.find('div', class_='lw_events_title')
                event_location = event_div.find('div', class_='lw_events_location')
                event_time = event_div.find('div', class_='lw_events_time')
                event_summary = event_div.find('div', class_='lw_events_summary')
                
                if event_title:
                    events.append({
                        'date': event_date,
                        'time': event_time.get_text(strip=True) if event_time else '',
                        'title': event_title.get_text(strip=True),
                        'location': event_location.get_text(strip=True) if event_location else '',
                        'description': event_summary.get_text(strip=True) if event_summary else ''
                    })
    
    return events

def fetch_events_from_pghcitypaper(url, retries=5, delay=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise Exception(f"Failed to load page {url} after {retries} attempts")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    events = []

    for event_div in soup.find_all('div', class_='fdn-pres-content'):
        try:
            title_tag = event_div.find('p', class_='fdn-teaser-headline')
            date_time_tag = event_div.find('p', class_='fdn-teaser-subheadline')
            location_name_tag = event_div.find('a', class_='fdn-event-teaser-location-link')
            location_address_tag = event_div.find('p', class_='fdn-event-teaser-location-address')
            description_tag = event_div.find('div', class_='fdn-teaser-description')

            if not title_tag:
                continue
            
            title = title_tag.get_text(strip=True) if title_tag else ''
            date_time = date_time_tag.get_text(strip=True) if date_time_tag else ''
            location_name = location_name_tag.get_text(strip=True) if location_name_tag else ''
            location_address = location_address_tag.get_text(strip=True) if location_address_tag else ''
            location = f"{location_name}, {location_address}"
            description = description_tag.get_text(strip=True).replace('\n', ' ') if description_tag else ''
            
            # Filter events happening after March 19th
            events.append({
                    'date': date_time.split(' ')[0] if date_time else '',
                    'time': ' '.join(date_time.split(' ')[1:]) if date_time else '',
                    'title': title,
                    'location': location,
                    'description': description
                })
        except Exception as e:
            print(f"Failed to extract event: {e}")
    
    return events

def fetch_events_from_cmu(url):
    response = requests.get(url)
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')

    start_class = 'grid column3 darkgrey boxes js-list'
    end_class = 'footer-grid'

    start_div = soup.find('div', class_=start_class)
    end_div = soup.find('div', id=end_class)

    events = []

    if start_div and end_div:
        divs = start_div.find_all_next('div')
        for div in divs[:-2]:  # Stop before the second last div
            if div == end_div:
                break
            text = div.get_text(separator=' ', strip=True)
            events.append({
                'date': '',
                'time': '',
                'title': text,
                'location': '',
                'description': ''
            })
    
    return events

def remove_duplicates(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    unique_lines = list(set(lines))
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(unique_lines)

def main():
    all_events = []

    # Fetch events from the first website
    month = ['march','april','may','june','july','august','september','october','november','december']
    for i in month:
        url = "https://pittsburgh.events/" + i
        try:
            events = fetch_events(url)
            all_events.extend(events)
        except Exception as e:
            print(f"Failed to fetch events for {i}: {e}")
    
    # Fetch events from the new website
    base_url = "https://downtownpittsburgh.com/events/?n={}&y=2025&cat=0"
    months = range(3, 13)  # March to December
    for month in months:
        url = base_url.format(month)
        try:
            events = fetch_events_new_website(url)
            all_events.extend(events)
        except Exception as e:
            print(f"Failed to fetch events for month {month}: {e}")
    
    # Fetch events from the HTML website
    url = "https://events.cmu.edu/all"
    try:
        events = fetch_events_from_html(url)
        all_events.extend(events)
    except Exception as e:
        print(f"Failed to fetch events from HTML website: {e}")
    
    # Fetch events from PGH City Paper
    first_url = "https://www.pghcitypaper.com/pittsburgh/EventSearch?narrowByDate=2025-03-19&sortType=date&v=d"
    urls = [first_url] + [f"https://www.pghcitypaper.com/pittsburgh/EventSearch?narrowByDate=2025-03-19&page={i}&sortType=date&v=d" for i in range(2, 6)]
    for url in urls:
        try:
            events = fetch_events_from_pghcitypaper(url)
            all_events.extend(events)
        except Exception as e:
            print(f"Failed to fetch events from PGH City Paper: {e}")
    
    # Fetch events from CMU
    url = "https://www.cmu.edu/engage/alumni/events/campus/index.html"
    try:
        events = fetch_events_from_cmu(url)
        all_events.extend(events)
    except Exception as e:
        print(f"Failed to fetch events from CMU: {e}")

    # Remove duplicates
    unique_events = [dict(t) for t in {tuple(d.items()) for d in all_events}]

    # Write all events to the file
    with open('events.txt', 'w', encoding='utf-8') as file:  # Open file in write mode with utf-8 encoding
        for event in unique_events:
            file.write(f"{event['date']} {event['time']} {event['title']} {event['location']} {event['description']}\n")

if __name__ == "__main__":
    main()









