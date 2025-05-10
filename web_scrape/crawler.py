"""
This file is the crawler for the assignment 2.
You can provide the initial urls and the max depth and the max pages per url.

The crawler will crawl the urls and save the text data to the text_data directory.
And it will save the metadata to the text_metadata.json file in the current directory.

The crawler will also save the visited urls to the visited_urls.json file.
"""



import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import List, Set, Dict
from tqdm import tqdm
import PyPDF2
import io
import json
from pathlib import Path
from collections import deque
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_directories():
    """Create necessary data directories"""
    directories = ['text_data']  # Change to text_data directory
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

class WikiCrawler:
    def __init__(self, 
                 initial_urls: List[str],
                 max_depth: int = 1,
                 max_pages_per_url: int = 10):
        """
        Initialize crawler with constraints and configurations
        
        Args:
            initial_urls: List of starting URLs to crawl
            max_depth: Maximum crawling depth from each initial URL
            max_pages_per_url: Maximum pages to crawl from each initial URL
            
        The crawler is configured with:
        - Allowed domains for crawling
        - Content quality thresholds
        - URL and content validation rules
        - Metadata tracking
        """
        self.initial_urls = initial_urls
        self.max_depth = max_depth
        self.max_pages_per_url = max_pages_per_url
        
        self.visited_urls = set()
        self.pages_per_initial_url = {url: 0 for url in initial_urls}
        
        # Track processed data size for each initial URL
        self.processed_data_size = {url: 0 for url in initial_urls}
        
        # List of allowed domains for crawling
        self.allowed_domains = [
            'wikipedia.org', 
            'cmu.edu', 
            'pittsburghpa.gov',
            'britannica.com',
            'visitpittsburgh.com'
        ]
        
        # Content quality thresholds
        self.min_word_count = 100  # Minimum words required in content
        self.max_newline_ratio = 0.5  # Maximum ratio of newlines to content length
        
    def download_and_process_html(self, url: str, process_data_dir: str) -> tuple:
        """Download webpage HTML content and directly process to text"""
        try:
            # Disable SSL verification
            response = requests.get(url, verify=False)
            response.raise_for_status()
            
            # Extract text and links
            text, links = self.extract_text_and_links(response.text, url)
            
            # Validate content quality
            if not self.validate_content(text):
                return None, None, 0
                
            # Clean and process text
            text = self.clean_text(text)
            
            # Generate filename from full URL
            filename = url.replace('://', '_').replace('/', '_').replace('?', '_').replace('&', '_')
            filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
            filename = f"{filename}.txt"
            
            # Save processed text
            filepath = os.path.join(process_data_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
                
            # Calculate file size in MB
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                
            return filename, links, file_size_mb
        
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None, None, 0

    def download_and_process_pdf(self, url: str, process_data_dir: str) -> tuple:
        """Download and process PDF content directly to text"""
        try:
            # Download PDF with SSL verification disabled
            response = requests.get(url, verify=False)
            response.raise_for_status()
            
            # Extract text from PDF
            pdf_text = ""
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() + "\n"
            
            # Clean and process text
            pdf_text = self.clean_text(pdf_text)
            
            # Validate content quality
            if not self.validate_content(pdf_text):
                return None, 0
            
            # Generate filename from full URL
            filename = url.replace('://', '_').replace('/', '_').replace('?', '_').replace('&', '_')
            filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
            filename = f"{filename}.txt"
            
            # Save processed text
            filepath = os.path.join(process_data_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(pdf_text)
            
            # Calculate file size in MB
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            return filename, file_size_mb
            
        except Exception as e:
            print(f"Error processing PDF {url}: {e}")
            return None, 0

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    def extract_text_and_links(self, html_content: str, base_url: str) -> tuple:
        """
        Extract text and relevant links from HTML content
        
        Args:
            html_content: Raw HTML content to process
            base_url: Base URL for resolving relative links
            
        Returns:
            tuple: (extracted text, list of relevant links)
            
        Features:
        - Removes unwanted HTML elements
        - Extracts clean text content
        - Filters and validates links based on domain rules
        - Smart relevancy checking based on URL and content
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        links = []
        relevant_keywords = ['pittsburgh', 'carnegie mellon university', 'cmu']
        
        def is_relevant_by_url(url: str) -> bool:
            """Check if URL itself contains relevant keywords"""
            url_lower = url.lower()
            return any(keyword in url_lower for keyword in relevant_keywords)
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            
            # Handle different URL patterns based on the base URL
            if 'wikipedia.org' in base_url:
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = urljoin("https://en.wikipedia.org", href)
                    # If URL contains keywords, add directly
                    if is_relevant_by_url(full_url):
                        links.append(full_url)
                    else:
                        # Otherwise check link text
                        link_text = a.get_text().lower()
                        if any(keyword in link_text for keyword in relevant_keywords):
                            links.append(full_url)
                        
            elif 'cmu.edu' in base_url:
                if not href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                    full_url = urljoin(base_url, href)
                    if 'cmu.edu' in full_url:  # Only include CMU domain links
                        # CMU domain is already relevant, no need for keyword check
                        links.append(full_url)
                        
            elif 'pittsburghpa.gov' in base_url:
                if not href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                    full_url = urljoin(base_url, href)
                    if 'pittsburghpa.gov' in full_url:  # Pittsburgh gov domain is relevant
                        links.append(full_url)
                        
            else:
                if not href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                    full_url = urljoin(base_url, href)
                    base_domain = urlparse(base_url).netloc
                    link_domain = urlparse(full_url).netloc
                    
                    if base_domain == link_domain:
                        # If URL contains keywords, add directly
                        if is_relevant_by_url(full_url):
                            links.append(full_url)
                        else:
                            # Otherwise check link text
                            link_text = a.get_text().lower()
                            if any(keyword in link_text for keyword in relevant_keywords):
                                links.append(full_url)
        
        return text, links

    def is_valid_url(self, url: str) -> bool:
        """
        Validate if a URL should be crawled based on multiple criteria
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL passes all validation checks, False otherwise
            
        Validation checks:
        1. File extension not in blocked list (images, scripts, etc.)
        2. Domain is in allowed list
        3. URL path not in excluded patterns (login, search, etc.)
        """
        parsed = urlparse(url)
        
        # Check file extensions
        invalid_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.css', '.js',
            '.xml', '.rss', '.zip', '.tar', '.gz'
        }
        if any(parsed.path.lower().endswith(ext) for ext in invalid_extensions):
            return False
            
        # Validate domain against allowlist
        if not any(domain in parsed.netloc for domain in self.allowed_domains):
            return False
            
        # Check for invalid URL patterns
        invalid_patterns = [
            '/action/', '/special:', '/category:', 
            '/file:', '/template:', '/help:', 
            'login', 'logout', 'register', 'search'
        ]
        if any(pattern in parsed.path.lower() for pattern in invalid_patterns):
            return False
            
        return True
        
    def validate_content(self, text: str) -> bool:
        """
        Validate content quality using multiple metrics
        
        Args:
            text: Extracted text content to validate
            
        Returns:
            bool: True if content meets quality standards, False otherwise
            
        Quality checks:
        1. Minimum word count
        2. Newline ratio within limits
        3. Paragraph structure
        4. Complete sentences count
        """
        # Check text length
        words = text.split()
        if len(words) < self.min_word_count:
            print(f"Content too short: {len(words)} words")
            return False
            
        # Check newline ratio
        newline_ratio = text.count('\n') / len(text) if text else 1
        if newline_ratio > self.max_newline_ratio:
            print(f"Too many newlines: ratio {newline_ratio:.2f}")
            return False
                        
        return True

    def crawl_url_bfs(self, start_url: str):
        """
        Perform breadth-first crawling starting from a given URL
        
        Args:
            start_url: Initial URL to start crawling from
        """
        queue = deque([(start_url, 0)])
        domain = urlparse(start_url).netloc
        
        while queue:
            url, depth = queue.popleft()
            
            # URL validation
            if not self.is_valid_url(url):
                continue
                
            if (depth > self.max_depth or 
                url in self.visited_urls or 
                self.pages_per_initial_url[start_url] >= self.max_pages_per_url):
                continue
            
            self.visited_urls.add(url)
            
            # Print current status with domain and count
            print(f"Crawling: {start_url} | Pages: {self.pages_per_initial_url[start_url]} | URL: {url} | Depth: {depth}")
            
            if url.lower().endswith('.pdf'):
                filename, file_size_mb = self.download_and_process_pdf(
                    url, 'text_data'
                )
                if filename:
                    self.pages_per_initial_url[start_url] += 1
                    self.processed_data_size[start_url] += file_size_mb
                continue
            
            filename, links, file_size_mb = self.download_and_process_html(
                url, 'text_data'
            )
            
            if filename:
                self.pages_per_initial_url[start_url] += 1
                self.processed_data_size[start_url] += file_size_mb
                
                if depth < self.max_depth:
                    for link in links:
                        if link not in self.visited_urls:
                            queue.append((link, depth + 1))

    def load_metadata(self) -> Dict:
        """
        Load existing metadata from file if it exists
        
        Returns:
            Dict containing previously saved metadata or empty structure if file doesn't exist
        """
        if os.path.exists('text_metadata.json'):
            try:
                with open('text_metadata.json', 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    print(f"Loaded existing metadata")
                    
                    # Update crawler state from metadata
                    for url in self.initial_urls:
                        if url in metadata['initial_urls']:
                            # Get previously crawled pages count
                            self.pages_per_initial_url[url] = metadata['initial_urls'][url]['subpages_crawled']
                            
                            # Get previously processed data size
                            if 'size_mb' in metadata['initial_urls'][url]['processed_data']:
                                self.processed_data_size[url] = metadata['initial_urls'][url]['processed_data']['size_mb']
                    
                    return metadata
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        # Return empty metadata structure if file doesn't exist or has errors
        return {
            'crawl_statistics': {
                'max_depth_allowed': self.max_depth,
                'max_pages_per_url_allowed': self.max_pages_per_url
            },
            'initial_urls': {},
            'total_statistics': {
                'total_pages': 0,
                'total_processed_size_mb': 0,
                'total_files': 0
            }
        }

    def save_metadata(self):
        """Save metadata about the crawling process"""
        # Try to load existing metadata first
        metadata = self.load_metadata()
        
        # Update crawl statistics
        metadata['crawl_statistics'] = {
            'max_depth_allowed': self.max_depth,
            'max_pages_per_url_allowed': self.max_pages_per_url
        }
        
        # 获取text_data目录中所有的txt文件
        processed_files = list(Path('text_data').glob('*.txt'))
        
        # 为每个初始URL更新元数据条目
        total_pages = 0
        for url in self.initial_urls:
            # 使用已经记录的数据
            pages_crawled = self.pages_per_initial_url[url]
            total_pages += pages_crawled
            
            # 获取域名，用于在文件名中匹配
            url_domain = urlparse(url).netloc
            
            # 找出属于这个初始URL的文件
            url_files = []
            for file_path in processed_files:
                file_name = file_path.name
                # 检查文件名中是否包含该URL的域名
                if url_domain in file_name:
                    url_files.append(file_name)
            
            # 更新元数据
            metadata['initial_urls'][url] = {
                'subpages_crawled': pages_crawled,
                'processed_data': {
                    'size_mb': round(self.processed_data_size[url], 2),
                    'file_count': len(url_files)
                }
            }
        
        # 更新总体统计
        total_processed_size = sum(self.processed_data_size.values())
        metadata['total_statistics'] = {
            'total_pages': total_pages,
            'total_processed_size_mb': round(total_processed_size, 2),
            'total_files': len(processed_files)
        }
        
        # 保存更新后的元数据
        with open('text_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # 保存已访问URL列表
        with open('visited_urls.json', 'w', encoding='utf-8') as f:
            json.dump(list(self.visited_urls), f, indent=2)
        
        print("\nMetadata saved to text_metadata.json")
        print(f"Visited URLs ({len(self.visited_urls)}) saved to visited_urls.json")

    def load_visited_urls(self) -> Set[str]:
        """
        Load previously visited URLs from file if it exists
        
        Returns:
            Set of previously visited URLs
        """
        if os.path.exists('visited_urls.json'):
            try:
                with open('visited_urls.json', 'r', encoding='utf-8') as f:
                    visited_urls = set(json.load(f))
                    print(f"Loaded {len(visited_urls)} previously visited URLs")
                    return visited_urls
            except Exception as e:
                print(f"Error loading visited URLs: {e}")
        
        print("No previous visited URLs found")
        return set()

    def crawl_all(self):
        """Crawl all initial URLs with progress tracking"""
        create_directories()
        
        # Load previously visited URLs
        previous_urls = self.load_visited_urls()
        if previous_urls:
            self.visited_urls = previous_urls
        
        # Load existing metadata
        self.load_metadata()
        
        for url in self.initial_urls:
            print(f"\n{'='*50}")
            print(f"Starting crawl from: {url}")
            print(f"{'='*50}")
            print(f"Already crawled: {self.pages_per_initial_url[url]} pages")
            
            # Only crawl if we haven't reached the limit
            if self.pages_per_initial_url[url] < self.max_pages_per_url:
                self.crawl_url_bfs(url)
            else:
                print(f"Skipping {url}: Already reached max pages limit")
            
            print(f"Completed {url}: {self.pages_per_initial_url[url]} pages crawled")
        
        print(f"\nCrawling completed:")
        total_pages = sum(self.pages_per_initial_url.values())
        print(f"Total pages crawled: {total_pages}")
        for url, count in self.pages_per_initial_url.items():
            print(f"Pages from {urlparse(url).netloc}: {count}")
        
        # Save metadata after crawling
        self.save_metadata()

# Example usage
if __name__ == "__main__":
    initial_urls = [
        "https://en.wikipedia.org/wiki/Pittsburgh",
        "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
        "https://www.pittsburghpa.gov/Home",
        "https://www.britannica.com/place/Pittsburgh", 
        "https://www.visitpittsburgh.com/",
        "https://www.pittsburghpa.gov/City-Government/Finances-Budget/Taxes/Tax-Forms", 
        "https://www.cmu.edu/about/",
        "https://apps.pittsburghpa.gov/redtail/images/23255_2024_Operating_Budget.pdf"
    ]
    
    crawler = WikiCrawler(
        initial_urls=initial_urls,
        max_depth=4,
        max_pages_per_url=2000
    )
    
    crawler.crawl_all() 