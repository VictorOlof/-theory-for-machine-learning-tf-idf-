import requests
from bs4 import BeautifulSoup


class BookScraper:

    def get_story(self, link):
        soup = self._scrape_website(link)
        story = []
        for p in soup.select("p"):
            stycke = p.get_text().split()
            for word in stycke:
                story.append(word)
        return story

    def _scrape_website(self, link):
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        # soup.prettify()
        return soup