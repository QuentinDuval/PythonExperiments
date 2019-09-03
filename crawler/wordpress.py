import asks
from io import StringIO
from lxml import etree
import os
import re
import trio


class Extractor:
    def __init__(self, domain: str):
        self.domain = domain
        self.parser = etree.HTMLParser()

    async def extract_links(self, url: str, visit_url, queue_url):
        response = await asks.get(url)
        try:
            content = response.content.decode('utf-8')
            visit_url(url, content)
            tree = etree.parse(StringIO(content), self.parser)
            for href in tree.xpath('//a/@href'):
                url = self.clean_href(href)
                if url is not None:
                    queue_url(url)
        except Exception as e:
            print("Failed to parse content of", url, "due to", e)

    def clean_href(self, href):
        if not href.startswith(self.domain):
            return None
        href = href.split("#")[0]
        href = href.split("?")[0]
        return href


class Crawler:
    def __init__(self, domain: str, blog_post_regex: str):
        self.domain = domain
        self.queue = []
        self.discovered = set()
        self.visited = {}
        self.blog_post_regex = re.compile(blog_post_regex)

    async def collect_links(self, start_url: str):
        self.add_url(start_url)
        extractor = Extractor(self.domain)
        while self.queue:
            # TODO - this is a BFS by design... try a DFS by using channels
            async with trio.open_nursery() as nursery:
                while self.queue:
                    url = self.queue.pop()
                    nursery.start_soon(extractor.extract_links, url, self.add_blog, self.add_url)

    def add_blog(self, url, content):
        if self.blog_post_regex.match(url):
            self.visited[url] = content

    def add_url(self, url):
        if url not in self.discovered:
            self.discovered.add(url)
            self.queue.append(url)


crawler = Crawler("https://deque.blog", "https://deque.blog/\d{4}/\d{2}/\d{2}.*")
trio.run(crawler.collect_links, "https://deque.blog/posts")

print("Visited", len(crawler.visited), "links")

if not os.path.exists("posts"):
    os.makedirs("posts")

for i, (url, content) in enumerate(crawler.visited.items()):
    file_name = "posts/deque-blog-" + str(i) + ".html"
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(content)

