import os
import re

import asks
import trio
from lxml import etree
from lxml import html


class LinkExtractor:
    def __init__(self, domain: str):
        self.domain = domain
        self.parser = etree.HTMLParser()

    async def extract_links(self, url: str, visit_url, queue_url):
        try:
            print("Try:", url)
            response = await asks.get(url)
            print("Got:", url)

            content = response.content.decode('utf-8')
            tree = html.fromstring(content)
            # tree = etree.parse(io.StringIO(content), self.parser) # Would parse in XML
            visit_url(url, tree)
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
        extractor = LinkExtractor(self.domain)
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


class BlogPostExtractor:
    def __init__(self, domain: str, blog_post_regex: str):
        self.crawler = Crawler(domain=domain, blog_post_regex=blog_post_regex)

    def extract(self, first_url: str, folder: str, file_prefix: str):
        trio.run(self.crawler.collect_links, first_url)
        print("Visited", len(self.crawler.visited), "links")
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, (url, tree) in enumerate(self.crawler.visited.items()):
            file_name = "posts/" + file_prefix + str(i) + ".html"
            content = self.get_blog_content(tree)
            if content is not None:
                with open(file_name, 'w', encoding='utf-8') as file:
                    file.write(content)

    def get_blog_content(self, tree):
        for node in tree.xpath("//article/div"):
            if "post-content" in node.attrib["class"]:
                # TODO - it would be nice to try to keep some of the formatting
                return node.text_content()


extractor = BlogPostExtractor(domain="https://deque.blog", blog_post_regex="https://deque.blog/\d{4}/\d{2}/\d{2}.*")
extractor.extract(first_url="https://deque.blog/posts", folder="posts", file_prefix="deque-blog-")

# TODO - add a rate limiter using a Semaphore (it is not nice on the web sites right now)

# extractor = BlogPostExtractor(domain="https://www.fluentcpp.com", blog_post_regex="https://www.fluentcpp.com/\d{4}/\d{2}/\d{2}.*")
# extractor.extract(first_url="https://www.fluentcpp.com/posts/", folder="posts", file_prefix="fluentcpp-")
