import os
import re

import asks
from asks.sessions import Session
import datetime
import trio
from lxml import etree
from lxml import html


"""
New version
"""

'''
class UrlWorker:
    def __init__(self, domain: str, input_channel, content_out_channel, url_out_channel):
        self.domain = domain
        self.parser = etree.HTMLParser()
        self.input_channel = input_channel
        self.content_out_channel = content_out_channel
        self.url_out_channel = url_out_channel

    async def consume(self):
        async with self.input_channel:
            async for url in self.input_channel:
                await self.extract_links(url)

    async def extract_links(self, url: str):
        try:
            response = await asks.get(url)
            content = response.content.decode('utf-8')
            tree = html.fromstring(content)
            self.content_out_channel.send((url, tree))
            for href in tree.xpath('//a/@href'):
                url = self.clean_href(href)
                if url is not None:
                    self.url_out_channel.send(url)
        except Exception as e:
            print("Failed to parse content of", url, "due to", e)

    def clean_href(self, href):
        if not href.startswith(self.domain):
            return None
        href = href.split("#")[0]
        href = href.split("?")[0]
        return href


class UrlCrawler:
    def __init__(self, domain: str, blog_post_regex: str, max_connection: int):
        self.domain = domain
        self.queue = []
        self.discovered = set()
        self.visited = {}
        self.blog_post_regex = re.compile(blog_post_regex)
        self.max_connection = max_connection

    async def collect_links(self, start_url: str):
        self.add_url_link(start_url)

        url_in_channel, url_out_channel = trio.open_memory_channel(0)

        async with trio.open_nursery() as nursery:
            for _ in range(self.max_connection):
                worker = UrlWorker(self.domain, input_channel, content_out_channel, url_out_channel)
                nursery.start_soon(worker.consume)

        while self.queue:
            # TODO - this is a BFS by design... try a DFS by using channels
            async with trio.open_nursery() as nursery:
                while self.queue:
                    url = self.queue.pop()
                    nursery.start_soon(self.extractor.extract_links, self.session, url, self.on_visited, self.add_url_link)

    def on_visited(self, url, content):
        if self.blog_post_regex.match(url):
            self.visited[url] = content

    def add_url_link(self, url):
        if url not in self.discovered:
            self.discovered.add(url)
            self.queue.append(url)
'''

"""
Old version using BFS
"""


class UrlExtractor:
    def __init__(self, domain: str):
        self.domain = domain
        self.parser = etree.HTMLParser()

    async def extract_links(self, session: Session, url: str, on_visit_url, add_url_to_queue):
        try:
            response = await session.get(url)
            content = response.content.decode('utf-8')
            tree = html.fromstring(content)
            on_visit_url(url, tree)
            for href in tree.xpath('//a/@href'):
                url = self.clean_href(href)
                if url is not None:
                    add_url_to_queue(url)
        except Exception as e:
            print("Failed to parse content of", url, "due to", e)

    def clean_href(self, href):
        if not href.startswith(self.domain):
            return None
        href = href.split("#")[0]
        href = href.split("?")[0]
        return href


class Crawler:
    def __init__(self, domain: str, blog_post_regex, max_connection: int):
        self.domain = domain
        self.queue = []
        self.discovered = set()
        self.visited = {}
        self.blog_post_regex = blog_post_regex
        self.session = Session(self.domain, connections=max_connection)
        self.extractor = UrlExtractor(self.domain)

    async def collect_links(self, start_url: str):
        self.add_url_link(start_url)
        while self.queue:
            async with trio.open_nursery() as nursery:
                while self.queue:
                    url = self.queue.pop()
                    nursery.start_soon(self.extractor.extract_links, self.session, url, self.on_visited, self.add_url_link)

    def on_visited(self, url, content):
        print("[{when}] Visited {url}".format(url=url, when=datetime.datetime.now()))
        if self.blog_post_regex.match(url):
            self.visited[url] = content

    def add_url_link(self, url):
        if url not in self.discovered:
            self.discovered.add(url)
            self.queue.append(url)


class BlogPostExtractor:
    def __init__(self, domain: str, blog_post_regex: str):
        self.blog_post_regex = re.compile(blog_post_regex)
        self.crawler = Crawler(domain=domain, blog_post_regex=self.blog_post_regex, max_connection=10)

    def extract(self, first_url: str, folder: str):
        trio.run(self.crawler.collect_links, first_url)
        print("Visited", len(self.crawler.visited), "links")
        self.dump_result(folder)

    def dump_result(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for url, tree in self.crawler.visited.items():
            file_name = self.get_file_name(url)
            content = self.get_blog_content(tree)
            full_file_path = folder + "/" + file_name + ".html"
            if content is not None:
                with open(full_file_path, 'w', encoding='utf-8') as file:
                    file.write(content)

    def get_file_name(self, url):
        m = self.blog_post_regex.match(url)
        article_name = m.group(1)
        return article_name.strip("/")

    def get_blog_content(self, tree):
        for article in tree.xpath("//article"):
            return article.text_content()
        '''
        title = ""
        content = ""
        for article in tree.xpath("//article"):
            for node in article:
                if node.tag == "header" and "post-header" in node.attrib.get("class", ""):
                    title = node.text_content()
                if node.tag == "div" and "post-content" in node.attrib.get("class", ""):
                    content = node.text_content()
        return title + "\n\n" + content
        '''


# extractor = BlogPostExtractor(domain="https://deque.blog", blog_post_regex="https://deque.blog/\d{4}/\d{2}/\d{2}/(.*)")
# extractor.extract(first_url="https://deque.blog/posts", folder="posts/deque")

# extractor = BlogPostExtractor(domain="https://www.fluentcpp.com", blog_post_regex="https://www.fluentcpp.com/\d{4}/\d{2}/\d{2}/(.*)")
# extractor.extract(first_url="https://www.fluentcpp.com/posts/", folder="posts/fluentcpp")
