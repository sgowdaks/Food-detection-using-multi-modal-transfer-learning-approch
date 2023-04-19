import csv

from icrawler.builtin import GoogleImageCrawler
from icrawler import ImageDownloader
from PIL import Image
from six import BytesIO
from pathlib import Path
import time
import os

root = "/home/sg/work/scrapy_selenium/left_out/names.tsv"

class KeywordGoogleImageCrawler(GoogleImageCrawler):

    def crawl(self,
              keyword,
              filters=None,
              offset=0,
              max_num=1000,
              min_size=None,
              max_size=None,
              language=None,
              file_idx_offset=0,
              overwrite=False):
        if offset + max_num > 1000:
            if offset > 1000:
                self.logger.error(
                    '"Offset" cannot exceed 1000, otherwise you will get '
                    'duplicated searching results.')
                return
            elif max_num > 1000:
                max_num = 1000 - offset
                self.logger.warning(
                    'Due to Google\'s limitation, you can only get the first '
                    '1000 result. "max_num" has been automatically set to %d. '
                    'If you really want to get more than 1000 results, you '
                    'can specify different date ranges.', 1000 - offset)

        feeder_kwargs = dict(
            keyword=keyword,
            offset=offset,
            max_num=max_num,
            language=language,
            filters=filters)
        downloader_kwargs = dict(
            keyword=keyword,  #<<< add this line
            max_num=max_num,
            min_size=min_size,
            max_size=max_size,
            file_idx_offset=file_idx_offset,
            overwrite=overwrite)
        super(GoogleImageCrawler, self).crawl(
            feeder_kwargs=feeder_kwargs, downloader_kwargs=downloader_kwargs)


class KeywordNameDownloader(ImageDownloader):
        
    def get_filename(self, task, default_ext, keyword):
        filename = super(KeywordNameDownloader, self).get_filename(
            task, default_ext)
        return keyword + filename

    def keep_file(self, task, response, min_size=None, max_size=None, **kwargs):
        """Decide whether to keep the image

        Compare image size with ``min_size`` and ``max_size`` to decide.

        Args:
            response (Response): response of requests.
            min_size (tuple or None): minimum size of required images.
            max_size (tuple or None): maximum size of required images.
        Returns:
            bool: whether to keep the image.
        """
        try:
            img = Image.open(BytesIO(response.content))
        except (IOError, OSError):
            return False
        task['img_size'] = img.size
        if min_size and not self._size_gt(img.size, min_size):
            return False
        if max_size and not self._size_lt(img.size, max_size):
            return False
        return True

    def download(self,
                task,
                default_ext,
                timeout=5,
                max_retry=3,
                overwrite=False,
                **kwargs):
        """Download the image and save it to the corresponding path.

        Args:
            task (dict): The task dict got from ``task_queue``.
            timeout (int): Timeout of making requests for downloading images.
            max_retry (int): the max retry times if the request fails.
            **kwargs: reserved arguments for overriding.
        """
        file_url = task['file_url']
        task['success'] = False
        task['filename'] = None
        retry = max_retry
        keyword = kwargs['keyword']


        if not overwrite:
            with self.lock:
                self.fetched_num += 1
                filename = self.get_filename(task, default_ext,keyword)
                if self.storage.exists(filename):
                    self.logger.info('skip downloading file %s', filename)
                    return
                self.fetched_num -= 1

        while retry > 0 and not self.signal.get('reach_max_num'):
            try:
                response = self.session.get(file_url, timeout=timeout)
            except Exception as e:
                self.logger.error('Exception caught when downloading file %s, '
                                'error: %s, remaining retry times: %d',
                                file_url, e, retry - 1)
            else:
                if self.reach_max_num():
                    self.signal.set(reach_max_num=True)
                    break
                elif response.status_code != 200:
                    self.logger.error('Response status code %d, file %s',
                                    response.status_code, file_url)
                    break
                elif not self.keep_file(task, response, **kwargs):
                    break
                with self.lock:
                    self.fetched_num += 1
                    filename = self.get_filename(task, default_ext,keyword)
                self.logger.info('image #%s\t%s', self.fetched_num, file_url)
                self.storage.write(filename, response.content)
                task['success'] = True
                task['filename'] = filename
                break
            finally:
                retry -= 1

with open("/mnt/e/sg/scrapy_selenium/piza_categories.tsv", "r") as output:
    csv_output = csv.reader(output, delimiter="\t")
    for item in csv_output:
        google_crawler = KeywordGoogleImageCrawler(downloader_cls=KeywordNameDownloader, storage={'root_dir': '/home/sg/work/scrapy_selenium/pizza_images'})
        item = item[0].replace(" ", "_")
        google_crawler.crawl(keyword=item, max_num=5)  


