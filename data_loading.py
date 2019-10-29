import csv
from os.path import join
import glob
import codecs
import os
import io
import zipfile
import tarfile
import os.path
from os import path
import requests
import sys
import gzip
import shutil


class CorpusLoader:
    def __init__(self, subset="train", encoding='utf8', root="C:\Data"):

        self.subset = subset
        self.encoding = encoding
        self.root = root

        self.target = []
        self.data = []

        self.find_path()
        self.__load__()

        self.pathname = ""

    def find_path(self):
        if type(self) is AGNews:
            url = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"
            file_name = "ag_news_csv.tgz"
            self.pathname = "ag_news_csv"
            self.download(file_name, url)

        elif type(self) is AmazonReviewFull:
            url = "https://s3.amazonaws.com/fast-ai-nlp/amazon_review_full_csv.tgz"
            file_name = "amazon_review_full_csv.tgz"
            self.pathname = "amazon_review_full_csv"
            self.download(file_name, url)

        elif type(self) is AmazonReviewPolarity:
            url = "https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz"
            file_name = "amazon_review_polarity_csv.tgz"
            self.pathname = "amazon_review_polarity_csv"
            self.download(file_name, url)

        elif type(self) is Dbpedia:
            url = "https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz"
            file_name = "dbpedia_csv.tgz"
            self.pathname = "dbpedia_csv"
            self.download(file_name, url)

        elif type(self) is YelpReviewFull:
            url = "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz"
            file_name = "yelp_review_full_csv.tgz"
            self.pathname = "yelp_review_full_csv"
            self.download(file_name, url)

        elif type(self) is YelpReviewPolarity:
            url = "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz"
            file_name = "yelp_review_polarity_csv.tgz"
            self.pathname = "yelp_review_polarity_csv"
            self.download(file_name, url)

        elif type(self) is Yahoo:
            url = "https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz"
            file_name = "yahoo_answers_csv.tgz"
            self.pathname = "yahoo_answers_csv"
            self.download(file_name, url)

        elif type(self) is IMDB:
            url = "https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz"
            file_name = "imdb.tgz"
            self.pathname = "imdb"
            self.download(file_name, url)

        elif type(self) is WebKB:
            url = "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz"
            file_name = "webkb-data.gtar.gz"
            self.pathname = "webkb-data"
            self.download(file_name, url)

    def download(self, file_name, url):

        if path.exists(join(self.root, self.pathname)):
            return

        with open(file_name, "wb") as f:
            print("Downloading " + file_name)
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()
        self.extract(file_name)

    def extract(self, file):
            if file.endswith("zip"):
                zip_ = zipfile(file)
                new_path = join(self.root, self.pathname)
                zip_.extractall(path=new_path)
                os.remove(file)
            elif file.endswith("tgz" or "tar.gz" or "tar"):
                tar = tarfile.open(file, "r:gz")
                new_path = join(self.root, self.pathname)
                tar.extractall(path=new_path)
                tar.close()
                os.remove(file)
            #elif file.endswith("gz"):
             #   with gzip.open(file, 'rb') as gz:
              #     f = open(self.pathname, 'wb')
                    #shutil.copyfileobj(gz, f)
                #TODO

    def __load__(self):
        if self.subset == "test":
            self.test_loader()

        elif self.subset == "train":
            self.train_loader()

        elif self.subset == "all":
            self.all_loader()

    def test_loader(self):
        raise NotImplementedError()

    def train_loader(self):
        raise NotImplementedError()

    def all_loader(self):
        raise NotImplementedError()


class AGNews(CorpusLoader):

    def test_loader(self):
        reader = csv.reader(open(join(self.root, "ag_news_csv", "ag_news_csv", "test.csv")), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[2])

    def train_loader(self):
        reader = csv.reader(open(join(self.root, "ag_news_csv", "ag_news_csv", "train.csv")), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[2])

    def all_loader(self):
        self.test_loader()
        self.train_loader()


class AmazonReviewFull(CorpusLoader):
    def test_loader(self):
        reader = csv.reader(open(join(self.root, self.pathname, "amazon_review_full_csv", "test.csv"), encoding=self.encoding), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[1].replace("\"\"", "\""))

    def train_loader(self):
        reader = csv.reader(open(join(self.root, self.pathname, "amazon_review_full_csv", "train.csv"), encoding=self.encoding), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[1].replace("\"\"", "\""))

    def all_loader(self):
        self.test_loader()
        self.train_loader()


class AmazonReviewPolarity(CorpusLoader):
    def test_loader(self):
        reader = csv.reader(
            open(join(self.root, self.pathname, "amazon_review_polarity_csv", "test.csv"), encoding=self.encoding), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[1].replace("\"\"", "\""))

    def train_loader(self):
        reader = csv.reader(
            open(join(self.root, self.pathname, "amazon_review_polarity_csv", "train.csv"), encoding=self.encoding), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[1].replace("\"\"", "\""))

    def all_loader(self):
        self.test_loader()
        self.train_loader()


class Dbpedia(CorpusLoader):
    def test_loader(self):
        reader = csv.reader(
            open(join(self.root, self.pathname, "dbpedia_csv", "test.csv"), encoding=self.encoding),
            delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[2].replace("\"\"", "\""))

    def train_loader(self):
        reader = csv.reader(
            open(join(self.root, self.pathname, "dbpedia_csv", "train.csv"), encoding=self.encoding),
            delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[2].replace("\"\"", "\""))

    def all_loader(self):
        self.test_loader()
        self.train_loader()


class YelpReviewFull(CorpusLoader):

    def test_loader(self):
        reader = csv.reader(open(join(self.root, self.pathname, "yelp_review_full_csv", "test.csv"), encoding=self.encoding), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[1].replace("\"\"", "\""))

    def train_loader(self):
        reader = csv.reader(open(join(self.root, self.pathname, "yelp_review_full_csv", "train.csv"), encoding=self.encoding), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[1].replace("\"\"", "\""))

    def all_loader(self):
        self.test_loader()
        self.train_loader()


class YelpReviewPolarity(CorpusLoader):

    def test_loader(self):
        reader = csv.reader(open(join(self.root,self.pathname, self.pathname, "test.csv"), encoding=self.encoding), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[1].replace("\"\"", "\""))

    def train_loader(self):
        reader = csv.reader(open(join(self.root, self.pathname, self.pathname, "train.csv"), encoding=self.encoding), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[1].replace("\"\"", "\""))

    def all_loader(self):
        self.test_loader()
        self.train_loader()


class Yahoo(CorpusLoader):

    def test_loader(self):
        reader = csv.reader(open(join(self.root, self.pathname, self.pathname, "test.csv"), encoding=self.encoding), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[3].replace("\"\"", "\""))

    def train_loader(self):
        reader = csv.reader(open(join(self.root, self.pathname, self.pathname, "train.csv"), encoding=self.encoding), delimiter=',')
        for record in reader:
            self.target.append(record[0])
            self.data.append(record[3].replace("\"\"", "\""))

    def all_loader(self):
        self.test_loader()
        self.train_loader()


class IMDB(CorpusLoader):

    def test_loader(self):
        for label in ['pos', 'neg']:
            for files in glob.iglob(os.path.join(self.root, self.pathname, 'imdb', 'test', label, '*.txt')):
                with open(files, 'r', encoding="utf-8") as f:
                    text = f.read()
                    if label == "pos":
                        self.target.append(1)
                    else:
                        self.target.append(0)
                    self.data.append(text)

    def train_loader(self):
        for label in ['pos', 'neg']:
            for files in glob.iglob(os.path.join(self.root, self.pathname, 'imdb', 'train', label, '*.txt')):
                with open(files, 'r', encoding="utf-8") as f:
                    text = f.read()
                    if label == "pos":
                        self.target.append(1)
                    else:
                        self.target.append(0)
                    self.data.append(text)

    def all_loader(self):
        self.test_loader()
        self.train_loader()


class WebKB(CorpusLoader):

    def find_class(self, argument):
        if argument == "course":
            self.target.append(0)
        elif argument == "department":
            self.target.append(1)
        elif argument == "faculty":
            self.target.append(2)
        elif argument == "other":
            self.target.append(3)
        elif argument == "project":
            self.target.append(4)
        elif argument == "staff":
            self.target.append(5)
        elif argument == "student":
            self.target.append(6)

    def test_loader(self):
        for class_ in ['course', 'department', 'faculty', 'other', 'project', 'staff', 'student']:
            for label1 in ['cornell', 'misc', 'texas', 'washington']:
                for files in glob.iglob(os.path.join(self.root, self.pathname, class_, label1, '*')):
                    f = codecs.open(files, 'r', encoding=self.encoding, errors='ignore')
                    text = f.read()
                    self.find_class(class_)
                    self.data.append(text)

    def train_loader(self):
        for class_ in ['course', 'department', 'faculty', 'other', 'project', 'staff', 'student']:
                for files in glob.iglob(os.path.join(self.root, self.pathname, class_, 'wisconsin', '*')):
                    f = codecs.open(files, 'r', encoding="utf-8", errors='ignore')
                    text = f.read()
                    self.find_class(class_)
                    self.data.append(text)

    def all_loader(self):
        for class_ in ['course', 'department', 'faculty', 'other', 'project', 'staff', 'student']:
            for label in ['cornell', 'misc', 'texas', 'washington', 'wisconsin']:
                for files in glob.iglob(os.path.join(self.root, self.pathname, class_, label, '*')):
                    f = codecs.open(files, 'r', encoding="utf-8", errors='ignore')
                    text = f.read()
                    self.find_class(class_)
                    self.data.append(text)



