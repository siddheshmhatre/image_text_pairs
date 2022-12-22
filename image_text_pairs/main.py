import re
import os
import heapq
import fsspec
import hashlib
import random
import cld3
import nltk
import datetime
import pandas as pd
import pyspark.sql.functions as F

from functools import partial
from .model import KenlmModel
from pyspark.sql import SparkSession
from multiprocessing.pool import ThreadPool
from pyspark import SparkContext
from fastwarc import ArchiveIterator
from io import BytesIO
from resiliparse.parse import detect_encoding
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text
from urllib.parse import urljoin
from collections import OrderedDict
from loguru import logger
from timeit import default_timer as timer

SEPERATOR = "###img###sep###"


def dowload_nltk():
    nltk.download("averaged_perceptron_tagger")
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def get_before_after_text(text):
    sep_span = re.search(SEPERATOR, text).span()

    # Remove urls and email ids - see pycld3 README
    url_re = r"\b(?:https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+(?:[/?].*)?"
    before_text = re.sub(url_re, "", text[: sep_span[0]].strip())
    after_text = re.sub(url_re, "", text[sep_span[1] :].strip())

    return [before_text, after_text]


def get_image_link_to_surrounding_text(filename):
    df = pd.read_parquet(filename)
    return df


def tokenize_sentences(text, language):
    if language == "en":
        sent_tokenizer = nltk.data.load("tokenizers/punkt/PY3/english.pickle")
        return sent_tokenizer.tokenize(text)
    else:
        return [text]


def generate_n_grams(candidates, ngram_range):
    from nltk import ngrams

    n_grams = []
    for i in range(len(candidates)):
        for n in range(*ngram_range):
            for item in ngrams(candidates[i].split(), n):
                item = " ".join(item)
                n_grams.append(item)

    return n_grams


def entity_filter(ngrams, language):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import wordnet as wn

    filtered_candidates = []

    for item in ngrams:
        if language == "en":
            word_tokens = word_tokenize(item)
            adj_present = False
            verb_or_noun_present = False

            for word in word_tokens:
                wordtype = set()
                for tmp in wn.synsets(word):
                    if tmp.name().split(".")[0] == word:
                        wordtype.add(tmp.pos())

                if "a" in wordtype or "s" in wordtype:
                    adj_present = True

                if "n" in wordtype or "v" in wordtype:
                    verb_or_noun_present = True

                if adj_present and verb_or_noun_present:
                    filtered_candidates.append(item)
                    break
        else:
            filtered_candidates.append(item)

    return filtered_candidates


def perplexity_filter(ngrams, language, models, n_largest):
    perp_filtered_candidates = []

    model = models.get(language, None)

    if model:
        for candidate in ngrams:
            perp_filtered_candidates.append(
                (candidate, model.get_perplexity(candidate))
            )

        top_n_candidates = heapq.nlargest(
            n_largest, perp_filtered_candidates, key=lambda x: -x[1]
        )

        return [candidate[0] for candidate in top_n_candidates]

    else:
        return []


def image_link_to_caption_candidates(
    before_text,
    after_text,
    tokenize_sentences_func,
    ngrams_func,
    ngrams_filter_func,
    perplexity_filter_func,
):
    candidates = []
    for text in [before_text, after_text]:
        # Detect language
        language = cld3.get_language(text)

        if language is not None:
            language = language.language
            # Tokenize sentences
            sentences = tokenize_sentences_func(text, language)

            # Generate n-grams
            n_grams = ngrams_func(sentences)

            # Filter based on noun or adjective if english
            filtered_n_grams = ngrams_filter_func(n_grams, language)

            # Filter based on perplexity
            filtered_n_grams = perplexity_filter_func(filtered_n_grams, language)

            candidates.extend(filtered_n_grams)

    return candidates


def local_session(num_cores=4, mem_gb=16):
    """Build a local spark session"""
    spark = (
        SparkSession.builder.config("spark.driver.memory", str(mem_gb) + "G")
        .master("local[" + str(num_cores) + "]")
        .appName("image_text_pairs")
        .getOrCreate()
    )
    return spark


def get_date_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_cc_warc_links(source_cc_protocol):

    if source_cc_protocol == "http":
        fs, p = fsspec.core.url_to_fs("https://commoncrawl.org/the-data/get-started/")
        a = fs.open(p).read()
        l = a.splitlines()
        l = [e.decode("utf8").replace("[WARC] ", "") for e in l]
        l = [e for e in l if "<li>s3://commoncrawl/crawl-data/" in e]
        l = [
            e.split(" ")[0]
            .replace("<li>s3://commoncrawl/", "https://data.commoncrawl.org/")
            .replace("<wbr>", "")
            for e in l
        ]
        l = [(e + "/warc.paths.gz").replace("//warc", "/warc") for e in l]
        return l
    elif source_cc_protocol == "s3":
        fs, p = fsspec.core.url_to_fs("s3://commoncrawl/crawl-data/")
        links = ["s3://" + e for e in fs.glob(p + "/*/warc.paths.gz")]
        return links


def read_warc_index_file(warc_index):
    with fsspec.open(warc_index, "rb", compression="gzip") as f:
        warcs = [a.decode("utf8").strip() for a in f.readlines()]

    return warcs


def read_warc_index_files(
    shard_count=None, warc_count=1000, source_cc_protocol="http"
):
    """Read all warc index files"""
    # Taken from https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L170
    cc_warc_links = get_cc_warc_links(source_cc_protocol)
    if shard_count is not None:
        cc_warc_links = cc_warc_links[
            -shard_count:
        ]  # pylint: disable=invalid-unary-operand-type
    all_warcs = []
    with ThreadPool(16) as pool:
        for wats in pool.imap_unordered(read_warc_index_file, cc_warc_links):
            all_warcs.extend(wats)
    if warc_count is not None:
        all_warcs = random.choices(all_warcs, k=warc_count)
    else:
        # shuffle to increase duplication over each part hence reduce size of each part after duplication
        random.shuffle(all_warcs)

    if source_cc_protocol == "http":
        prefix = "https://data.commoncrawl.org/"
    elif source_cc_protocol == "s3":
        prefix = "s3://commoncrawl/"

    all_warcs = [prefix + warc_link for warc_link in all_warcs]

    return all_warcs


def get_images_and_surrounding_text(text, images_to_url, candidate_generation_func):
    # Get start and end indices of every image tag in text
    # Hack think more about this
    image_to_idxs = OrderedDict()
    for image_name in images_to_url.keys():
        if image_name in text:
            image_to_idxs[image_name] = re.search(image_name, text).span()

    last_end = 0
    image_links_and_surrounding_text = []
    # For every image
    for idx, (img_name, img_span) in enumerate(image_to_idxs.items()):

        # get text before and text after image
        start, end = img_span

        before_text = text[last_end:start]
        last_end = end

        if idx == (len(image_to_idxs) - 1):
            after_text = text[end:]
        else:
            next_img_name = list(image_to_idxs.keys())[idx + 1]
            after_text = text[end : image_to_idxs[next_img_name][0]]

        url = images_to_url[img_name]

        image_links_and_surrounding_text.append((url, before_text, after_text))

    for image_url, before_text, after_text in image_links_and_surrounding_text:
        # Create hash of image url to deduplicate
        url_hash = hashlib.md5(image_url.encode()).hexdigest()

        yield (url_hash, image_url, candidate_generation_func(before_text, after_text))


def process_warc_record(html_bytes, url, candidate_generation_func):
    # Refer - https://github.com/siddheshmhatre/Big-Interleaved-Dataset/blob/optimize_script/bild/extraction_utils.py#L10
    encoding = detect_encoding(html_bytes)
    tree = HTMLTree.parse_from_bytes(html_bytes, encoding)
    image_count = 0
    images_to_url = {}

    for ele in tree.body.get_elements_by_tag_name("nav"):
        ele.parent.remove_child(ele)

    # Get all image links and surrounding text
    for ele in tree.body.get_elements_by_tag_name("img"):
        csrc = ele.getattr("src")
        images_to_url[f"###img###{image_count}###"] = urljoin(url, csrc)
        ele.setattr("alt", f"###img###{image_count}###")
        image_count += 1

    text = extract_plain_text(
        tree,
        preserve_formatting=False,
        main_content=False,
        list_bullets=False,
        alt_texts=True,
        links=False,
        form_fields=False,
        noscript=False,
    )

    for url_hash, image_url, candidates in get_images_and_surrounding_text(
        text, images_to_url, candidate_generation_func
    ):
        yield url_hash, image_url, candidates


def process_warc(x, logging_frequency, candidate_generation_func):
    # Refer - https://github.com/siddheshmhatre/Big-Interleaved-Dataset/blob/optimize_script/bild/pipeline_utils.py#L9
    x = list(x)

    warc_url = x[0]

    start = timer()

    records_processed = 0

    # Iterate through each record
    with fsspec.open(warc_url, "rb") as f:
        stream = BytesIO(f.read())
        for record in ArchiveIterator(stream, max_content_length=4 * 1024**2):
            try:
                if record.headers is None:
                    continue
                if record.http_headers is None:
                    continue
                if (
                    record.headers["WARC-Type"] == "response"
                    and record.content_length >= 128
                ):
                    content_type = str(record.http_content_type).lower()

                    if content_type.startswith("text/html"):

                        records_processed += 1

                        if (records_processed % logging_frequency) == 0:
                            logger.info(f"Processing record {records_processed}")

                        url = str(record.headers["WARC-Target-URI"])
                        html_bytes = record.reader.read()
                        for url_hash, image_url, candidates in process_warc_record(
                            html_bytes, url, candidate_generation_func
                        ):
                            yield url_hash, image_url, candidates

            except Exception as e:
                logger.info(e)

    end = timer()
    logger.info(f"Time to proces one WARC : {end - start}")


def process_one_part(
    output_path,
    warc_index_files,
    logging_frequency,
    ngram_range=(3, 20),
    tokenize_sentences=tokenize_sentences,
    ngrams_filter=entity_filter,
    perplexity_filter_func=perplexity_filter,
    lang_to_perplexity_models={"en": KenlmModel.from_pretrained("wikipedia", "en")},
    n_largest=10,
):
    # Create output path
    job_id = get_date_str()
    output_path = os.path.join(output_path, job_id)

    # Create spark session
    spark = local_session(num_cores=16, mem_gb=32)

    # Create spark context
    sc = SparkContext.getOrCreate()

    # Extract image links and candidate captions from warc index files
    warc_index_files_rdd = sc.parallelize(warc_index_files, len(warc_index_files))

    ngrams_func = partial(generate_n_grams, ngram_range=ngram_range)
    perp_func = partial(
        perplexity_filter_func, models=lang_to_perplexity_models, n_largest=n_largest
    )
    candidate_generation_func = partial(
        image_link_to_caption_candidates,
        tokenize_sentences_func=tokenize_sentences,
        ngrams_func=ngrams_func,
        ngrams_filter_func=ngrams_filter,
        perplexity_filter_func=perp_func,
    )
    process_warc_function = partial(
        process_warc,
        logging_frequency=logging_frequency,
        candidate_generation_func=candidate_generation_func,
    )

    # Create image to captions df
    image_to_candidate_caps_rdd = warc_index_files_rdd.mapPartitions(
        process_warc_function
    )

    # Filter if no candidate captions
    image_to_candidate_caps_rdd = image_to_candidate_caps_rdd.filter(
        lambda x: len(x[2]) > 0
    )

    # Convert to df
    df = image_to_candidate_caps_rdd.toDF(["uid", "url", "candidates"])

    # Groupby by url
    agg_candidates = df.groupBy(["url"]).agg(
        F.flatten(F.collect_list("candidates")).alias("candidates")
    )

    df = (
        df.join(agg_candidates, "url", "inner")
        .drop(df.candidates)
        .drop_duplicates(["uid"])
    )

    logger.info(f"Writing to {output_path}")

    df = df.repartition(max(256, len(warc_index_files)))

    # Write to disk
    df.write.mode("overwrite").parquet(output_path)


def image_text_pairs(
    output_path,
    num_shards=None,
    num_warcs=None,
    source_cc_protocol="http",
    download_nltk_models=False,
    logging_frequency=1000,
):

    if download_nltk_models:
        dowload_nltk()

    start = timer()

    # Read in all warc index files from cc
    warc_index_files = read_warc_index_files(
        num_shards, num_warcs, source_cc_protocol=source_cc_protocol
    )

    logger.info(f"Processing {len(warc_index_files)} warcs")

    # Create map from url to potential captions
    process_one_part(output_path, warc_index_files, logging_frequency=logging_frequency)

    end = timer()

    logger.info(f"{num_warcs} took {end - start}")
