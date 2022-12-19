import re
import heapq
import hashlib
import cld3
import nltk
import pyspark.sql.functions as F
from functools import partial
from model import KenlmModel
from pyspark.sql import SparkSession
import pandas as pd

from pyspark import SparkContext

SEPERATOR = "###img###sep###"
nltk_download = False

# TODO - clean this up
if nltk_download:
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
    x, tokenize_sentences_func, ngrams_func, ngrams_filter_func, perplexity_filter_func
):
    x = list(list(x)[0])

    candidates = []
    for text in get_before_after_text(x[1]):
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

    # Create hash of image url to deduplicate
    url_hash = hashlib.md5(x[0].encode()).hexdigest()

    yield (url_hash, x[0], candidates)


def local_session(num_cores=4, mem_gb=16):
    """Build a local spark session"""
    spark = (
        SparkSession.builder.config("spark.driver.memory", str(mem_gb) + "G")
        .master("local[" + str(num_cores) + "]")
        .appName("image_text_pairs")
        .getOrCreate()
    )
    return spark


def get_filtered_captions(
    ngram_range=(3, 20),
    tokenize_sentences=tokenize_sentences,
    ngrams_filter=entity_filter,
    perplexity_filter_func=perplexity_filter,
    lang_to_perplexity_models={},
    n_largest=10,
):

    spark = local_session(num_cores=16, mem_gb=32)

    filename = "/home/siddhesh1793/data/bild/00000_url_to_text.parquet"
    image_link_to_surrounding_text = get_image_link_to_surrounding_text(filename)

    sc = SparkContext.getOrCreate()
    num_rows = image_link_to_surrounding_text.shape[0]
    data = [(tup[1], tup[2]) for tup in image_link_to_surrounding_text.itertuples()]
    image_to_text_rdd = sc.parallelize(data, num_rows)

    ngrams_func = partial(generate_n_grams, ngram_range=ngram_range)
    perp_func = partial(
        perplexity_filter_func, models=lang_to_perplexity_models, n_largest=n_largest
    )
    link_processing_func = partial(
        image_link_to_caption_candidates,
        tokenize_sentences_func=tokenize_sentences,
        ngrams_func=ngrams_func,
        ngrams_filter_func=ngrams_filter,
        perplexity_filter_func=perp_func,
    )

    # Create image to captions df
    image_to_candidate_caps_rdd = image_to_text_rdd.mapPartitions(link_processing_func)

    # Filter if no candidate captions
    image_to_candidate_caps_rdd = image_to_candidate_caps_rdd.filter(lambda x : len(x[2]) > 0)

    df = image_to_candidate_caps_rdd.toDF(['uid', 'url', 'candidates'])

    # Group by uid
    agg_candidates = df.groupBy(["uid"]).agg(F.flatten(F.collect_list("candidates")).alias("candidates"))

    df = df.join(agg_candidates, "uid", "inner").drop(df.candidates).drop_duplicates(["uid"])

    # Group by uid
    import pdb

    pdb.set_trace()


if __name__ == "__main__":

    lang_to_perp_model = {"en": KenlmModel.from_pretrained("wikipedia", "en")}

    get_filtered_captions(lang_to_perplexity_models=lang_to_perp_model)
