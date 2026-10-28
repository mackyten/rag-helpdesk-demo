from ragdemo.bm25 import BM25Index, tokenize
from ragdemo.chunking import load_corpus, split_markdown
from ragdemo.pipeline import DEFAULT_DOCS


def test_split_by_heading_keeps_heading_in_text():
    chunks = split_markdown("doc", "# Title\n\nintro\n\n## Part\n\nbody text")

    assert [c.heading for c in chunks] == ["Title", "Part"]
    assert chunks[1].text == "Part\nbody text"
    assert chunks[1].id == "doc#1"


def test_long_section_is_split_on_paragraph_boundaries():
    paragraph = " ".join(["word"] * 80)
    chunks = split_markdown("doc", f"# T\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}", max_words=100)

    assert len(chunks) == 3
    assert all(len(c.text.split()) <= 82 for c in chunks)


def test_tokenize_drops_stopwords_and_single_letters_and_stems_plurals():
    assert tokenize("What's the refunds policy?") == ["refund", "policy"]


def test_load_corpus_reads_every_sample_document():
    docs = {c.doc_id for c in load_corpus(DEFAULT_DOCS)}

    assert {"refunds", "shipping", "warranty", "account-security", "billing-and-plans", "data-privacy"} <= docs


def test_bm25_ranks_the_matching_chunk_first_and_ignores_nonmatches():
    index = BM25Index(load_corpus(DEFAULT_DOCS))

    top, score = index.search("how long is the warranty", k=1)[0]

    assert top.doc_id == "warranty" and score > 0
    assert index.search("zzzzqqqq xylophone") == []
