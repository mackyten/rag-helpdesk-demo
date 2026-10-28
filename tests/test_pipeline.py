from ragdemo.chunking import Chunk
from ragdemo.generate import ClaudeGenerator, ExtractiveGenerator, REFUSED, SYSTEM_PROMPT, format_sources
from ragdemo.pipeline import ABSTAIN, build_pipeline


class FixedGenerator:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def generate(self, question, sources):
        self.calls += 1
        return self.text


def test_answer_cites_real_sources():
    answer = build_pipeline().ask("After how many wrong sign-in attempts is my account locked?")

    assert not answer.abstained
    assert answer.citations and answer.citations[0].doc_id == "account-security"
    assert "5 failed" in answer.answer
    assert answer.ungrounded_citations == []


def test_pipeline_abstains_without_calling_the_generator_when_nothing_matches():
    generator = FixedGenerator("should never be used")

    answer = build_pipeline(generator=generator).ask("Who won the football world cup in 2022?")

    assert answer.abstained and answer.answer == ABSTAIN
    assert generator.calls == 0


def test_hallucinated_citation_numbers_are_flagged():
    generator = FixedGenerator("Refunds take 30 days [1] and also [9].")

    answer = build_pipeline(generator=generator).ask("How many days do I have to ask for a refund?")

    assert answer.ungrounded_citations == [9]
    assert [c.number for c in answer.citations] == [1]


def test_model_saying_it_does_not_know_counts_as_abstaining():
    generator = FixedGenerator("I don't know based on the documentation.")

    answer = build_pipeline(generator=generator).ask("How many days do I have to ask for a refund?")

    assert answer.abstained and answer.citations == []


def test_extractive_generator_prefers_the_sentence_that_overlaps_the_question():
    chunk = Chunk("d#0", "d", "H", "H\nUnrelated sentence here. Shipping is free above 75 dollars.")

    text = ExtractiveGenerator(max_sentences=1).generate("Is shipping free?", [chunk])

    assert text == "Shipping is free above 75 dollars. [1]"


class FakeMessage:
    def __init__(self, text, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.content = [type("Block", (), {"type": "text", "text": text})()]


class FakeClient:
    """Records the request so the Claude request shape can be asserted without a network call."""

    def __init__(self, message):
        self.calls = []
        outer = self

        class Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return message

        self.beta = type("Beta", (), {"messages": Messages()})()


def test_claude_generator_sends_grounded_prompt_and_fallbacks():
    client = FakeClient(FakeMessage("30 days [1]"))
    sources = [Chunk("refunds#0", "refunds", "Refund Policy", "Refund Policy\n30-day guarantee.")]

    text = ClaudeGenerator(client=client, model="claude-opus-5-5").generate("How long?", sources)

    request = client.calls[0]
    assert text == "30 days [1]"
    assert request["system"] == SYSTEM_PROMPT
    assert request["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in request["betas"]
    assert format_sources(sources) in request["messages"][0]["content"]
    assert "How long?" in request["messages"][0]["content"]


def test_claude_generator_returns_a_safe_message_on_refusal():
    client = FakeClient(FakeMessage("", stop_reason="refusal"))

    assert ClaudeGenerator(client=client).generate("q", []) == REFUSED
