"""The FANToM adapter: presence from leaves and returns, the official question blocks and scoring."""

import hashlib
import io
import json
import random
import tarfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import bench.fantom as fantom
from baselines.full_context import FullContextMemory
from bench.runner import CostMeter
from bench.fantom import (
    Accessibility,
    FantomAnswer,
    FantomSet,
    QuestionKind,
    absence_spans,
    asked_about,
    belief_choice_prompt,
    binary_value,
    choice_is_correct,
    choose_conversations,
    extract_dataset,
    is_correct,
    is_well_formed,
    list_is_correct,
    presence,
    presence_agreement,
    set_questions,
    speakers_of,
    summarize_answers,
    to_conversation,
    to_conversations,
    weighted_f1,
)
from magma.embed import Matrix, Vector, normalize
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient, Usage
from magma.router import KeywordClassifier
from magma.temporal import DateparserTagger

LINES = [
    "Ana: Hi Bia, hi Caio.",
    "Bia: Hi Ana.",
    "Caio: I'll grab a coffee, be right back.",
    "Bia: Bye Caio.",
    "Ana: I adopted a dog, Rex.",
    "Bia: Nice!",
    "Caio: I'm back! What's up?",
    "Dan: Hey all, I just arrived.",
    "Ana, Bia & Caio: Welcome, Dan!",
]


def record() -> dict[str, object]:
    return {
        "set_id": "0-0-0",
        "part_id": "0-0",
        "conv_id": 0,
        "full_context": "\n".join(LINES),
        "short_context": "\n".join(LINES[2:7]),
        "missed_info": "Ana told Bia she adopted a dog, Rex.",
        "joining_speaker": "Caio",
        "factQA": {"question": "What did Ana adopt?", "question_type": "fact", "correct_answer": "A dog named Rex.", "wrong_answer": "A cat."},
        "beliefQAs": [
            {
                "question": "What does Caio believe about what Ana adopted?",
                "question_type": "tom:belief:inaccessible",
                "tom_type": "first-order",
                "correct_answer": "Caio does not know what Ana adopted.",
                "wrong_answer": "Caio believes Ana adopted a dog named Rex.",
                "missed_info_accessibility": "inaccessible",
            }
        ],
        "infoAccessibilityQA_list": {
            "question": "List all the characters who know this information.",
            "question_type": "tom:info_accessibility:list",
            "correct_answer": ["Ana", "Bia"],
            "wrong_answer": ["Caio", "Dan"],
            "missed_info_accessibility": "inaccessible",
        },
        "answerabilityQA_list": None,
        "infoAccessibilityQAs_binary": [
            {"question": "Does Caio know this information?", "question_type": "tom:info_accessibility:binary", "correct_answer": "no", "missed_info_accessibility": "inaccessible"},
            {"question": "Does Bia know this information?", "question_type": "tom:info_accessibility:binary", "correct_answer": "yes", "missed_info_accessibility": "inaccessible"},
        ],
        "answerabilityQAs_binary": [
            {"question": "Does Dan know the precise correct answer to this question?", "question_type": "tom:answerability:binary", "correct_answer": "no:long", "missed_info_accessibility": "inaccessible"},
        ],
    }


def fantom_set() -> FantomSet:
    return FantomSet.model_validate(record())


def test_presence_follows_the_leave_and_the_return() -> None:
    spans = absence_spans(LINES, [fantom_set()])
    assert [(span.character, span.start, span.end) for span in spans] == [("Caio", 3, 6)]  # his goodbye is heard
    present = presence(LINES, spans)
    assert present[0] == {"Ana", "Bia", "Caio"}  # the opening group, before Bia and Caio speak
    assert present[4] == {"Ana", "Bia"}  # Caio is away when Ana tells about Rex
    assert present[6] == {"Ana", "Bia", "Caio"}
    assert present[7] == {"Ana", "Bia", "Caio", "Dan"}  # Dan joins at his first line
    assert speakers_of(LINES[8]) == {"Ana", "Bia", "Caio"}


def test_presence_agrees_with_the_aware_list() -> None:
    conversation = to_conversation([fantom_set()], seed=7)
    assert presence_agreement(conversation) == (1, 1)
    assert conversation.conversation.turns[4].present == {"Ana", "Bia"}
    assert conversation.conversation.turns[8].speaker == "Ana, Bia & Caio"


def test_questions_use_the_official_blocks_and_ask_as_the_character() -> None:
    questions = {question.kind: question for question in set_questions(fantom_set(), "fantom-0", seed=7)}
    assert set(questions) == {QuestionKind.BELIEF_CHOICE, QuestionKind.INFO_ACCESSIBILITY_LIST, QuestionKind.INFO_ACCESSIBILITY_BINARY, QuestionKind.ANSWERABILITY_BINARY}
    belief = questions[QuestionKind.BELIEF_CHOICE]
    assert belief.viewer == "Caio" and belief.accessibility is Accessibility.INACCESSIBLE
    assert belief.prompt.endswith("\n\nChoose an answer from above:")
    assert f"({belief.correct[0]}) Caio does not know what Ana adopted." in belief.prompt
    listing = questions[QuestionKind.INFO_ACCESSIBILITY_LIST]
    assert listing.viewer is None  # "who knows" is asked by the omniscient reader
    assert listing.prompt == "Information: What did Ana adopt? A dog named Rex.\nQuestion: List all the characters who know this information."
    assert listing.correct == ("Ana", "Bia") and listing.wrong == ("Caio", "Dan")
    binary = questions[QuestionKind.ANSWERABILITY_BINARY]
    assert binary.prompt == "Target: What did Ana adopt?\nQuestion: Does Dan know the precise correct answer to this question? Answer yes or no."
    assert binary.viewer == "Dan" and binary.correct == ("no",)  # "no:long" is a no


def test_the_choice_order_is_drawn_from_the_seed() -> None:
    qa = fantom_set().beliefQAs[0]
    first = belief_choice_prompt(qa, random.Random("7:q"))
    assert first == belief_choice_prompt(qa, random.Random("7:q"))
    letters = {belief_choice_prompt(qa, random.Random(f"7:{n}"))[1] for n in range(20)}
    assert letters == {"a", "b"}


def test_scoring_matches_the_official_rules() -> None:
    assert choice_is_correct("(b) Caio does not know", "b")
    assert choice_is_correct("b.", "b") and not choice_is_correct("(a) He knows", "b")
    assert list_is_correct("Ana and Bia know it.", ["Ana", "Bia"], ["Caio", "Dan"])
    assert not list_is_correct("Ana, Bia and Caio", ["Ana", "Bia"], ["Caio", "Dan"])  # an unaware name
    assert not list_is_correct("Ana", ["Ana", "Bia"], ["Caio"])  # an aware name missing
    assert binary_value("No.") == 0 and binary_value("Yes, he does") == 1
    assert binary_value("He does not know it") == 0
    assert binary_value("Information not found") == -1
    assert asked_about("What does Sara believe about Javier's belief?") == "Sara"


def test_weighted_f1_by_hand() -> None:
    # yes: precision 1/1, recall 1/2 → 2/3; no: precision 1/2, recall 1/1 → 2/3; weighted (2/3·2 + 2/3·1)/3
    assert weighted_f1([1, 1, 0], [1, 0, 0]) == pytest.approx(0.6667, abs=1e-4)
    assert weighted_f1([1, 0], [-1, -1]) == 0.0


def answer(set_id: str, kind: QuestionKind, accessibility: Accessibility, correct: bool, expected: str = "yes", given: str = "yes") -> FantomAnswer:
    return FantomAnswer("q", "c", set_id, kind, accessibility, None, "p", (expected,), given, None, correct, 0.0, Usage(), None, {})


def test_set_all_needs_every_closed_question_right() -> None:
    answers = [
        answer("s1", QuestionKind.BELIEF_CHOICE, Accessibility.INACCESSIBLE, True),
        answer("s1", QuestionKind.INFO_ACCESSIBILITY_BINARY, Accessibility.INACCESSIBLE, False, "no", "yes"),
        answer("s2", QuestionKind.BELIEF_CHOICE, Accessibility.INACCESSIBLE, True),
        answer("s2", QuestionKind.BELIEF_CHOICE, Accessibility.ACCESSIBLE, True),  # s2 is mixed: not a control set
        answer("s3", QuestionKind.BELIEF_CHOICE, Accessibility.ACCESSIBLE, True),
    ]
    report = summarize_answers(answers)
    inaccessible = report["inaccessible"]
    accessible = report["accessible"]
    assert isinstance(inaccessible, dict) and isinstance(accessible, dict)
    assert inaccessible["set_all"] == 0.5  # s1 fails, s2 passes
    assert accessible["set_all"] == 1.0  # only s3 counts
    assert inaccessible["info_accessibility_binary_weighted_f1"] == 0.0


def test_the_sample_is_of_whole_conversations() -> None:
    sets = []
    for conversation_id in range(10):
        data = record()
        data["conv_id"] = conversation_id
        data["set_id"] = f"{conversation_id}-0-0"
        sets.append(FantomSet.model_validate(data))
    conversations = to_conversations(sets, seed=7)
    chosen = choose_conversations(conversations, 3, seed=7)
    assert len(chosen) == 3 and chosen == choose_conversations(conversations, 3, seed=7)
    assert all(len(conversation.questions) == 5 for conversation in chosen)  # 1 belief, 1 list, 3 binary


def test_the_archive_is_checked_and_only_the_dataset_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    content = json.dumps([record()]).encode()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo(fantom.DATA_MEMBER)
        info.size = len(content)
        bundle.addfile(info, io.BytesIO(content))
    archive = buffer.getvalue()
    with pytest.raises(ValueError, match="SHA-256"):
        extract_dataset(archive, tmp_path)
    monkeypatch.setattr(fantom, "DATA_SHA256", hashlib.sha256(archive).hexdigest())
    path = extract_dataset(archive, tmp_path)
    assert fantom.load_sets(path)[0].joining_speaker == "Caio"


class EchoBackend:
    def complete(self, request: ChatRequest) -> Completion:
        return Completion(text="No.", prompt_tokens=10, completion_tokens=1, reasoning_tokens=0)


class SameVectorEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return normalize(np.array([(1.0, 1.0) for _ in texts], dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_the_character_does_not_see_what_was_said_while_away() -> None:
    conversation = to_conversation([fantom_set()], seed=7)
    llm = LLMClient(ClientSettings(model="test-model"), backend=EchoBackend(), cache_dir=None)
    embedder = SameVectorEmbedder()
    memory = FullContextMemory.create(llm, embedder, DateparserTagger(), embedder, classifier=KeywordClassifier(), visible_only=True)
    for turn in conversation.conversation.turns:
        memory.add_turn(turn)
    caio = next(q for q in conversation.questions if q.kind is QuestionKind.INFO_ACCESSIBILITY_BINARY and q.viewer == "Caio")
    reply = memory.query(caio.prompt, conversation.conversation.reference_time, viewer=caio.viewer)
    assert reply.trace["included"] == ["e1", "e2", "e3", "e7", "e8", "e9"]  # lines 3-5 happened while he was away
    assert is_correct(caio, reply.answer)


def test_the_form_check_is_fixed_per_question_type() -> None:
    questions = {question.kind: question for question in set_questions(fantom_set(), "fantom-0", seed=7)}
    characters = frozenset({"Ana", "Bia", "Caio", "Dan"})
    binary = questions[QuestionKind.INFO_ACCESSIBILITY_BINARY]
    assert is_well_formed(binary, "No.", characters) and is_well_formed(binary, "yes, he does", characters)
    assert not is_well_formed(binary, "Information not found shutting down. No further strings needed.", characters)
    choice = questions[QuestionKind.BELIEF_CHOICE]
    assert is_well_formed(choice, "(b) Caio does not know", characters) and is_well_formed(choice, "a", characters)
    assert not is_well_formed(choice, "Information not found", characters)
    listing = questions[QuestionKind.INFO_ACCESSIBILITY_LIST]
    assert is_well_formed(listing, "Ana and Bia", characters)
    assert not is_well_formed(listing, "Information not found", characters)
    assert not is_well_formed(listing, "Goodbye", characters)


class DegenerateOnceBackend:
    """Answers out of form for the first seed and in form for the next one."""

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def complete(self, request: ChatRequest) -> Completion:
        self.seeds.append(request["seed"])
        text = "Information not found shutting down. No further strings." if len(self.seeds) == 1 else "Yes."
        return Completion(text=text, prompt_tokens=10, completion_tokens=2, reasoning_tokens=0)


def test_an_answer_out_of_form_is_asked_once_more_and_the_retry_is_scored() -> None:
    conversation = to_conversation([fantom_set()], seed=7)
    backend = DegenerateOnceBackend()
    llm = LLMClient(ClientSettings(model="test-model"), backend=backend, cache_dir=None)
    embedder = SameVectorEmbedder()
    memory = FullContextMemory.create(llm, embedder, DateparserTagger(), embedder, classifier=KeywordClassifier())
    for turn in conversation.conversation.turns:
        memory.add_turn(turn)
    bia = next(q for q in conversation.questions if q.kind is QuestionKind.INFO_ACCESSIBILITY_BINARY and q.viewer == "Bia")
    meter = CostMeter(None, None, None)
    result = fantom.ask(memory, llm, conversation, bia, meter)
    assert backend.seeds[1] == backend.seeds[0] + 1
    assert result.first_answer is not None and result.first_answer.startswith("Information not found")
    assert result.answer == "Yes." and result.correct  # Bia heard it
    assert result.usage.calls == 2
    assert summarize_answers([result])["retried"] == 1
