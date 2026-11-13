"""FANToM (Kim et al., EMNLP 2023): who knows what in a conversation that characters leave and rejoin.

Each record of the dataset is a *set*: one piece of information discussed while
a character was away, and the questions about it. A memory is built from the
whole conversation (`full_context`, shared by every set of a conversation) and
answers the closed-form questions, scored as in the authors' `eval_fantom.py`:

- belief, multiple choice: the character's belief against the omniscient one,
  in an order drawn per question from the seed;
- answerability and info-accessibility lists: every aware character named and
  no unaware one;
- answerability and info-accessibility yes/no: weighted F1 over yes and no.

The set scores are the paper's "All" (every closed question of the set right).

An answer out of the expected form (a yes/no that does not start with yes, no,
true or false; a choice that does not start with an option letter; a list that
names no character of the conversation, or says "Information not found") is
asked once more, with the seed moved so the call is new, in every system alike.
The second answer is scored as it comes; the record keeps both, and the summary
counts the retries.
Free-response belief and fact questions are not asked: they need a similarity
model or token F1, and the closed questions already cover the same sets.

The dataset has no field for who is present at each line: characters leave and
come back inside the dialogue ("I'll grab a coffee"). Presence is derived from
each set: its short context starts where its `joining_speaker` leaves, who is
away until their next line. A character is present from their first line on,
except while away; the opening group, who speak before anyone leaves, is
there from the first line. `presence_agreement` checks this against the dataset's own
lists of who knows each inaccessible piece of information.

A question about a character ("Does Gianna know...", "What does Gianna believe...")
is asked with that character as the viewer; the list questions ("List all the
characters who know...") are asked by the omniscient reader. The answer prompt
is the same C.2 of every system, with the official question block in place of
the question. Lines get synthetic times one minute apart, because the dataset
has none.

    python -m bench.fantom download
    python -m bench.fantom presence    # agreement of the derived presence with the aware lists, no LLM
    python -m bench.fantom run --system narrative --conversations 30 --max-cost-usd 1 \\
        --llm-prices <input> <output>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import tarfile
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from bench.benchmark import Conversation
from bench.runner import BudgetExceeded, CostMeter, Prices, System, _git, build_memory, create_memory
from magma.config import Ablation
from magma.embed import InstructedEmbedder
from magma.ingest import Turn
from magma.json_types import JSONObject
from magma.llm import LLMClient, Usage
from magma.memory import MemorySystem
from magma.router import build_default_classifier
from magma.temporal import DateparserTagger

DATA_URL = "https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz"
DATA_SHA256 = "1d08dfa0ea474c7f83b9bc7e3a7b466eab25194043489dd618b4c5223e1253a4"  # from the authors' dataset_loader.py
DATA_MEMBER = "fantom_v1.json"
DATA_DIRECTORY = Path("data/fantom")
START = datetime(2023, 1, 1, 12, 0)
PILOT_QUESTIONS = 50

_SPEAKER_SEPARATOR = re.compile(r",|&|\band\b")
_ASKED_ABOUT = re.compile(r"^(?:What does|Does) (\S+?) (?:believe|think|know)\b")


class QuestionKind(StrEnum):
    BELIEF_CHOICE = "BELIEF_CHOICE"
    ANSWERABILITY_LIST = "ANSWERABILITY_LIST"
    ANSWERABILITY_BINARY = "ANSWERABILITY_BINARY"
    INFO_ACCESSIBILITY_LIST = "INFO_ACCESSIBILITY_LIST"
    INFO_ACCESSIBILITY_BINARY = "INFO_ACCESSIBILITY_BINARY"


class Accessibility(StrEnum):
    ACCESSIBLE = "ACCESSIBLE"  # the character later heard the information: the control questions
    INACCESSIBLE = "INACCESSIBLE"  # the character never heard it: the FANToM questions proper


class FantomQA(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    question_type: str
    correct_answer: str | list[str]
    wrong_answer: str | list[str] | None = None
    missed_info_accessibility: str | None = None


class FantomSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    set_id: str
    part_id: str
    conv_id: int
    full_context: str
    short_context: str
    joining_speaker: str
    factQA: FantomQA
    beliefQAs: list[FantomQA]
    answerabilityQA_list: FantomQA | None = None
    infoAccessibilityQA_list: FantomQA | None = None
    answerabilityQAs_binary: list[FantomQA]
    infoAccessibilityQAs_binary: list[FantomQA]


@dataclass(frozen=True)
class FantomQuestion:
    id: str
    conversation_id: str
    set_id: str
    kind: QuestionKind
    accessibility: Accessibility
    viewer: str | None  # the character asked about; None for the omniscient reader
    prompt: str  # the official question block, given to the memory as the question
    correct: tuple[str, ...]  # "a"/"b" for a choice, "yes"/"no" for a binary, the aware names for a list
    wrong: tuple[str, ...]  # the unaware names, for a list


@dataclass(frozen=True)
class FantomConversation:
    id: str
    conversation: Conversation  # the turns, with derived presence, for build_memory
    questions: tuple[FantomQuestion, ...]
    sets: tuple[FantomSet, ...]
    characters: frozenset[str]  # everyone who speaks, for the form check of the list answers


@dataclass(frozen=True)
class AbsenceSpan:
    character: str
    start: int  # first line the character misses
    end: int  # the character's next line, where they are back (exclusive)


# ---------------------------------------------------------------- data


def download(directory: Path = DATA_DIRECTORY) -> Path:
    """Fetch the authors' archive, check its SHA-256 and extract the dataset file."""
    with urllib.request.urlopen(DATA_URL) as response:
        archive = bytes(response.read())
    return extract_dataset(archive, directory)


def extract_dataset(archive: bytes, directory: Path) -> Path:
    digest = hashlib.sha256(archive).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"fantom.tar.gz has SHA-256 {digest}, expected {DATA_SHA256}")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        member = bundle.extractfile(DATA_MEMBER)  # only this file is read: no path from the archive is trusted
        if member is None:
            raise ValueError(f"{DATA_MEMBER} is not in the archive")
        content = member.read()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DATA_MEMBER
    path.write_bytes(content)
    return path


def load_sets(path: Path) -> list[FantomSet]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [FantomSet.model_validate(record) for record in records]


# ---------------------------------------------------------------- presence


def split_line(line: str) -> tuple[str, str]:
    """`"Ana, Bia: Cheers!"` → (`"Ana, Bia"`, `"Cheers!"`)."""
    head, _, text = line.partition(":")
    return head.strip(), text.strip()


def speakers_of(line: str) -> frozenset[str]:
    head, _ = split_line(line)
    return frozenset(name.strip() for name in _SPEAKER_SEPARATOR.split(head) if name.strip())


def absence_spans(lines: Sequence[str], sets: Sequence[FantomSet]) -> list[AbsenceSpan]:
    """One span per distinct leave: from where the set's short context starts to the joining speaker's next line."""
    spans: dict[tuple[str, int], AbsenceSpan] = {}
    for fantom_set in sets:
        short = fantom_set.short_context.strip().split("\n")
        start = _find(lines, short)
        if start is None:
            continue
        character = fantom_set.joining_speaker
        back = next((k for k in range(start + 1, len(lines)) if character in speakers_of(lines[k])), None)
        if back is None:
            continue
        first_missed = start + 1 if character in speakers_of(lines[start]) else start  # their goodbye is heard
        spans[(character, first_missed)] = AbsenceSpan(character, first_missed, back)
    return sorted(spans.values(), key=lambda span: (span.start, span.character))


def presence(lines: Sequence[str], spans: Sequence[AbsenceSpan]) -> list[frozenset[str]]:
    """Who is present at each line, except those away.

    Whoever speaks before the first leave is there from the first line (the
    opening group, often addressed before they speak); a character who first
    speaks later joins at that line.
    """
    opening_end = min((span.start for span in spans), default=len(lines))
    joined_at: dict[str, int] = {}
    for number, line in enumerate(lines):
        for name in speakers_of(line):
            joined_at.setdefault(name, 0 if number < opening_end else number)
    present: list[frozenset[str]] = []
    for number in range(len(lines)):
        away = {span.character for span in spans if span.start <= number < span.end}
        present.append(frozenset(name for name, joined in joined_at.items() if joined <= number and name not in away))
    return present


def presence_agreement(conversation: FantomConversation) -> tuple[int, int]:
    """(matches, checked): inaccessible sets whose derived witnesses of the absence equal the dataset's aware list."""
    lines = _lines(conversation.sets[0])
    spans = absence_spans(lines, conversation.sets)
    present = presence(lines, spans)
    matches = checked = 0
    for fantom_set in conversation.sets:
        gold = fantom_set.infoAccessibilityQA_list
        if gold is None or gold.missed_info_accessibility != "inaccessible":
            continue
        span = next((s for s in spans if s.character == fantom_set.joining_speaker and _starts_set(lines, s, fantom_set)), None)
        if span is None:
            continue
        witnesses: set[str] = set()
        for number in range(span.start, span.end):
            witnesses |= present[number]
        checked += 1
        matches += witnesses == set(_names(gold.correct_answer))
    return matches, checked


# ---------------------------------------------------------------- conversations and questions


def to_conversations(sets: Sequence[FantomSet], seed: int) -> list[FantomConversation]:
    by_conversation: dict[int, list[FantomSet]] = {}
    for fantom_set in sets:
        by_conversation.setdefault(fantom_set.conv_id, []).append(fantom_set)
    return [to_conversation(group, seed) for _, group in sorted(by_conversation.items())]


def to_conversation(sets: Sequence[FantomSet], seed: int) -> FantomConversation:
    conversation_id = f"fantom-{sets[0].conv_id}"
    lines = _lines(sets[0])
    present = presence(lines, absence_spans(lines, sets))
    turns: list[Turn] = []
    for number, line in enumerate(lines):
        head, text = split_line(line)
        turns.append(Turn(head, text, START + timedelta(minutes=number), present[number] or frozenset({head})))
    questions = tuple(question for fantom_set in sets for question in set_questions(fantom_set, conversation_id, seed))
    reference_time = START + timedelta(minutes=len(lines))
    characters = frozenset(name for line in lines for name in speakers_of(line))
    conversation = Conversation(conversation_id, tuple(turns), reference_time, ())
    return FantomConversation(conversation_id, conversation, questions, tuple(sets), characters)


def set_questions(fantom_set: FantomSet, conversation_id: str, seed: int) -> list[FantomQuestion]:
    """The closed-form questions of a set, with the prompts of the official evaluation."""
    fact_q = fantom_set.factQA.question
    fact_a = _text(fantom_set.factQA.correct_answer)
    questions: list[FantomQuestion] = []

    def add(kind: QuestionKind, index: int, qa: FantomQA, prompt: str, correct: tuple[str, ...], wrong: tuple[str, ...]) -> None:
        question_id = f"{fantom_set.set_id}:{kind.value.lower()}:{index}"
        viewer = None if kind in (QuestionKind.ANSWERABILITY_LIST, QuestionKind.INFO_ACCESSIBILITY_LIST) else asked_about(qa.question)
        questions.append(
            FantomQuestion(question_id, conversation_id, fantom_set.set_id, kind, _accessibility(qa), viewer, prompt, correct, wrong)
        )

    for index, qa in enumerate(fantom_set.beliefQAs):
        question_id = f"{fantom_set.set_id}:{QuestionKind.BELIEF_CHOICE.value.lower()}:{index}"
        prompt, letter = belief_choice_prompt(qa, random.Random(f"{seed}:{question_id}"))
        add(QuestionKind.BELIEF_CHOICE, index, qa, prompt, (letter,), ())
    if fantom_set.answerabilityQA_list is not None:
        qa = fantom_set.answerabilityQA_list
        prompt = f"Target: {fact_q}\nQuestion: {qa.question}"
        add(QuestionKind.ANSWERABILITY_LIST, 0, qa, prompt, _names(qa.correct_answer), _names(qa.wrong_answer))
    for index, qa in enumerate(fantom_set.answerabilityQAs_binary):
        prompt = f"Target: {fact_q}\nQuestion: {qa.question} Answer yes or no."
        add(QuestionKind.ANSWERABILITY_BINARY, index, qa, prompt, (_yes_no(qa.correct_answer),), ())
    if fantom_set.infoAccessibilityQA_list is not None:
        qa = fantom_set.infoAccessibilityQA_list
        prompt = f"Information: {fact_q} {fact_a}\nQuestion: {qa.question}"
        add(QuestionKind.INFO_ACCESSIBILITY_LIST, 0, qa, prompt, _names(qa.correct_answer), _names(qa.wrong_answer))
    for index, qa in enumerate(fantom_set.infoAccessibilityQAs_binary):
        prompt = f"Information: {fact_q} {fact_a}\nQuestion: {qa.question} Answer yes or no."
        add(QuestionKind.INFO_ACCESSIBILITY_BINARY, index, qa, prompt, (_yes_no(qa.correct_answer),), ())
    return questions


def belief_choice_prompt(qa: FantomQA, rng: random.Random) -> tuple[str, str]:
    """The official two-option question: the omniscient belief (wrong) and the character's (right), order by coin flip."""
    options = [_text(qa.wrong_answer or ""), _text(qa.correct_answer)]
    answer_goes_last = rng.choice([True, False])
    if not answer_goes_last:
        options.reverse()
    letter = "b" if answer_goes_last else "a"
    choices = "\n".join(f"({label}) {option}" for label, option in zip("ab", options, strict=True))
    return f"{qa.question}\n{choices}\n\nChoose an answer from above:", letter


def asked_about(question: str) -> str | None:
    found = _ASKED_ABOUT.match(question)
    return found.group(1) if found else None


# ---------------------------------------------------------------- form check (before scoring)

_CHOICE_STARTS = ("(a)", "(b)", "a)", "b)", "a.", "b.", "a:", "b:", "a,", "b,")


def is_well_formed(question: FantomQuestion, answer: str, characters: frozenset[str]) -> bool:
    """Whether the answer has the form its question asks for: a fixed rule, the same for every system."""
    text = answer.strip().lstrip("\"'*").lower()
    match question.kind:
        case QuestionKind.BELIEF_CHOICE:
            return text in ("a", "b") or text.startswith(_CHOICE_STARTS)
        case QuestionKind.ANSWERABILITY_BINARY | QuestionKind.INFO_ACCESSIBILITY_BINARY:
            return text.startswith(("yes", "no", "true", "false"))
        case QuestionKind.ANSWERABILITY_LIST | QuestionKind.INFO_ACCESSIBILITY_LIST:
            return not text.startswith("information not found") and any(name.lower() in text for name in characters)


# ---------------------------------------------------------------- scoring (as in eval_fantom.py)


def is_correct(question: FantomQuestion, answer: str) -> bool:
    match question.kind:
        case QuestionKind.BELIEF_CHOICE:
            return choice_is_correct(answer, question.correct[0])
        case QuestionKind.ANSWERABILITY_LIST | QuestionKind.INFO_ACCESSIBILITY_LIST:
            return list_is_correct(answer, question.correct, question.wrong)
        case QuestionKind.ANSWERABILITY_BINARY | QuestionKind.INFO_ACCESSIBILITY_BINARY:
            return binary_value(answer) == (1 if question.correct[0] == "yes" else 0)


def choice_is_correct(answer: str, letter: str) -> bool:
    text = answer.lower().strip()
    starts = (f"({letter})", f"{letter})", f"{letter}.", f"{letter}:", f"{letter},")
    return text.startswith(starts) or f"({letter})" in text or text == letter


def list_is_correct(answer: str, aware: Sequence[str], unaware: Sequence[str]) -> bool:
    text = answer.lower()
    return all(name.lower() in text for name in aware) and not any(name.lower() in text for name in unaware)


def binary_value(answer: str) -> int:
    """1 for yes, 0 for no, -1 when neither is found; the yes patterns are checked first, as in the official code."""
    text = answer.lower().replace('"', "").replace("'", "").strip()
    padded = f" {text} "
    if any(pattern in padded for pattern in (" yes,", " yes ", " yes.", " knows ")) or text.startswith(("yes", "true")):
        return 1
    if any(pattern in padded for pattern in (" no,", " no ", " no.", " does not know ", " doesnt know ")) or text.startswith(("no", "false")):
        return 0
    return -1


def weighted_f1(gold: Sequence[int], predicted: Sequence[int]) -> float | None:
    """F1 per gold class (yes, no), weighted by its support; a prediction of -1 is wrong for both."""
    if not gold:
        return None
    total = 0.0
    for label in sorted(set(gold)):
        true_positive = sum(1 for g, p in zip(gold, predicted, strict=True) if g == label and p == label)
        predicted_positive = sum(1 for p in predicted if p == label)
        support = sum(1 for g in gold if g == label)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        total += f1 * support
    return round(total / len(gold), 4)


@dataclass(frozen=True)
class FantomAnswer:
    question_id: str
    conversation_id: str
    set_id: str
    kind: QuestionKind
    accessibility: Accessibility
    viewer: str | None
    prompt: str
    expected: tuple[str, ...]
    answer: str  # the scored answer: the retry's, when there was one
    first_answer: str | None  # the answer out of form that was asked again; None when the first was in form
    correct: bool
    latency_s: float
    usage: Usage
    cost_usd: float | None
    trace: JSONObject


def summarize_answers(answers: Sequence[FantomAnswer]) -> JSONObject:
    """Per question type and accessibility: accuracy (lists, choice) or weighted F1 (binary); and the set-level All."""
    retried = [answer for answer in answers if answer.first_answer is not None]
    retries: JSONObject = {kind.value.lower(): sum(1 for answer in retried if answer.kind is kind) for kind in QuestionKind}
    report: JSONObject = {
        "questions": len(answers),
        "retried": len(retried),  # answers out of form, asked once more
        "retried_by_kind": retries,
    }
    for accessibility in Accessibility:
        part = [answer for answer in answers if answer.accessibility is accessibility]
        scores: JSONObject = {"questions": len(part)}
        for kind in QuestionKind:
            of_kind = [answer for answer in part if answer.kind is kind]
            if not of_kind:
                continue
            accuracy = round(sum(answer.correct for answer in of_kind) / len(of_kind), 4)
            scores[f"{kind.value.lower()}_accuracy"] = accuracy
            if kind in (QuestionKind.ANSWERABILITY_BINARY, QuestionKind.INFO_ACCESSIBILITY_BINARY):
                gold = [1 if answer.expected[0] == "yes" else 0 for answer in of_kind]
                predicted = [binary_value(answer.answer) for answer in of_kind]
                scores[f"{kind.value.lower()}_weighted_f1"] = weighted_f1(gold, predicted)
        scores["set_all"] = _set_all(part, answers, accessibility)
        report[accessibility.value.lower()] = scores
    return report


def _set_all(part: Sequence[FantomAnswer], every: Sequence[FantomAnswer], accessibility: Accessibility) -> float | None:
    """Share of sets whose closed questions of this accessibility are all right; control sets must be fully accessible."""
    by_set: dict[str, list[FantomAnswer]] = {}
    for answer in part:
        by_set.setdefault(answer.set_id, []).append(answer)
    if accessibility is Accessibility.ACCESSIBLE:
        mixed = {answer.set_id for answer in every if answer.accessibility is Accessibility.INACCESSIBLE}
        by_set = {set_id: group for set_id, group in by_set.items() if set_id not in mixed}
    if not by_set:
        return None
    return round(sum(all(answer.correct for answer in group) for group in by_set.values()) / len(by_set), 4)


# ---------------------------------------------------------------- run


def choose_conversations(conversations: Sequence[FantomConversation], count: int | None, seed: int) -> list[FantomConversation]:
    """A seeded sample of whole conversations (each one carries every question type), in dataset order."""
    if count is None or count >= len(conversations):
        return list(conversations)
    chosen = set(random.Random(seed).sample([conversation.id for conversation in conversations], count))
    return [conversation for conversation in conversations if conversation.id in chosen]


def ask(
    memory: MemorySystem, llm: LLMClient, conversation: FantomConversation, question: FantomQuestion, meter: CostMeter
) -> FantomAnswer:
    """Ask once; an answer out of form is asked once more with a new seed, and the second answer is scored."""
    now = conversation.conversation.reference_time
    reply = memory.query(question.prompt, now, viewer=question.viewer)
    first_answer: str | None = None
    usage = Usage()
    usage.merge(reply.usage)
    latency_s = reply.latency_s
    if not is_well_formed(question, reply.answer, conversation.characters):
        first_answer = reply.answer
        with llm.retry_attempt(1):
            reply = memory.query(question.prompt, now, viewer=question.viewer)
        usage.merge(reply.usage)
        latency_s += reply.latency_s
    return FantomAnswer(
        question_id=question.id,
        conversation_id=question.conversation_id,
        set_id=question.set_id,
        kind=question.kind,
        accessibility=question.accessibility,
        viewer=question.viewer,
        prompt=question.prompt,
        expected=question.correct,
        answer=reply.answer,
        first_answer=first_answer,
        correct=is_correct(question, reply.answer),
        latency_s=round(latency_s, 3),
        usage=usage,
        cost_usd=meter.charge_question(usage, Usage()),
        trace=reply.trace,
    )


def run(arguments: argparse.Namespace) -> int:
    load_dotenv()
    system = System(arguments.system.upper())
    ablation = Ablation(arguments.ablation.upper()) if arguments.ablation else None
    every = to_conversations(load_sets(arguments.data), arguments.seed)
    conversations = choose_conversations(every, arguments.conversations, arguments.seed)
    questions_planned = sum(len(conversation.questions) for conversation in conversations)
    run_id = arguments.run_id or f"fantom-{system.run_prefix}-c{len(conversations)}-seed{arguments.seed}"
    directory = arguments.results_dir / run_id
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("builds.jsonl", "answers.jsonl", "summary.json", "llm_calls.jsonl"):
        (directory / name).unlink(missing_ok=True)

    llm = LLMClient.from_env("LLM", log_path=directory / "llm_calls.jsonl")
    prices = Prices(*arguments.llm_prices) if arguments.llm_prices else None
    meter = CostMeter(arguments.max_cost_usd, prices, Prices(0.0, 0.0) if prices else None)
    agreement = [presence_agreement(conversation) for conversation in conversations]
    settings: JSONObject = {
        "run_id": run_id,
        "command": " ".join(sys.argv),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "system": system.value,
        "ablation": ablation.value if ablation else None,
        "seed": arguments.seed,
        "conversations": [conversation.id for conversation in conversations],
        "questions_planned": questions_planned,
        "presence_agreement": {"matches": sum(m for m, _ in agreement), "checked": sum(c for _, c in agreement)},
        "max_cost_usd": arguments.max_cost_usd,
        "llm_prices": arguments.llm_prices,
        "llm": {"model": llm.settings.model, "base_url": llm.settings.base_url, "context_window": llm.settings.context_window},
    }
    (directory / "run.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    embedder = InstructedEmbedder.from_config()
    tagger = DateparserTagger()
    classifier = build_default_classifier(tagger)
    answers: list[FantomAnswer] = []
    stopped: str | None = None
    start = time.perf_counter()
    try:
        for conversation in conversations:
            memory = create_memory(system, ablation, llm, embedder, embedder, tagger, classifier)
            try:
                seconds, usage = build_memory(conversation.conversation, memory)
                memory.save(directory / "memories" / conversation.id)
                build: JSONObject = {
                    "conversation_id": conversation.id,
                    "turns": len(conversation.conversation.turns),
                    "seconds": round(seconds, 2),
                    "usage": json.loads(json.dumps(asdict(usage))),
                    "cost_usd": meter.charge_build(usage),
                }
                _append(directory / "builds.jsonl", build)
                meter.check()
                for question in conversation.questions:
                    answer = ask(memory, llm, conversation, question, meter)
                    answers.append(answer)
                    _append(directory / "answers.jsonl", json.loads(json.dumps(asdict(answer))))
                    meter.check()
                    if len(answers) == PILOT_QUESTIONS:
                        meter.check_projection(len(answers), questions_planned - len(answers))
            finally:
                memory.close()
    except BudgetExceeded as error:
        stopped = str(error)

    summary = summarize_answers(answers)
    summary["run_id"] = run_id
    summary["questions_planned"] = questions_planned
    summary["wall_hours"] = round((time.perf_counter() - start) / 3600, 4)
    summary["spent_usd"] = round(meter.spent_usd, 4) if meter.priced else None
    summary["stopped"] = stopped
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if stopped else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="fetch and check the authors' archive")
    fetch.add_argument("--directory", type=Path, default=DATA_DIRECTORY)
    check = commands.add_parser("presence", help="agreement of the derived presence with the dataset's aware lists")
    check.add_argument("--data", type=Path, default=DATA_DIRECTORY / DATA_MEMBER)
    runner = commands.add_parser("run", help="build each conversation and ask its closed-form questions")
    runner.add_argument("--data", type=Path, default=DATA_DIRECTORY / DATA_MEMBER)
    runner.add_argument("--system", choices=[system.value.lower() for system in System], default="narrative")
    runner.add_argument("--ablation", choices=[ablation.value.lower() for ablation in Ablation], default=None)
    runner.add_argument("--conversations", type=int, default=30, help="seeded sample of whole conversations; 0 for all")
    runner.add_argument("--seed", type=int, default=7)
    runner.add_argument("--max-cost-usd", type=float, default=None)
    runner.add_argument("--llm-prices", type=float, nargs=2, metavar=("INPUT", "OUTPUT"), default=None)
    runner.add_argument("--run-id", default=None)
    runner.add_argument("--results-dir", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    if arguments.command == "download":
        print(f"written {download(arguments.directory)}")
        return
    if arguments.command == "presence":
        conversations = to_conversations(load_sets(arguments.data), seed=0)
        counts = [presence_agreement(conversation) for conversation in conversations]
        matches, checked = sum(m for m, _ in counts), sum(c for _, c in counts)
        print(f"{len(conversations)} conversations; derived witnesses equal the aware list in {matches}/{checked} inaccessible sets")
        return
    if arguments.conversations == 0:
        arguments.conversations = None
    if arguments.max_cost_usd is not None and arguments.llm_prices is None:
        parser.error("a cost cap needs --llm-prices")
    raise SystemExit(run(arguments))


# ---------------------------------------------------------------- helpers


def _lines(fantom_set: FantomSet) -> list[str]:
    return [line for line in fantom_set.full_context.strip().split("\n") if line.strip()]


def _find(lines: Sequence[str], block: Sequence[str]) -> int | None:
    block = [line for line in block if line.strip()]
    for start in range(len(lines) - len(block) + 1):
        if list(lines[start : start + len(block)]) == list(block):
            return start
    return None


def _starts_set(lines: Sequence[str], span: AbsenceSpan, fantom_set: FantomSet) -> bool:
    start = _find(lines, fantom_set.short_context.strip().split("\n"))
    return start is not None and span.start in (start, start + 1)


def _names(value: str | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(value) if isinstance(value, list) else (value,)


def _text(value: str | list[str]) -> str:
    return value if isinstance(value, str) else ", ".join(value)


def _yes_no(value: str | list[str]) -> str:
    return "yes" if _text(value).split(":")[0] == "yes" else "no"  # "no:long" counts as no in the full-context scoring


def _accessibility(qa: FantomQA) -> Accessibility:
    label = qa.missed_info_accessibility or qa.question_type.rsplit(":", 1)[-1]
    return Accessibility.INACCESSIBLE if label == "inaccessible" else Accessibility.ACCESSIBLE


def _append(path: Path, record: JSONObject) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
