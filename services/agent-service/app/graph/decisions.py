"""Pydantic schemas of every structured LLM decision, with validators.

with_structured_output(Schema) turns each class into the JSON schema the LLM
must fill (the Field descriptions are part of the prompt). On top of that, the
validators below check what the model actually returned, in two stages:

  1. STRICT  model_validate(args, context={...})
     Validators that need run-time facts (how many passages were shown, which
     sources exist, which tasks were already done) read them from
     `info.context`. A violation raises, and decide() (common.py) sends the
     error text back to the model as a tool message: the model corrects itself.

  2. LENIENT model_validate(args, context={..., "lenient": True})
     If the model still gets it wrong after the retry, the same validators
     REPAIR instead of raising (drop the bad passage numbers, mark the
     unverifiable claim as unsupported...), so one bad output never fails a turn.

Validators that need no context (shape, length, duplicates) run in both stages
and also when LangChain or a test builds the object directly.
"""
import re
from typing import Annotated, Literal

from pydantic import (BaseModel, Field, StringConstraints, ValidationInfo, field_validator,
                      model_validator)

Worker = Literal["researcher", "planner", "analyst", "librarian", "writer"]
Text = Annotated[str, StringConstraints(strip_whitespace=True)]


def _ctx(info: ValidationInfo) -> dict:
    return info.context or {}


def _unique(items: list[str]) -> list[str]:
    """Drop empty strings and case-insensitive duplicates, keep the order."""
    seen, out = set(), []
    for item in items:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


# ---------------------------------------------------------------- supervisor
class RouteDecision(BaseModel):
    """The supervisor's next move."""
    next: Worker = Field(description="Which worker acts next. 'writer' = enough information, answer now.")
    instruction: Text = Field(default="", max_length=500, description=(
        "Concrete task for that worker. For researcher: ONE short question in the document's terms, "
        "e.g. 'referral threshold for the chief underwriter' (it is used as the search query, so no "
        "'search the documents for...'). For analyst: the calculation to do."))
    reason: Text = Field(default="", max_length=300, description="One short sentence: why this worker.")

    @model_validator(mode="after")
    def _instruction_and_no_repeats(self, info: ValidationInfo):
        ctx = _ctx(info)
        if self.next == "writer":
            return self
        if not self.instruction:
            if ctx.get("lenient"):
                return self                      # the node falls back to the user's question
            raise ValueError(f"instruction is required when next is '{self.next}'")
        done: dict = ctx.get("done") or {}       # {(agent, task.lower()): result}
        result = done.get((self.next, self.instruction.lower()))
        if result is not None:
            if ctx.get("lenient"):
                self.next, self.reason = "writer", "same task already done"
                return self
            raise ValueError(f"{self.next} already did exactly this task (result: {result[:150]}). "
                             f"Choose a different instruction or worker, or 'writer'.")
        return self


# ---------------------------------------------------------------- planner
class Plan(BaseModel):
    """Plan-and-execute: independent sub-questions researched in parallel."""
    sub_questions: list[Annotated[str, StringConstraints(strip_whitespace=True)]] = Field(
        min_length=1, description="2-4 self-contained questions, each answerable by one search.")

    @field_validator("sub_questions", mode="after")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        v = _unique(v)
        if not v:
            raise ValueError("at least one non-empty sub-question is required")
        short = [q for q in v if len(q.split()) < 3]
        if short:
            raise ValueError(f"sub-questions must be self-contained questions of 3+ words: {short}")
        return v

    @model_validator(mode="after")
    def _at_most(self, info: ValidationInfo):
        limit = _ctx(info).get("max_subquestions")
        if limit and len(self.sub_questions) > limit:
            if _ctx(info).get("lenient"):
                self.sub_questions = self.sub_questions[:limit]
            else:
                raise ValueError(f"at most {limit} sub-questions, got {len(self.sub_questions)}")
        return self


# ---------------------------------------------------------------- corrective RAG
class Grades(BaseModel):
    """Corrective RAG: which retrieved passages actually help."""
    relevant: list[int] = Field(default_factory=list,
                                description="Numbers of the passages that help answer the question.")

    @field_validator("relevant", mode="after")
    @classmethod
    def _sorted_unique(cls, v: list[int]) -> list[int]:
        return sorted(set(v))

    @model_validator(mode="after")
    def _in_range(self, info: ValidationInfo):
        n = _ctx(info).get("n_passages")
        if n is None:
            return self
        bad = [i for i in self.relevant if not 1 <= i <= n]
        if bad:
            if _ctx(info).get("lenient"):
                self.relevant = [i for i in self.relevant if 1 <= i <= n]
            else:
                raise ValueError(f"passage numbers must be between 1 and {n}; got {bad}")
        return self


class Rewrite(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=300)] = Field(
        description="A new search query using different key terms.")

    @model_validator(mode="after")
    def _new_query(self, info: ValidationInfo):
        tried = [q.lower() for q in _ctx(info).get("tried") or []]
        if self.query.lower() in tried and not _ctx(info).get("lenient"):
            raise ValueError(f"'{self.query}' was already tried; use different key terms")
        return self


# ---------------------------------------------------------------- self-RAG
_DASHES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                         "—": "-", " ": " "})


def _norm(s: str) -> str:
    """Compare WITHOUT any whitespace: PDF text extraction produces "at any time , request"
    and "Total Period of Cover age", which a model quoting the sentence fixes silently."""
    return re.sub(r"\s+", "", s.translate(_DASHES)).lower()


def quote_in_source(quote: str, source: str) -> bool:
    """Is `quote` copied from `source`? Ignores case, whitespace and typographic
    quotes; an ellipsis may skip text, as long as every part appears in order."""
    text, pos = _norm(source), 0
    parts = [p.strip(".\"'") for p in re.split(r"\.\.\.|…", _norm(quote))]
    parts = [p for p in parts if p]
    if not parts:
        return False
    for part in parts:
        found = text.find(part, pos)
        if found < 0:
            return False
        pos = found + len(part)
    return True


class ClaimCheck(BaseModel):
    claim: Text = Field(description="One factual claim made by the answer.")
    source_id: Text = Field(default="", description=(
        "The S-number that supports it, 'findings' if a worker finding supports it, or '' if nothing does."))
    quote: Text = Field(default="", description=(
        "ONE exact sentence (or part of it) of that source that supports it, copied verbatim."))
    supported: bool = Field(description="True only if the quote states the claim INCLUDING its conditions and timing.")
    note: str = Field(default="", exclude=True, description="Set by validation, not by the model.")


class Grounding(BaseModel):
    """Self-RAG: is every claim of the answer supported by the sources?
    The checker must go claim by claim and QUOTE its evidence before giving the
    verdict (field order = the order the model writes them). The validator then
    checks in code that every quote really is in the cited source: the model
    cannot "support" a claim with a sentence that does not exist."""
    claims: list[ClaimCheck] = Field(default_factory=list,
                                     description="Every factual claim of the answer, checked one by one.")
    grounded: bool = Field(description="True if every factual claim is supported by the sources/findings.")
    unsupported_claims: list[Text] = Field(default_factory=list,
                                           description="Claims that are NOT supported (empty if grounded).")

    @model_validator(mode="after")
    def _quotes_exist(self, info: ValidationInfo):
        sources: dict[str, str] | None = _ctx(info).get("sources")      # {"S1": text, ...}
        if not sources:
            return self
        errors = []
        for i, c in enumerate(self.claims):
            if not c.supported:
                continue
            if c.source_id not in sources:
                msg = f"claims[{i}].source_id '{c.source_id}' is not one of {sorted(sources)}"
            elif not c.quote:
                msg = f"claims[{i}] is marked supported but has no quote"
            elif not quote_in_source(c.quote, sources[c.source_id]):
                msg = f"claims[{i}].quote is not a verbatim sentence of {c.source_id}: \"{c.quote[:100]}\""
            else:
                continue
            if _ctx(info).get("lenient"):
                c.supported, c.note = False, msg
                self.grounded = False
            else:
                errors.append(msg)
        if errors:
            raise ValueError("; ".join(errors) + ". Copy each quote exactly from its source, "
                                                 "or mark the claim unsupported.")
        return self

    def problems(self) -> list[str]:
        """Unsupported claims from the verdict AND from the per-claim checks."""
        out = list(self.unsupported_claims)
        for c in self.claims:
            if not c.supported and c.claim not in out:
                why = c.note or (f'the source says: "{c.quote}"' if c.quote else "")
                out.append(c.claim + (f" ({why})" if why else ""))
        return out


# ---------------------------------------------------------------- long-term memory
class Memories(BaseModel):
    facts: list[Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)]] = Field(
        default_factory=list, description=(
            "At most 5 durable facts or preferences the USER stated about themselves (role, name, preferred "
            "answer style). Empty if none. Never include facts about documents."))

    @field_validator("facts", mode="after")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        return _unique(v)[:5]
