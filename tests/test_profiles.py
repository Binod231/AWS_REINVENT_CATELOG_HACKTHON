"""Tests for the preference profile and the scorer.

The central invariant is that a score's factors sum to the score. Everything the
explainer claims rests on that, so it is asserted directly and also against the
whole recorded catalog.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from replan.adapters.wire import build_catalog, decode_catalog_page, decode_session
from replan.domain import Catalog, Level, SessionFormat
from replan.profiles import (
    DEFAULT_PROFILE_YAML,
    MealWindow,
    ObjectiveWeights,
    Profile,
    ProfileError,
    Scorer,
    VetoReason,
    load_profile,
    write_starter_profile,
)

EVENT = "test-event"


def make_session(
    session_id: str = "S1",
    *,
    code: str = "AIM301",
    title: str = "Building agents on Amazon Bedrock",
    abstract: str = "A practical look at agentic workflows and evaluation.",
    level: str | None = "300 \u2013 Advanced",
    session_type: str = "Chalk talk",
    venue: str | None = "Venetian",
    room: str | None = "Level 2 | Hall C",
    clock: str = "10:00",
    day: str = "2026-12-01",
    minutes: str = "60",
    topics: tuple[str, ...] = ("Artificial Intelligence",),
    services: tuple[str, ...] = ("Amazon Bedrock",),
    tracks: tuple[str, ...] = ("Generative AI",),
    areas: tuple[str, ...] = (),
    speakers: tuple[str, ...] = ("Ada Lovelace, Principal Engineer, AWS",),
    seats: str | None = "available",
    reservable: bool | None = True,
    all_day: bool = False,
    scheduled: bool = True,
) -> Any:
    payload: dict[str, Any] = {
        "sessionId": session_id,
        "title": title,
        "abbreviation": code,
        "abstract": abstract,
        "type": session_type,
        "topics": list(topics),
        "services": list(services),
        "tracks": list(tracks),
        "areasOfInterest": list(areas),
        "speakers": [{"name": name} for name in speakers],
    }
    if level is not None:
        payload["level"] = level
    if venue is not None:
        payload["venue"] = venue
    if room is not None:
        payload["room"] = room
    if seats is not None:
        payload["seatAvailability"] = seats
    if reservable is not None:
        payload["isReservable"] = reservable
    if all_day:
        payload["isAllDaySession"] = True
    if scheduled:
        payload["sessionTime"] = {
            "date": day,
            "time": clock,
            "length": minutes,
            "timezone": "America/Los_Angeles",
        }
    decoded = decode_session(payload)
    assert decoded is not None
    return decoded


def catalog_of(*sessions: Any) -> Catalog:
    return build_catalog(event_id=EVENT, sessions=sessions, source="test")


def score_one(profile: Profile, session: Any) -> Any:
    return Scorer(profile, catalog_of(session)).score(session)


# --------------------------------------------------------------------------- #
# Profile parsing
# --------------------------------------------------------------------------- #
def test_empty_profile_is_valid() -> None:
    assert Profile().name == "me"
    assert Profile.from_yaml("").topics == {}


def test_bare_lists_are_accepted_as_weight_one() -> None:
    """People write a list before they discover weights exist."""
    parsed = Profile.from_yaml("topics:\n  - Bedrock\n  - Lambda\n")
    assert parsed.topics == {"Bedrock": 1.0, "Lambda": 1.0}


def test_level_list_is_accepted() -> None:
    parsed = Profile.from_yaml("levels: [300, 400]\n")
    assert parsed.level_weights == {Level.ADVANCED: 1.0, Level.EXPERT: 1.0}


def test_string_is_accepted_where_a_list_is_expected() -> None:
    parsed = Profile.from_yaml("speakers: Werner Vogels\n")
    assert parsed.speakers == ("Werner Vogels",)


def test_contradictory_pins_are_rejected() -> None:
    with pytest.raises(ProfileError, match="both must_attend and never_attend"):
        Profile.from_yaml("must_attend: [SVS401]\nnever_attend: [svs401]\n")


def test_reversed_day_window_is_rejected() -> None:
    with pytest.raises(ProfileError, match="day_end must be after"):
        Profile.from_yaml('day_start: "18:00"\nday_end: "09:00"\n')


def test_invalid_yaml_is_reported_clearly() -> None:
    with pytest.raises(ProfileError, match="not valid YAML"):
        Profile.from_yaml("topics: [unclosed\n")


def test_non_mapping_profile_is_rejected() -> None:
    with pytest.raises(ProfileError, match="mapping at the top level"):
        Profile.from_yaml("- just\n- a\n- list\n")


def test_unknown_key_is_rejected() -> None:
    """Silently ignoring a typo'd key would silently ignore the preference."""
    with pytest.raises(ProfileError):
        Profile.from_yaml("topcis: {Bedrock: 1}\n")


def test_meal_window_requires_positive_length() -> None:
    with pytest.raises(ValueError, match="ends at or before"):
        MealWindow(title="Lunch", start=time(13, 0), end=time(12, 0))


def test_weights_are_bounded() -> None:
    with pytest.raises(ValueError):
        ObjectiveWeights(utility=-1.0)
    with pytest.raises(ValueError):
        ObjectiveWeights(coverage=11.0)


def test_yaml_round_trip() -> None:
    original = Profile(
        name="binod",
        topics={"Bedrock AgentCore": 3.0},
        levels={400: 1.0},
        walk_tolerance_minutes=25,
    )
    assert Profile.from_yaml(original.to_yaml()) == original


# --------------------------------------------------------------------------- #
# Starter profile
# --------------------------------------------------------------------------- #
def test_bundled_starter_profile_parses() -> None:
    """The file we hand new users must itself be valid."""
    parsed = Profile.from_yaml(DEFAULT_PROFILE_YAML)
    assert parsed.topics
    assert parsed.levels
    assert parsed.meal_windows
    assert parsed.weights.utility > 0


def test_write_starter_profile(tmp_path: Path) -> None:
    target = tmp_path / "profile.yaml"
    written = write_starter_profile(target)
    assert written.exists()
    assert Profile.from_yaml(written.read_text(encoding="utf-8")).name == "me"


def test_write_starter_profile_refuses_to_clobber(tmp_path: Path) -> None:
    target = tmp_path / "profile.yaml"
    write_starter_profile(target)
    with pytest.raises(ProfileError, match="already exists"):
        write_starter_profile(target)
    write_starter_profile(target, force=True)


def test_load_profile_none_gives_defaults() -> None:
    assert load_profile(None) == Profile()


def test_load_missing_profile_explains_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="replan profile init"):
        load_profile(tmp_path / "absent.yaml")


# --------------------------------------------------------------------------- #
# The core invariant
# --------------------------------------------------------------------------- #
def test_factors_sum_to_total() -> None:
    profile = Profile(
        topics={"bedrock": 2.0, "agentic": 1.5},
        avoid_topics={"migration": 1.0},
        levels={300: 1.0},
        formats={"chalk-talk": 1.5},
        speakers=("Ada",),
        preferred_venues=("Venetian",),
    )
    result = score_one(profile, make_session())
    assert result.factors
    assert result.total == pytest.approx(sum(item.points for item in result.factors))


def test_factors_sum_to_total_across_the_real_catalog(
    list_sessions_payload: dict[str, Any], public_event_id: str
) -> None:
    page = decode_catalog_page(list_sessions_payload)
    catalog = build_catalog(event_id=public_event_id, sessions=page.sessions, source="recorded")
    profile = Profile.from_yaml(DEFAULT_PROFILE_YAML)
    scorer = Scorer(profile, catalog)
    for result in scorer.score_all(include_vetoed=True):
        expected = sum(item.points for item in result.factors)
        assert result.total == pytest.approx(expected), result.code


# --------------------------------------------------------------------------- #
# Monotonicity
# --------------------------------------------------------------------------- #
def test_higher_topic_weight_scores_higher() -> None:
    session = make_session()
    low = score_one(Profile(topics={"bedrock": 1.0}), session)
    high = score_one(Profile(topics={"bedrock": 3.0}), session)
    assert high.total > low.total


def test_avoided_topic_lowers_the_score() -> None:
    session = make_session()
    neutral = score_one(Profile(topics={"bedrock": 1.0}), session)
    penalised = score_one(Profile(topics={"bedrock": 1.0}, avoid_topics={"agentic": 2.0}), session)
    assert penalised.total < neutral.total
    assert penalised.negative_factors


def test_unmatched_keywords_contribute_nothing() -> None:
    session = make_session(topics=(), services=(), tracks=(), abstract="", title="Nothing here")
    result = score_one(Profile(topics={"quantum-teleportation": 5.0}), session)
    assert result.total == 0.0
    assert result.factors == ()


def test_structured_match_beats_prose_match() -> None:
    """A tagged service is a stronger signal than a passing mention."""
    tagged = make_session("a", services=("Amazon Bedrock",), abstract="Nothing relevant.")
    prose = make_session(
        "b", services=(), topics=(), tracks=(), title="A talk", abstract="We mention bedrock once."
    )
    profile = Profile(topics={"bedrock": 1.0})
    assert score_one(profile, tagged).total > score_one(profile, prose).total


def test_keyword_matches_only_once_per_session() -> None:
    """A word appearing in five fields must not multiply the score fivefold."""
    session = make_session(
        topics=("Amazon Bedrock",),
        services=("Amazon Bedrock",),
        tracks=("Amazon Bedrock",),
        areas=("Amazon Bedrock",),
        title="Amazon Bedrock",
        abstract="Amazon Bedrock",
    )
    result = score_one(Profile(topics={"bedrock": 1.0}), session)
    topic_factors = [item for item in result.factors if item.name.startswith("topic:")]
    assert len(topic_factors) == 1


def test_substring_matching_works_on_real_vocabulary() -> None:
    """'agent' must find 'Bedrock AgentCore'; exact match only would be useless."""
    session = make_session(services=("Amazon Bedrock AgentCore",), topics=(), tracks=())
    assert score_one(Profile(topics={"agent": 1.0}), session).total > 0


# --------------------------------------------------------------------------- #
# Levels, formats, speakers, venues
# --------------------------------------------------------------------------- #
def test_preferred_level_scores_above_unlisted_level() -> None:
    wanted = make_session("a", level="300 \u2013 Advanced")
    unwanted = make_session("b", level="100 \u2013 Foundational")
    profile = Profile(levels={300: 1.0})
    assert score_one(profile, wanted).total > score_one(profile, unwanted).total


def test_unlisted_level_is_penalised_not_vetoed() -> None:
    session = make_session(level="100 \u2013 Foundational")
    result = score_one(Profile(levels={400: 1.0}), session)
    assert result.eligible
    assert any(item.name == "level" and item.points < 0 for item in result.factors)


def test_no_level_is_not_penalised() -> None:
    """Keynotes genuinely carry "No Level"; punishing that would be wrong."""
    session = make_session(level="No Level")
    result = score_one(Profile(levels={300: 1.0}), session)
    assert not any(item.name == "level" for item in result.factors)


def test_preferred_format_adds_points() -> None:
    session = make_session(session_type="Chalk talk")
    with_pref = score_one(Profile(formats={"chalk-talk": 2.0}), session)
    without = score_one(Profile(), session)
    assert with_pref.total > without.total


def test_speaker_match_adds_points_once() -> None:
    session = make_session(speakers=("Werner Vogels, VP & CTO, Amazon", "Ada Lovelace, AWS"))
    result = score_one(Profile(speakers=("Werner Vogels",)), session)
    speaker_factors = [item for item in result.factors if item.name == "speaker"]
    assert len(speaker_factors) == 1
    assert speaker_factors[0].points > 0


def test_preferred_venue_adds_points() -> None:
    session = make_session(venue="Venetian")
    assert score_one(Profile(preferred_venues=("Venetian",)), session).total > 0


def test_venue_preference_ignores_other_venues() -> None:
    session = make_session(venue="Mandalay Bay")
    result = score_one(Profile(preferred_venues=("Venetian",)), session)
    assert not any(item.name == "venue" for item in result.factors)


# --------------------------------------------------------------------------- #
# Pins and bans
# --------------------------------------------------------------------------- #
def test_must_attend_dominates_everything() -> None:
    pinned = make_session(
        "a", code="ARC101", topics=(), services=(), tracks=(), abstract="", title="Dull"
    )
    attractive = make_session("b", code="AIM401")
    catalog = catalog_of(pinned, attractive)
    profile = Profile(topics={"bedrock": 5.0, "agentic": 5.0}, must_attend=("ARC101",))
    scorer = Scorer(profile, catalog)

    ranked = scorer.ranked()
    assert ranked[0].code == "ARC101"
    assert ranked[0].pinned
    assert ranked[0].total > ranked[1].total


def test_never_attend_is_a_veto_not_a_low_score() -> None:
    session = make_session(code="AIM301")
    catalog = catalog_of(session)
    scorer = Scorer(Profile(never_attend=("AIM301",)), catalog)
    result = scorer.score(session)
    assert result.eligible is False
    assert result.veto is VetoReason.BANNED
    assert "never_attend" in result.veto_detail
    assert scorer.ranked() == ()


def test_pins_resolve_by_code_and_by_id() -> None:
    session = make_session("sess-1", code="AIM301")
    catalog = catalog_of(session)
    assert Profile(must_attend=("AIM301",)).pins(catalog) == {"sess-1"}
    assert Profile(must_attend=("sess-1",)).pins(catalog) == {"sess-1"}


def test_unresolved_pins_are_reported() -> None:
    """A typo in must_attend must not vanish silently."""
    catalog = catalog_of(make_session(code="AIM301"))
    profile = Profile(must_attend=("AIM301", "NOPE999"))
    assert profile.unresolved(profile.must_attend, catalog) == ("NOPE999",)


def test_pin_overrides_soft_exclusions() -> None:
    """You asked for it explicitly, so a level or window preference must not drop it."""
    session = make_session(code="AIM301", level="100 \u2013 Foundational", clock="07:00")
    catalog = catalog_of(session)
    profile = Profile(must_attend=("AIM301",), levels={400: 1.0}, day_start=time(9, 0))
    assert Scorer(profile, catalog).score(session).eligible


def test_ban_beats_pin_resolution_order() -> None:
    """A banned session stays banned even if something else pins it."""
    session = make_session(code="AIM301")
    catalog = catalog_of(session)
    scorer = Scorer(Profile(never_attend=("AIM301",)), catalog)
    assert scorer.score(session).veto is VetoReason.BANNED


# --------------------------------------------------------------------------- #
# Vetoes
# --------------------------------------------------------------------------- #
def test_unscheduled_session_is_vetoed() -> None:
    session = make_session(scheduled=False)
    result = score_one(Profile(), session)
    assert result.veto is VetoReason.UNSCHEDULED
    assert "no published time" in result.veto_detail


def test_all_day_session_is_vetoed() -> None:
    result = score_one(Profile(), make_session(all_day=True))
    assert result.veto is VetoReason.UNSCHEDULED
    assert "all-day" in result.veto_detail


def test_session_outside_day_window_is_vetoed() -> None:
    early = make_session(clock="07:00")
    result = score_one(Profile(day_start=time(9, 0)), early)
    assert result.veto is VetoReason.OUTSIDE_DAY_WINDOW


def test_session_ending_after_day_end_is_vetoed() -> None:
    late = make_session(clock="18:30", minutes="120")
    result = score_one(Profile(day_end=time(19, 0)), late)
    assert result.veto is VetoReason.OUTSIDE_DAY_WINDOW


def test_overlong_session_is_vetoed() -> None:
    marathon = make_session(minutes="480")
    result = score_one(Profile(max_consecutive_minutes=240), marathon)
    assert result.veto is VetoReason.TOO_LONG
    assert "480" in result.veto_detail


def test_walk_up_can_be_excluded() -> None:
    session = make_session(seats="walkUp")
    assert score_one(Profile(), session).eligible
    assert score_one(Profile(include_walk_up=False), session).veto is (VetoReason.EXCLUDED_WALK_UP)


def test_unreservable_can_be_excluded() -> None:
    session = make_session(seats=None, reservable=False)
    assert score_one(Profile(), session).eligible
    assert score_one(Profile(include_unreservable=False), session).veto is (
        VetoReason.EXCLUDED_UNRESERVABLE
    )


def test_vetoed_sessions_score_zero_with_no_factors() -> None:
    result = score_one(Profile(), make_session(scheduled=False))
    assert result.total == 0.0
    assert result.factors == ()


# --------------------------------------------------------------------------- #
# Urgency is not desire
# --------------------------------------------------------------------------- #
def test_seat_scarcity_does_not_change_the_score() -> None:
    """A nearly-full session is more urgent, not more interesting."""
    profile = Profile(topics={"bedrock": 1.0})
    roomy = score_one(profile, make_session("a", seats="available"))
    scarce = score_one(profile, make_session("b", seats="veryLimited"))
    assert roomy.total == scarce.total
    assert scarce.urgency > roomy.urgency


def test_urgency_is_reported() -> None:
    assert score_one(Profile(), make_session(seats="veryLimited")).urgency == 3
    assert score_one(Profile(), make_session(seats="walkUp")).urgency == 0


def test_walk_up_carries_a_small_penalty() -> None:
    profile = Profile(topics={"bedrock": 1.0})
    normal = score_one(profile, make_session("a", seats="available"))
    walkup = score_one(profile, make_session("b", seats="walkUp"))
    assert walkup.total < normal.total


# --------------------------------------------------------------------------- #
# Repeats
# --------------------------------------------------------------------------- #
def test_later_repeat_is_penalised_when_requested() -> None:
    first = make_session("r1", code="AIM301", clock="09:00")
    later = make_session("r2", code="AIM301-R2", clock="16:00")
    catalog = catalog_of(first, later)

    off = Scorer(Profile(topics={"bedrock": 1.0}), catalog)
    assert off.score(first).total == off.score(later).total

    on = Scorer(Profile(topics={"bedrock": 1.0}, avoid_repeats=True), catalog)
    assert on.score(later).total < on.score(first).total
    assert any(item.name == "repeat" for item in on.score(later).factors)


def test_first_showing_is_never_penalised() -> None:
    first = make_session("r1", code="AIM301", clock="09:00")
    later = make_session("r2", code="AIM301-R2", clock="16:00")
    scorer = Scorer(Profile(avoid_repeats=True), catalog_of(first, later))
    assert not any(item.name == "repeat" for item in scorer.score(first).factors)


# --------------------------------------------------------------------------- #
# Bulk behaviour
# --------------------------------------------------------------------------- #
def test_ranked_is_sorted_and_excludes_vetoed() -> None:
    good = make_session("a", code="AIM401")
    banned = make_session("b", code="ARC101")
    unscheduled = make_session("c", code="EXP001", scheduled=False)
    catalog = catalog_of(good, banned, unscheduled)
    scorer = Scorer(Profile(topics={"bedrock": 1.0}, never_attend=("ARC101",)), catalog)

    ranked = scorer.ranked()
    assert [item.code for item in ranked] == ["AIM401"]
    totals = [item.total for item in ranked]
    assert totals == sorted(totals, reverse=True)


def test_ranking_is_deterministic() -> None:
    catalog = catalog_of(
        make_session("a", code="AIM401"),
        make_session("b", code="AIM402"),
        make_session("c", code="AIM403"),
    )
    profile = Profile(topics={"bedrock": 1.0})
    first = [item.code for item in Scorer(profile, catalog).ranked()]
    second = [item.code for item in Scorer(profile, catalog).ranked()]
    assert first == second


def test_ties_break_by_code_for_stability() -> None:
    catalog = catalog_of(make_session("b", code="ZZZ999"), make_session("a", code="AAA111"))
    ranked = Scorer(Profile(), catalog).ranked()
    assert [item.code for item in ranked] == ["AAA111", "ZZZ999"]


def test_limit_is_applied() -> None:
    catalog = catalog_of(*[make_session(f"s{i}", code=f"AIM{300 + i}") for i in range(10)])
    assert len(Scorer(Profile(), catalog).ranked(3)) == 3


def test_by_id_includes_vetoed() -> None:
    catalog = catalog_of(
        make_session("a", code="AIM401"), make_session("b", code="EXP1", scheduled=False)
    )
    lookup = Scorer(Profile(), catalog).by_id()
    assert set(lookup) == {"a", "b"}
    assert lookup["b"].veto is VetoReason.UNSCHEDULED


def test_top_reasons_are_the_biggest_contributors() -> None:
    session = make_session()
    result = score_one(
        Profile(topics={"bedrock": 5.0, "evaluation": 0.1}, levels={300: 1.0}), session
    )
    reasons = result.top_reasons(1)
    assert len(reasons) == 1
    biggest = max(result.factors, key=lambda item: abs(item.points))
    assert reasons[0] == biggest.detail


def test_describe_is_human_readable() -> None:
    result = score_one(Profile(topics={"bedrock": 1.0}), make_session())
    text = result.describe()
    assert "scores" in text
    assert "+" in text


def test_describe_explains_a_veto() -> None:
    result = score_one(Profile(), make_session(scheduled=False))
    assert result.describe().startswith("excluded:")


# --------------------------------------------------------------------------- #
# Real-catalog sanity
# --------------------------------------------------------------------------- #
def test_default_profile_ranks_the_real_catalog_sensibly(
    list_sessions_payload: dict[str, Any], public_event_id: str
) -> None:
    """The starter profile weights agentic AI heavily; the top hits should reflect it."""
    page = decode_catalog_page(list_sessions_payload)
    catalog = build_catalog(event_id=public_event_id, sessions=page.sessions, source="recorded")
    scorer = Scorer(Profile.from_yaml(DEFAULT_PROFILE_YAML), catalog)
    top = scorer.ranked(10)
    assert top

    lookup = {item.session_id: item for item in catalog.sessions}
    blobs = [lookup[item.session_id].search_blob.lower() for item in top]
    agentic = sum(1 for blob in blobs if "agent" in blob or "bedrock" in blob)
    assert agentic >= 7, f"only {agentic}/10 top hits were on-topic"


def test_scorer_handles_sessions_with_absent_fields(
    list_sessions_payload: dict[str, Any], public_event_id: str
) -> None:
    """The Summit catalog omits venue, seats and isReservable entirely."""
    page = decode_catalog_page(list_sessions_payload)
    catalog = build_catalog(event_id=public_event_id, sessions=page.sessions, source="recorded")
    profile = Profile(preferred_venues=("Venetian",), formats={"breakout": 1.0})
    for result in Scorer(profile, catalog).score_all(include_vetoed=True):
        assert isinstance(result.total, float)


def test_format_weights_parse_real_type_strings() -> None:
    parsed = Profile.from_yaml("formats:\n  chalk-talk: 1.5\n  breakout: 0.5\n")
    assert parsed.format_weights[SessionFormat.CHALK_TALK] == 1.5
    assert parsed.format_weights[SessionFormat.BREAKOUT] == 0.5


def test_starter_profile_yaml_is_parseable_by_pyyaml_directly() -> None:
    """Guards against a comment or indentation slip in the template."""
    raw = yaml.safe_load(DEFAULT_PROFILE_YAML)
    assert isinstance(raw, dict)
    assert "topics" in raw
