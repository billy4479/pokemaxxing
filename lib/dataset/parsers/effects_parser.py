"""Rule-based parser from effect prose to typed payload classes."""

from __future__ import annotations

import re
from dataclasses import asdict

import pandas as pd

from ..models.effects import (
    AbilityInteractionPayload,
    AccuracyRulePayload,
    Chance,
    CritRulePayload,
    DamageLimitPayload,
    DamagePayload,
    DrainPayload,
    EffectClass,
    EffectComponent,
    FaintPayload,
    HealPayload,
    HitPatternPayload,
    ItemInteractionPayload,
    MiscPayload,
    ProtectionInteractionPayload,
    RecoilPayload,
    RulePayload,
    StatStagePayload,
    StatusCurePayload,
    StatusPayload,
    SwitchPayload,
    TargetingRulePayload,
    Target,
    TrapPayload,
    TurnPatternPayload,
)

_TAG_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\{[^}]+\}")
_SPACES_PATTERN = re.compile(r"\s+")

_CHANCE_PREFIX_PATTERN = re.compile(
    r"has an? (?P<chance>\$effect_chance%|\d+%) chance to"
)
_STATUS_CHANCE_PATTERN = re.compile(
    r"has an? (?P<chance>\$effect_chance%|\d+%) chance to (?P<action>poison|burn|freeze|paraly(?:ze|sis)|confuse) the target"
)
_FLINCH_CHANCE_PATTERN = re.compile(
    r"has an? (?P<chance>\$effect_chance%|\d+%) chance to make the target flinch"
)
_STAGE_CHANCE_PATTERN = re.compile(
    r"has an? (?P<chance>\$effect_chance%|\d+%) chance to "
    r"(?P<direction>raise|lower) (?:the )?(?P<owner>user(?:'s)?|target(?:'s)?) "
    r"(?P<stats>attack|defense|special attack|special defense|speed|accuracy|evasion) by "
    r"(?P<delta>one|two) stage"
)

_STAGE_DIRECT_PATTERN = re.compile(
    r"(?P<direction>raise|lower)s? "
    r"(?:the )?(?P<owner>user(?:'s)?|target(?:'s)?) "
    r"(?P<stats>attack|defense|special attack|special defense|speed|accuracy|evasion) by "
    r"(?P<delta>one|two) stage"
)


def _normalize_text(value: str) -> str:
    text = _TAG_LINK_PATTERN.sub(r"\1", value or "")
    text = text.replace("Pok\u00e9mon", "pokemon")
    text = text.replace("\u2019", "'")
    text = text.lower().strip()
    return _SPACES_PATTERN.sub(" ", text)


def _parse_chance(token: str | None) -> Chance:
    if token is None:
        return None
    token = token.strip()
    if token == "$effect_chance%":
        return "effect_chance"
    if token.endswith("%") and token[:-1].isdigit():
        return int(token[:-1]) / 100.0
    return None


def _stage_word_to_int(value: str) -> int:
    return {"one": 1, "two": 2}[value]


def _stat_key(value: str) -> str:
    return {
        "attack": "attack",
        "defense": "defense",
        "special attack": "special_attack",
        "special defense": "special_defense",
        "speed": "speed",
        "accuracy": "accuracy",
        "evasion": "evasion",
    }[value]


def _contains_any(text: str, parts: list[str]) -> bool:
    return any(part in text for part in parts)


def _new_component(
    payload: object, target: Target = None, chance: Chance = None
) -> EffectComponent:
    return EffectComponent(payload=payload, target=target, chance=chance)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _parse_move_ids(value: object) -> set[int]:
    text = _safe_text(value).strip()
    if not text:
        return set()

    move_ids: set[int] = set()
    for part in text.split("|"):
        part = part.strip()
        if part.isdigit():
            move_ids.add(int(part))
    return move_ids


def parse_effect_row(row: pd.Series | dict[str, object]) -> EffectClass:
    """Parse one filtered effect row into a typed EffectClass."""

    if isinstance(row, dict):
        row_data = row
    else:
        row_data = row.to_dict()

    effect_id = int(row_data["effect_id"])
    effect_key = str(row_data.get("effect_identifier", effect_id))
    raw_short = _safe_text(row_data.get("short_effect", ""))
    raw_effect = _safe_text(row_data.get("effect", ""))
    move_ids = _parse_move_ids(row_data.get("move_ids", ""))

    short_text = _normalize_text(raw_short)
    full_text = _normalize_text(raw_effect)
    text = f"{short_text} {full_text}".strip()

    components: list[EffectComponent] = []
    seen: set[tuple[object, ...]] = set()

    def add_component(component: EffectComponent) -> None:
        payload_data = tuple(sorted(asdict(component.payload).items()))
        key = (
            type(component.payload).__name__,
            payload_data,
            component.target,
            component.chance,
        )
        if key in seen:
            return
        seen.add(key)
        components.append(component)

    # Damage family.
    if "one-hit ko" in text:
        add_component(_new_component(DamagePayload(mode="ohko"), target="opponent"))

    if "equal to the user's level" in text:
        add_component(
            _new_component(
                DamagePayload(mode="fixed", fixed_source="user_level"),
                target="opponent",
            )
        )

    if "equal to half the target's hp" in text:
        add_component(
            _new_component(
                DamagePayload(mode="fixed", fixed_source="target_hp_half"),
                target="opponent",
            )
        )

    if "user's type changes to match the target's" in text or "user becomes the target's type" in text:
        add_component(_new_component(RulePayload(rule="user_type_matches_target"), target="self"))

    if "exchanges the user's speed with the target's" in text or "exchanges the original speed stats of the user and target" in text:
        add_component(_new_component(RulePayload(rule="swap_user_target_speed"), target="both"))

    if "inflicts 40 points of damage" in text:
        add_component(_new_component(RulePayload(rule="fixed_damage_40"), target="opponent"))
    if "inflicts 20 points of damage" in text:
        add_component(_new_component(RulePayload(rule="fixed_damage_20"), target="opponent"))

    scaling_rules = [
        (
            "more damage when the user has less hp remaining",
            "lower_user_hp_higher_power",
        ),
        (
            "more damage when the user has more hp remaining",
            "higher_user_hp_higher_power",
        ),
        (
            "power increases against targets with more hp remaining",
            "higher_target_hp_higher_power",
        ),
        (
            "more damage to heavier targets",
            "heavier_target_higher_power",
        ),
        (
            "power is higher when the user weighs more than the target",
            "heavier_user_vs_target_higher_power",
        ),
        (
            "power doubles every turn this move is used in succession",
            "successive_use_power_doubles",
        ),
        (
            "power is higher the more the user's stats have been raised",
            "higher_user_stat_boost_higher_power",
        ),
        (
            "inflicts double damage if the user takes damage before attacking this turn",
            "double_if_user_hit_before_attacking",
        ),
        (
            "if target has a berry, inflicts double damage",
            "double_if_target_has_berry",
        ),
    ]
    for pattern, rule in scaling_rules:
        if pattern in text:
            add_component(
                _new_component(
                    DamagePayload(mode="scaling", scaling_rule=rule),
                    target="opponent",
                )
            )

    if _contains_any(
        text,
        [
            "inflicts regular damage",
            "inflicts damage",
            "hits twice in one turn",
            "hits 2-5 times in one turn",
            "drains half the damage inflicted",
            "drains 75% of the damage inflicted",
            "hits every turn for 2-3 turns",
            "causes a one-hit ko",
            "inflicts damage, and the user takes damage",
        ],
    ):
        add_component(_new_component(DamagePayload(mode="regular"), target="opponent"))

    # Status family.
    if "puts the target to sleep" in text:
        add_component(_new_component(StatusPayload(status="sleep"), target="opponent"))
    if "user sleeps for two turns" in text:
        add_component(_new_component(StatusPayload(status="sleep"), target="self"))
    if "poisons the target" in text:
        add_component(_new_component(StatusPayload(status="poison"), target="opponent"))
    if "burns the target" in text:
        add_component(_new_component(StatusPayload(status="burn"), target="opponent"))
    if "paralyzes the target" in text:
        add_component(
            _new_component(StatusPayload(status="paralysis"), target="opponent")
        )
    if "confuses the target" in text:
        add_component(
            _new_component(StatusPayload(status="confusion"), target="opponent")
        )

    for match in _STATUS_CHANCE_PATTERN.finditer(text):
        action = match.group("action")
        chance = _parse_chance(match.group("chance"))
        mapped_status = {
            "poison": "poison",
            "burn": "burn",
            "freeze": "freeze",
            "paralyze": "paralysis",
            "paralysis": "paralysis",
            "confuse": "confusion",
        }[action]
        add_component(
            _new_component(
                StatusPayload(status=mapped_status), target="opponent", chance=chance
            )
        )

    for match in _FLINCH_CHANCE_PATTERN.finditer(text):
        chance = _parse_chance(match.group("chance"))
        add_component(
            _new_component(
                StatusPayload(status="flinch"), target="opponent", chance=chance
            )
        )

    # Stat stage changes.
    if "raises all of the user's stats by one stage" in text:
        add_component(
            _new_component(
                StatStagePayload(stat="all_stats", delta=1),
                target="self",
            )
        )

    if "raises the user's attack and special attack by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="attack", delta=1), target="self")
        )
        add_component(
            _new_component(StatStagePayload(stat="special_attack", delta=1), target="self")
        )

    if "raises the user's attack and defense by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="attack", delta=1), target="self")
        )
        add_component(
            _new_component(StatStagePayload(stat="defense", delta=1), target="self")
        )

    if "raises the user's special attack and special defense by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="special_attack", delta=1), target="self")
        )
        add_component(
            _new_component(StatStagePayload(stat="special_defense", delta=1), target="self")
        )

    if "raises the user's attack and speed by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="attack", delta=1), target="self")
        )
        add_component(_new_component(StatStagePayload(stat="speed", delta=1), target="self"))

    if "raises the user's attack and accuracy by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="attack", delta=1), target="self")
        )
        add_component(
            _new_component(StatStagePayload(stat="accuracy", delta=1), target="self")
        )

    if "raises the user's defense by three stages" in text:
        add_component(
            _new_component(StatStagePayload(stat="defense", delta=3), target="self")
        )

    if "lowers the target's attack and defense by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="attack", delta=-1), target="opponent")
        )
        add_component(
            _new_component(StatStagePayload(stat="defense", delta=-1), target="opponent")
        )

    if "lowers the target's attack and special attack by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="attack", delta=-1), target="opponent")
        )
        add_component(
            _new_component(
                StatStagePayload(stat="special_attack", delta=-1),
                target="opponent",
            )
        )

    if "lowers the target's attack, special attack, and speed by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="attack", delta=-1), target="opponent")
        )
        add_component(
            _new_component(
                StatStagePayload(stat="special_attack", delta=-1),
                target="opponent",
            )
        )
        add_component(_new_component(StatStagePayload(stat="speed", delta=-1), target="opponent"))

    if (
        "raises user's attack, special attack, and speed by two stages" in text
        or "raises the user's attack, special attack, and speed by two stages" in text
    ):
        add_component(
            _new_component(StatStagePayload(stat="attack", delta=2), target="self")
        )
        add_component(
            _new_component(StatStagePayload(stat="special_attack", delta=2), target="self")
        )
        add_component(_new_component(StatStagePayload(stat="speed", delta=2), target="self"))

    if "lower user's defense and special defense by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="defense", delta=-1), target="self")
        )
        add_component(
            _new_component(
                StatStagePayload(stat="special_defense", delta=-1),
                target="self",
            )
        )

    if "raises one of a friendly pokemon's stats at random by two stages" in text:
        add_component(
            _new_component(StatStagePayload(stat="all_stats", delta=2), target="self")
        )

    if "lowers opponents' speed by one stage upon switching in" in text:
        add_component(_new_component(StatStagePayload(stat="speed", delta=-1), target="opponent"))

    if "raises a selected ally's special defense by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="special_defense", delta=1), target="self")
        )

    if "raises the attack and special attack of all grass pokemon in battle" in text:
        add_component(_new_component(StatStagePayload(stat="attack", delta=1), target="all"))
        add_component(
            _new_component(StatStagePayload(stat="special_attack", delta=1), target="all")
        )

    if "raises the user's special attack, special defense, and speed by one stage each" in text:
        add_component(
            _new_component(StatStagePayload(stat="special_attack", delta=1), target="self")
        )
        add_component(
            _new_component(
                StatStagePayload(stat="special_defense", delta=1),
                target="self",
            )
        )
        add_component(_new_component(StatStagePayload(stat="speed", delta=1), target="self"))

    if "raises the user's attack, defense, and accuracy by one stage each" in text:
        add_component(
            _new_component(StatStagePayload(stat="attack", delta=1), target="self")
        )
        add_component(
            _new_component(StatStagePayload(stat="defense", delta=1), target="self")
        )
        add_component(
            _new_component(StatStagePayload(stat="accuracy", delta=1), target="self")
        )

    if "raises the user's defense and special defense by one stage" in text:
        add_component(
            _new_component(StatStagePayload(stat="defense", delta=1), target="self")
        )
        add_component(
            _new_component(
                StatStagePayload(stat="special_defense", delta=1), target="self"
            )
        )

    for match in _STAGE_CHANCE_PATTERN.finditer(text):
        direction = match.group("direction")
        owner = match.group("owner")
        delta = _stage_word_to_int(match.group("delta"))
        stat = _stat_key(match.group("stats"))
        chance = _parse_chance(match.group("chance"))

        if direction == "lower":
            delta = -delta
        target = "self" if "user" in owner else "opponent"
        add_component(
            _new_component(
                StatStagePayload(stat=stat, delta=delta),
                target=target,
                chance=chance,
            )
        )

    for match in _STAGE_DIRECT_PATTERN.finditer(text):
        prefix = text[max(0, match.start() - 30) : match.start()]
        if "chance to" in prefix:
            continue

        direction = match.group("direction")
        owner = match.group("owner")
        delta = _stage_word_to_int(match.group("delta"))
        stat = _stat_key(match.group("stats"))

        if direction == "lower":
            delta = -delta
        target = "self" if "user" in owner else "opponent"
        add_component(
            _new_component(StatStagePayload(stat=stat, delta=delta), target=target)
        )

    # Heal/drain/recoil family.
    if "heals the user by half its max hp" in text:
        add_component(
            _new_component(HealPayload(ratio=0.5, base="max_hp"), target="self")
        )
    if "completely healing itself" in text:
        add_component(_new_component(HealPayload(ratio=1.0, base="max_hp"), target="self"))
    if "heals the target for half its max hp" in text:
        add_component(
            _new_component(HealPayload(ratio=0.5, base="max_hp"), target="opponent")
        )
    if "restores 1/16 of the user's max hp each turn" in text:
        add_component(_new_component(HealPayload(ratio=1 / 16, base="max_hp"), target="self"))
    if "target loses 1/4 its max hp every turn as long as it's asleep" in text:
        add_component(_new_component(RecoilPayload(ratio=0.25, base="max_hp"), target="opponent"))
        add_component(_new_component(RulePayload(rule="sleep_required_for_dot"), target="opponent"))
    if "seeds the target, stealing hp from it every turn" in text:
        add_component(_new_component(RulePayload(rule="leech_seed"), target="opponent"))
    if "drains half the damage inflicted to heal the user" in text:
        add_component(
            _new_component(
                DrainPayload(ratio=0.5, base="damage_inflicted"), target="self"
            )
        )
    if "drains 75% of the damage inflicted to heal the user" in text:
        add_component(
            _new_component(
                DrainPayload(ratio=0.75, base="damage_inflicted"), target="self"
            )
        )

    if "user receives 1/4 the damage it inflicts in recoil" in text:
        add_component(
            _new_component(
                RecoilPayload(ratio=0.25, base="damage_inflicted"), target="self"
            )
        )
    if "user receives 1/3 the damage inflicted in recoil" in text:
        add_component(
            _new_component(
                RecoilPayload(ratio=1 / 3, base="damage_inflicted"), target="self"
            )
        )
    if "user receives 1/2 the damage inflicted in recoil" in text:
        add_component(
            _new_component(
                RecoilPayload(ratio=0.5, base="damage_inflicted"), target="self"
            )
        )
    if (
        "if the user misses, it takes half the damage it would have inflicted in recoil"
        in text
    ):
        add_component(
            _new_component(
                RecoilPayload(ratio=0.5, base="missed_damage"), target="self"
            )
        )
    if "user takes damage equal to half of its max hp" in text:
        add_component(
            _new_component(RecoilPayload(ratio=0.5, base="max_hp"), target="self")
        )
    if "user pays half its max hp" in text:
        add_component(
            _new_component(RecoilPayload(ratio=0.5, base="max_hp"), target="self")
        )

    # Hit/turn family.
    if "hits twice in one turn" in text:
        add_component(
            _new_component(
                HitPatternPayload(mode="fixed", fixed_hits=2), target="opponent"
            )
        )
    if "hits 2-5 times in one turn" in text:
        add_component(
            _new_component(
                HitPatternPayload(mode="range", min_hits=2, max_hits=5),
                target="opponent",
            )
        )
    if "hits every turn for 2-3 turns" in text:
        add_component(
            _new_component(
                TurnPatternPayload(active_turns_min=2, active_turns_max=3),
                target="opponent",
            )
        )
    if "inflicts damage for 2-5 turns" in text:
        add_component(
            _new_component(
                TurnPatternPayload(active_turns_min=2, active_turns_max=5),
                target="opponent",
            )
        )
    if "user foregoes its next turn to recharge" in text:
        add_component(
            _new_component(TurnPatternPayload(recharge_turns=1), target="self")
        )
    if "for five turns" in text:
        add_component(
            _new_component(TurnPatternPayload(active_turns_min=5, active_turns_max=5), target="field")
        )
    if "for three turns" in text:
        add_component(
            _new_component(TurnPatternPayload(active_turns_min=3, active_turns_max=3), target="field")
        )
    if "for the next few turns" in text:
        add_component(
            _new_component(TurnPatternPayload(active_turns_min=2, active_turns_max=5), target="field")
        )
    if "requires a turn to charge before attacking" in text:
        add_component(_new_component(TurnPatternPayload(charge_turns=1), target="self"))
    if "hits the target two turns later" in text:
        add_component(
            _new_component(TurnPatternPayload(delay_turns=2), target="opponent")
        )
    if "user vanishes, dodging all attacks, and hits next turn" in text:
        add_component(_new_component(TurnPatternPayload(charge_turns=1), target="self"))

    # Accuracy/crit family.
    if "never misses" in text:
        add_component(
            _new_component(AccuracyRulePayload(rule="never_miss"), target="opponent")
        )
    if "ensures that the user's next move will hit the target" in text:
        add_component(
            _new_component(AccuracyRulePayload(rule="next_move_hits"), target="self")
        )
    if "has an increased chance for a critical hit" in text:
        add_component(_new_component(CritRulePayload(rule="high_crit"), target="self"))
    if "increases the user's chance to score a critical hit" in text:
        add_component(_new_component(CritRulePayload(rule="high_crit"), target="self"))
    if "always scores a critical hit" in text:
        add_component(
            _new_component(CritRulePayload(rule="always_crit"), target="self")
        )
    if "guarantees a critical hit with the user's next move" in text:
        add_component(_new_component(CritRulePayload(rule="always_crit"), target="self"))

    # Trap/switch/item families.
    if "prevents the target from fleeing" in text:
        add_component(
            _new_component(TrapPayload(rule="prevent_escape"), target="opponent")
        )
    if "prevents the target from leaving battle" in text or "traps the target" in text:
        add_component(
            _new_component(TrapPayload(rule="trap_target"), target="opponent")
        )

    if "forces trainers to switch pokemon" in text:
        add_component(
            _new_component(SwitchPayload(rule="force_target_switch"), target="opponent")
        )
    if "user must switch out after attacking" in text:
        add_component(
            _new_component(SwitchPayload(rule="force_self_switch"), target="self")
        )
    if "makes the user switch out" in text:
        add_component(
            _new_component(SwitchPayload(rule="force_self_switch"), target="self")
        )
    if "ends wild battles" in text or "immediately ends wild battles" in text:
        add_component(_new_component(SwitchPayload(rule="flee_wild"), target="self"))
    if "allows the trainer to switch out the user and pass effects along to its replacement" in text:
        add_component(
            _new_component(SwitchPayload(rule="switch_self_pass_effects"), target="self")
        )
    if "user switches places with the friendly pokemon opposite it" in text:
        add_component(_new_component(SwitchPayload(rule="switch_with_ally"), target="self"))

    if "takes the target's item" in text:
        add_component(
            _new_component(ItemInteractionPayload(rule="steal_item"), target="opponent")
        )
    if "user and target swap items" in text:
        add_component(
            _new_component(ItemInteractionPayload(rule="swap_items"), target="both")
        )
    if "if target has a berry" in text:
        add_component(
            _new_component(
                ItemInteractionPayload(rule="consume_target_berry"), target="opponent"
            )
        )
    if "holding a appropriate plate or drive" in text:
        add_component(
            _new_component(
                ItemInteractionPayload(rule="type_from_held_item"), target="self"
            )
        )
    if "gives the user's held item to the target" in text:
        add_component(
            _new_component(ItemInteractionPayload(rule="give_item_to_target"), target="opponent")
        )
    if "user recovers the item it last used up" in text:
        add_component(
            _new_component(
                ItemInteractionPayload(rule="recover_last_used_item"),
                target="self",
            )
        )
    if "target cannot use held items" in text:
        add_component(
            _new_component(ItemInteractionPayload(rule="disable_target_items"), target="opponent")
        )
    if "negates held items for five turns" in text:
        add_component(
            _new_component(
                ItemInteractionPayload(rule="negate_all_items_temporary"),
                target="field",
            )
        )

    # Protection/targeting/ability interactions.
    if "prevents any moves from hitting the user this turn" in text:
        add_component(
            _new_component(
                ProtectionInteractionPayload(rule="self_protect"), target="self"
            )
        )
    if "ignores and destroys protection effects" in text:
        add_component(
            _new_component(
                ProtectionInteractionPayload(rule="break_protect"), target="opponent"
            )
        )
    if "hits through protect and detect" in text:
        add_component(
            _new_component(
                ProtectionInteractionPayload(rule="bypass_protect"), target="opponent"
            )
        )
    if "destroys reflect and light screen" in text:
        add_component(
            _new_component(
                ProtectionInteractionPayload(rule="break_screens"), target="opponent"
            )
        )
    if "prevents any multi-target moves from hitting friendly pokemon this turn" in text:
        add_component(
            _new_component(
                ProtectionInteractionPayload(rule="ally_wide_guard"),
                target="self",
            )
        )
    if "prevents any priority moves from hitting friendly pokemon this turn" in text:
        add_component(
            _new_component(
                ProtectionInteractionPayload(rule="ally_quick_guard"),
                target="self",
            )
        )

    if "forces the target to have no evade" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="ignore_target_evasion"), target="opponent"
            )
        )
    if "allows it to be hit by normal and fighting moves even if it's a ghost" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="ghost_hit_by_normal_fighting"),
                target="opponent",
            )
        )
    if "redirects the target's single-target effects to the user" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="redirect_single_target_to_user"),
                target="self",
            )
        )
    if "ignores the target's stat modifiers" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="ignore_target_stat_modifiers"),
                target="opponent",
            )
        )
    if "uses the target's last used move" in text:
        add_component(
            _new_component(TargetingRulePayload(rule="copy_target_last_move"), target="opponent")
        )
    if "randomly selects and uses any move in the game" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="metronome_random_any_move"),
                target="self",
            )
        )
    if "randomly uses one of the user's other three moves" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="sleep_random_user_move"),
                target="self",
            )
        )
    if "inflicts twice the damage the user received from the last physical hit it took" in text:
        add_component(
            _new_component(TargetingRulePayload(rule="counter_physical"), target="opponent")
        )
    if "inflicts twice the damage the user received from the last special hit it took" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="mirror_coat_special_counter"),
                target="opponent",
            )
        )
    if "copies the target's last used move" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="mirror_move_last_target_move"),
                target="opponent",
            )
        )
    if "copies the target's ability" in text:
        add_component(
            _new_component(TargetingRulePayload(rule="copy_target_ability"), target="opponent")
        )
    if "discards the user's stat changes and copies the target's" in text:
        add_component(
            _new_component(TargetingRulePayload(rule="copy_target_stat_changes"), target="opponent")
        )
    if "copies the user's ability onto the target" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="copy_user_ability_to_target"),
                target="opponent",
            )
        )
    if "changes the target's ability to insomnia" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="copy_target_ability_to_user"),
                target="opponent",
            )
        )
    if "changes the target's ability to simple" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="copy_target_ability_to_user"),
                target="opponent",
            )
        )
    if "transfers the user's major status effect to the target" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="copy_user_status_to_target"),
                target="opponent",
            )
        )
    if "forces the target to repeat its last used move every turn" in text:
        add_component(
            _new_component(TargetingRulePayload(rule="repeat_target_last_move"), target="opponent")
        )
    if "disables the target's last used move" in text:
        add_component(
            _new_component(TargetingRulePayload(rule="disable_target_last_move"), target="opponent")
        )
    if "prevents the target from using the same move twice in a row" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="prevent_target_same_move_twice"),
                target="opponent",
            )
        )
    if "steals the target's move, if it's self-targeted" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="steal_target_self_move"),
                target="opponent",
            )
        )
    if "uses the target's move against it before it attacks" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="strike_before_target_with_targets_move"),
                target="opponent",
            )
        )
    if "for five turns, slower pokemon will act before faster pokemon" in text:
        add_component(
            _new_component(TargetingRulePayload(rule="switch_turn_order"), target="field")
        )
    if "prevents target from restoring its hp for five turns" in text:
        add_component(
            _new_component(TargetingRulePayload(rule="target_cannot_restore_hp"), target="opponent")
        )
    if "inflicts damage based on the target's defense, not special defense" in text:
        add_component(
            _new_component(
                TargetingRulePayload(rule="use_target_defense_stat"), target="opponent"
            )
        )

    if "cannot be disrupted by abilities" in text:
        add_component(
            _new_component(
                AbilityInteractionPayload(rule="ignores_abilities"), target="self"
            )
        )
    if "nullifies target's ability until it leaves battle" in text:
        add_component(
            _new_component(AbilityInteractionPayload(rule="ignores_abilities"), target="opponent")
        )
    if "user and target swap abilities" in text:
        add_component(_new_component(RulePayload(rule="swap_abilities"), target="both"))

    # Cures and constraints.
    if "cures the entire party of major status effects" in text:
        add_component(
            _new_component(
                StatusCurePayload(rule="cure_party_major_status"), target="party"
            )
        )
    if "lets frozen pokemon thaw themselves" in text:
        add_component(
            _new_component(StatusCurePayload(rule="cure_self_freeze"), target="self")
        )
    if "cleanses the user of a burn, paralysis, or poison" in text:
        add_component(
            _new_component(StatusCurePayload(rule="cure_self_major_status"), target="self")
        )
    if "cannot lower the target's hp below 1" in text:
        add_component(
            _new_component(
                DamageLimitPayload(rule="cannot_reduce_target_hp_below_1"),
                target="opponent",
            )
        )
    if "user faints" in text:
        add_component(_new_component(FaintPayload(rule="user_faints"), target="self"))
    if "user and target both faint after three turns" in text:
        add_component(_new_component(FaintPayload(rule="user_faints"), target="both"))

    # Generic field rules for common weather/terrain/screen effects.
    if "resets all pokemon's stats, accuracy, and evasion" in text:
        add_component(_new_component(RulePayload(rule="reset_all_stat_modifiers"), target="all"))
    if "user waits for two turns, then hits back for twice the damage it took" in text:
        add_component(_new_component(RulePayload(rule="bide_counter"), target="opponent"))
    if "transfers 1/4 of the user's max hp into a doll" in text:
        add_component(_new_component(RulePayload(rule="substitute"), target="self"))
        add_component(_new_component(RecoilPayload(ratio=0.25, base="max_hp"), target="self"))
    if "sets the user's and targets's hp to the average of their current hp" in text:
        add_component(_new_component(RulePayload(rule="set_both_hp_to_average"), target="both"))
    if "lowers the pp of the target's last used move by 4" in text:
        add_component(_new_component(RulePayload(rule="lower_target_last_move_pp_4"), target="opponent"))
    if "prevents the user's hp from lowering below 1 this turn" in text:
        add_component(_new_component(RulePayload(rule="endure"), target="self"))
    if "target falls in love if it has the opposite gender" in text:
        add_component(_new_component(RulePayload(rule="infatuation"), target="opponent"))
    if "hits once for every conscious pokemon the trainer has" in text:
        add_component(_new_component(RulePayload(rule="hits_per_healthy_party_member"), target="opponent"))
    if "recovers 1/4 hp after one stockpile" in text:
        add_component(_new_component(RulePayload(rule="swallow_heal_scales_with_stockpile"), target="self"))
    if "uses a move which depends upon the terrain" in text:
        add_component(_new_component(RulePayload(rule="terrain_dependent_move"), target="opponent"))
    if "for the next few turns, the target can only use damaging moves" in text:
        add_component(_new_component(RulePayload(rule="target_can_only_use_damaging_moves"), target="opponent"))
    if "ally's next move inflicts half more damage" in text:
        add_component(_new_component(RulePayload(rule="boost_ally_next_move"), target="self"))
    if "randomly selects and uses one of the trainer's other pokemon's moves" in text:
        add_component(_new_component(RulePayload(rule="random_move_from_party"), target="self"))
    if "prevents the user from leaving battle" in text:
        add_component(_new_component(TrapPayload(rule="prevent_escape"), target="self"))
    if "reflects back the first effect move used on the user this turn" in text:
        add_component(_new_component(RulePayload(rule="reflect_first_status_move"), target="self"))
    if "lowers the target's hp to equal the user's" in text:
        add_component(_new_component(RulePayload(rule="set_target_hp_to_user_hp"), target="opponent"))
    if "prevents the target from using any moves that the user also knows" in text:
        add_component(_new_component(RulePayload(rule="imprison_shared_moves"), target="opponent"))
    if "halves all electric-type damage" in text:
        add_component(_new_component(RulePayload(rule="halve_electric_damage"), target="field"))
    if "halves all fire-type damage" in text:
        add_component(_new_component(RulePayload(rule="halve_fire_damage"), target="field"))
    if "user's type changes to match the terrain" in text:
        add_component(_new_component(RulePayload(rule="user_type_matches_terrain"), target="self"))
    if "allows it to be hit by psychic moves even if it's dark" in text:
        add_component(_new_component(RulePayload(rule="dark_hit_by_psychic"), target="opponent"))
    if "strikes back at the last pokemon to hit the user this turn with 1.5× the damage" in text:
        add_component(_new_component(RulePayload(rule="metal_burst_counter"), target="opponent"))
    if "user swaps attack and defense" in text:
        add_component(_new_component(RulePayload(rule="swap_user_attack_defense"), target="self"))
    if "user swaps attack and special attack changes with the target" in text:
        add_component(_new_component(RulePayload(rule="swap_atk_spatk_stage_changes"), target="both"))
    if "user swaps defense and special defense changes with the target" in text:
        add_component(_new_component(RulePayload(rule="swap_def_spdef_stage_changes"), target="both"))
    if "averages defense and special defense with the target" in text:
        add_component(_new_component(RulePayload(rule="average_def_spdef"), target="both"))
    if "averages attack and special attack with the target" in text:
        add_component(_new_component(RulePayload(rule="average_atk_spatk"), target="both"))
    if "changes the target's type to water" in text:
        add_component(_new_component(RulePayload(rule="target_type_to_water"), target="opponent"))
    if "makes the target act next this turn" in text:
        add_component(_new_component(RulePayload(rule="target_moves_next"), target="opponent"))
    if "makes the target act last this turn" in text:
        add_component(_new_component(RulePayload(rule="target_moves_last"), target="opponent"))
    if "raises the attack and special attack of all" in text and "pokemon in battle" in text:
        add_component(_new_component(StatStagePayload(stat="attack", delta=1), target="all"))
        add_component(_new_component(StatStagePayload(stat="special_attack", delta=1), target="all"))
    if "covers the opposing field, lowering opponents' speed by one stage upon switching in" in text:
        add_component(_new_component(StatStagePayload(stat="speed", delta=-1), target="opponent"))
    if "protects pokemon on the ground from priority moves" in text:
        add_component(_new_component(ProtectionInteractionPayload(rule="ally_quick_guard"), target="self"))
    if "increases the power of their psychic moves by 50%" in text:
        add_component(_new_component(RulePayload(rule="psychic_terrain_power_boost"), target="field"))
    if "reduces damage five turns, but must be used during hail" in text:
        add_component(_new_component(RulePayload(rule="aurora_veil_hail_required"), target="field"))
    if "changes the weather to rain for five turns" in text:
        add_component(_new_component(RulePayload(rule="weather_rain_5_turns"), target="field"))
    if "changes the weather to sunny for five turns" in text:
        add_component(_new_component(RulePayload(rule="weather_sun_5_turns"), target="field"))
    if "changes the weather to a sandstorm for five turns" in text:
        add_component(_new_component(RulePayload(rule="weather_sandstorm_5_turns"), target="field"))
    if "changes the weather to a hailstorm for five turns" in text:
        add_component(_new_component(RulePayload(rule="weather_hail_5_turns"), target="field"))
    if "reduces damage from special attacks by 50% for five turns" in text:
        add_component(_new_component(RulePayload(rule="special_damage_halved_5_turns"), target="field"))
    if "reduces damage from physical attacks by half" in text:
        add_component(_new_component(RulePayload(rule="physical_damage_halved_5_turns"), target="field"))
    if "scatters spikes, hurting opposing pokemon that switch in" in text:
        add_component(_new_component(RulePayload(rule="entry_hazard_spikes"), target="opponent"))
    if "scatters poisoned spikes, poisoning opposing pokemon that switch in" in text:
        add_component(_new_component(RulePayload(rule="entry_hazard_toxic_spikes"), target="opponent"))
    if "causes damage when opposing pokemon switch in" in text:
        add_component(_new_component(RulePayload(rule="entry_hazard_rocks"), target="opponent"))
    if "does nothing" in text:
        add_component(_new_component(RulePayload(rule="no_effect"), target="self"))

    # Fallbacks for effects with missing prose in this dataset.
    if not text and 776 in move_ids:
        add_component(_new_component(DamagePayload(mode="regular"), target="opponent"))
        add_component(_new_component(RulePayload(rule="damage_uses_user_defense_stat"), target="self"))

    if not text and 791 in move_ids:
        add_component(_new_component(HealPayload(ratio=0.25, base="max_hp"), target="all"))

    # Fallback.
    if not components:
        components.append(
            _new_component(
                MiscPayload(rule="unparsed", text=raw_short or raw_effect),
                target=None,
            )
        )

    first_payload = components[0].payload
    confidence = (
        0.25
        if getattr(first_payload, "family", None) == "misc"
        and getattr(first_payload, "op", None) == "unparsed"
        else 1.0
    )

    return EffectClass(
        effect_id=effect_id,
        effect_key=effect_key,
        components=components,
        raw_short_effect=raw_short,
        raw_effect=raw_effect,
        confidence=confidence,
    )


def parse_effects_dataframe(table: pd.DataFrame) -> list[EffectClass]:
    """Parse each row in an effects DataFrame into typed effect classes."""

    return [parse_effect_row(row) for _, row in table.iterrows()]
