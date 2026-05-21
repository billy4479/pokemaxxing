"""Typed classes for simplified move-effect interpretation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias, Union

Target: TypeAlias = Literal["self", "opponent", "both", "all", "field", "party"] | None
Chance: TypeAlias = float | Literal["effect_chance"] | None

Family: TypeAlias = Literal[
    "damage",
    "status",
    "stat",
    "hp",
    "timing",
    "accuracy",
    "crit",
    "trap",
    "switch",
    "item",
    "protection",
    "targeting",
    "ability",
    "cure",
    "constraint",
    "rule",
    "misc",
]


@dataclass(frozen=True)
class EffectPayload:
    """Canonical payload representation used by the parser outputs."""

    family: Family
    op: str
    value: float | int | None = None
    unit: str | None = None
    min_value: int | None = None
    max_value: int | None = None
    tags: tuple[str, ...] = ()
    details: tuple[tuple[str, str], ...] = ()
    text: str | None = None


# Compatibility payload classes used by parser rules. They are converted
# to EffectPayload in EffectComponent.__post_init__.
@dataclass(frozen=True)
class DamagePayload:
    mode: Literal["regular", "ohko", "fixed", "scaling"]
    fixed_source: Literal["user_level", "target_hp_half"] | None = None
    scaling_rule: str | None = None


@dataclass(frozen=True)
class StatusPayload:
    status: Literal[
        "sleep", "poison", "burn", "freeze", "paralysis", "confusion", "flinch"
    ]


@dataclass(frozen=True)
class StatStagePayload:
    stat: Literal[
        "attack",
        "defense",
        "special_attack",
        "special_defense",
        "speed",
        "accuracy",
        "evasion",
        "all_stats",
    ]
    delta: int


@dataclass(frozen=True)
class HealPayload:
    ratio: float
    base: Literal["max_hp"]


@dataclass(frozen=True)
class DrainPayload:
    ratio: float
    base: Literal["damage_inflicted"]


@dataclass(frozen=True)
class RecoilPayload:
    ratio: float
    base: Literal["damage_inflicted", "max_hp", "missed_damage"]


@dataclass(frozen=True)
class HitPatternPayload:
    mode: Literal["fixed", "range", "locked_turns"]
    fixed_hits: int | None = None
    min_hits: int | None = None
    max_hits: int | None = None


@dataclass(frozen=True)
class TurnPatternPayload:
    charge_turns: int = 0
    recharge_turns: int = 0
    delay_turns: int = 0
    active_turns_min: int | None = None
    active_turns_max: int | None = None


@dataclass(frozen=True)
class AccuracyRulePayload:
    rule: Literal["never_miss", "next_move_hits"]


@dataclass(frozen=True)
class CritRulePayload:
    rule: Literal["high_crit", "always_crit"]


@dataclass(frozen=True)
class TrapPayload:
    rule: Literal["prevent_escape", "trap_target"]


@dataclass(frozen=True)
class SwitchPayload:
    rule: Literal[
        "force_target_switch",
        "force_self_switch",
        "flee_wild",
        "switch_self_pass_effects",
        "switch_with_ally",
    ]


@dataclass(frozen=True)
class ItemInteractionPayload:
    rule: Literal[
        "steal_item",
        "swap_items",
        "consume_target_berry",
        "type_from_held_item",
        "give_item_to_target",
        "recover_last_used_item",
        "disable_target_items",
        "negate_all_items_temporary",
    ]


@dataclass(frozen=True)
class ProtectionInteractionPayload:
    rule: Literal[
        "break_protect",
        "bypass_protect",
        "self_protect",
        "break_screens",
        "ally_wide_guard",
        "ally_quick_guard",
    ]


@dataclass(frozen=True)
class TargetingRulePayload:
    rule: Literal[
        "ground_target",
        "ignore_target_evasion",
        "ghost_hit_by_normal_fighting",
        "redirect_single_target_to_user",
        "ignore_target_stat_modifiers",
        "use_target_defense_stat",
        "copy_target_last_move",
        "metronome_random_any_move",
        "sleep_random_user_move",
        "mirror_coat_special_counter",
        "counter_physical",
        "mirror_move_last_target_move",
        "copy_target_ability",
        "copy_target_stat_changes",
        "copy_user_ability_to_target",
        "copy_target_ability_to_user",
        "copy_user_status_to_target",
        "repeat_target_last_move",
        "disable_target_last_move",
        "prevent_target_same_move_twice",
        "steal_target_self_move",
        "strike_before_target_with_targets_move",
        "switch_turn_order",
        "target_cannot_restore_hp",
    ]


@dataclass(frozen=True)
class AbilityInteractionPayload:
    rule: Literal["ignores_abilities"]


@dataclass(frozen=True)
class StatusCurePayload:
    rule: Literal[
        "cure_party_major_status", "cure_self_freeze", "cure_self_major_status"
    ]


@dataclass(frozen=True)
class DamageLimitPayload:
    rule: Literal["cannot_reduce_target_hp_below_1"]


@dataclass(frozen=True)
class FaintPayload:
    rule: Literal["user_faints"]


@dataclass(frozen=True)
class RulePayload:
    rule: str


@dataclass(frozen=True)
class MiscPayload:
    rule: Literal["unparsed", "residual_text"]
    text: str


LegacyPayload: TypeAlias = Union[
    DamagePayload,
    StatusPayload,
    StatStagePayload,
    HealPayload,
    DrainPayload,
    RecoilPayload,
    HitPatternPayload,
    TurnPatternPayload,
    AccuracyRulePayload,
    CritRulePayload,
    TrapPayload,
    SwitchPayload,
    ItemInteractionPayload,
    ProtectionInteractionPayload,
    TargetingRulePayload,
    AbilityInteractionPayload,
    StatusCurePayload,
    DamageLimitPayload,
    FaintPayload,
    RulePayload,
    MiscPayload,
]

Payload: TypeAlias = EffectPayload


@dataclass(frozen=True)
class EffectComponent:
    payload: EffectPayload | LegacyPayload
    target: Target
    chance: Chance = None

    def __post_init__(self) -> None:
        if not isinstance(self.payload, EffectPayload):
            object.__setattr__(self, "payload", _to_effect_payload(self.payload))

    def __str__(self) -> str:
        target_label = self.target if self.target is not None else "any"
        chance_label = f", chance={self.chance}" if self.chance is not None else ""
        return f"{_payload_to_string(self.payload)} @ {target_label}{chance_label}"

    def to_markdown(self, index: int | None = None) -> str:
        """Render this component as one Markdown table row."""

        idx = "" if index is None else str(index)
        payload_label = _markdown_escape(_payload_to_markdown(self.payload))
        target_label = _markdown_escape(
            self.target if self.target is not None else "any"
        )
        chance_label = _markdown_escape(_chance_to_markdown(self.chance))
        return f"| {idx} | {payload_label} | {target_label} | {chance_label} |"


@dataclass(frozen=True)
class EffectClass:
    effect_id: int
    effect_key: str
    components: list[EffectComponent]
    raw_short_effect: str
    raw_effect: str
    confidence: float

    def __str__(self) -> str:
        components_label = "; ".join(str(component) for component in self.components)
        return (
            f"EffectClass(id={self.effect_id}, key={self.effect_key}, "
            f"confidence={self.confidence:.2f}, components=[{components_label}])"
        )

    def to_markdown(self, include_raw: bool = True) -> str:
        """Render this effect class as a Markdown section."""

        lines = [
            f"### EffectClass {self.effect_id} (`{_markdown_escape(self.effect_key)}`)",
            "",
            f"- Confidence: `{self.confidence:.2f}`",
            f"- Components: `{len(self.components)}`",
        ]

        if include_raw:
            lines.extend(
                [
                    f"- Short Effect: {_markdown_escape(self.raw_short_effect)}",
                    f"- Effect: {_markdown_escape(self.raw_effect)}",
                ]
            )

        lines.extend(
            [
                "",
                "| # | component | target | chance |",
                "| --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            component.to_markdown(index=i)
            for i, component in enumerate(self.components, start=1)
        )

        return "\n".join(lines)


def _details(**values: object) -> tuple[tuple[str, str], ...]:
    kept = [(key, str(value)) for key, value in values.items() if value is not None]
    return tuple(sorted(kept))


def _to_effect_payload(payload: LegacyPayload) -> EffectPayload:
    if isinstance(payload, DamagePayload):
        return EffectPayload(
            family="damage",
            op=payload.mode,
            details=_details(
                fixed_source=payload.fixed_source,
                scaling_rule=payload.scaling_rule,
            ),
        )

    if isinstance(payload, StatusPayload):
        return EffectPayload(family="status", op="inflict", tags=(payload.status,))

    if isinstance(payload, StatStagePayload):
        return EffectPayload(
            family="stat",
            op="stage_change",
            value=payload.delta,
            unit="stage",
            tags=(payload.stat,),
        )

    if isinstance(payload, HealPayload):
        return EffectPayload(
            family="hp",
            op="heal",
            value=payload.ratio,
            unit=payload.base,
        )

    if isinstance(payload, DrainPayload):
        return EffectPayload(
            family="hp",
            op="drain",
            value=payload.ratio,
            unit=payload.base,
        )

    if isinstance(payload, RecoilPayload):
        return EffectPayload(
            family="hp",
            op="recoil",
            value=payload.ratio,
            unit=payload.base,
        )

    if isinstance(payload, HitPatternPayload):
        return EffectPayload(
            family="timing",
            op=f"hit_{payload.mode}",
            value=payload.fixed_hits,
            unit="hits",
            min_value=payload.min_hits,
            max_value=payload.max_hits,
        )

    if isinstance(payload, TurnPatternPayload):
        return EffectPayload(
            family="timing",
            op="turn_pattern",
            details=_details(
                charge_turns=payload.charge_turns or None,
                recharge_turns=payload.recharge_turns or None,
                delay_turns=payload.delay_turns or None,
                active_turns_min=payload.active_turns_min,
                active_turns_max=payload.active_turns_max,
            ),
        )

    if isinstance(payload, AccuracyRulePayload):
        return EffectPayload(family="accuracy", op=payload.rule)

    if isinstance(payload, CritRulePayload):
        return EffectPayload(family="crit", op=payload.rule)

    if isinstance(payload, TrapPayload):
        return EffectPayload(family="trap", op=payload.rule)

    if isinstance(payload, SwitchPayload):
        return EffectPayload(family="switch", op=payload.rule)

    if isinstance(payload, ItemInteractionPayload):
        return EffectPayload(family="item", op=payload.rule)

    if isinstance(payload, ProtectionInteractionPayload):
        return EffectPayload(family="protection", op=payload.rule)

    if isinstance(payload, TargetingRulePayload):
        return EffectPayload(family="targeting", op=payload.rule)

    if isinstance(payload, AbilityInteractionPayload):
        return EffectPayload(family="ability", op=payload.rule)

    if isinstance(payload, StatusCurePayload):
        return EffectPayload(family="cure", op=payload.rule)

    if isinstance(payload, DamageLimitPayload):
        return EffectPayload(family="constraint", op=payload.rule)

    if isinstance(payload, FaintPayload):
        return EffectPayload(
            family="hp",
            op="faint",
            details=_details(rule=payload.rule),
        )

    if isinstance(payload, RulePayload):
        return EffectPayload(family="rule", op=payload.rule)

    if isinstance(payload, MiscPayload):
        return EffectPayload(family="misc", op=payload.rule, text=payload.text)

    return EffectPayload(family="misc", op="unknown", text=repr(payload))


def _chance_to_string(chance: Chance) -> str:
    if chance is None:
        return ""
    if isinstance(chance, float):
        return f"{chance:g}"
    return str(chance)


def _chance_to_markdown(chance: Chance) -> str:
    if chance is None:
        return ""
    if isinstance(chance, float):
        return f"{chance * 100:g}%"
    return "$effect_chance"


def _markdown_escape(value: object) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return text


def _format_value(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _payload_to_string(payload: EffectPayload) -> str:
    tags = f"[{','.join(payload.tags)}]" if payload.tags else ""
    range_part = ""
    if payload.min_value is not None and payload.max_value is not None:
        range_part = (
            f"({payload.min_value}-{payload.max_value}"
            f"{':' + payload.unit if payload.unit else ''})"
        )
    value_part = ""
    if payload.value is not None:
        unit = f":{payload.unit}" if payload.unit else ""
        value_part = f"({_format_value(payload.value)}{unit})"
    details_part = ""
    if payload.details:
        detail_text = ",".join(f"{key}={value}" for key, value in payload.details)
        details_part = f"{{{detail_text}}}"
    text_part = f"<{payload.text}>" if payload.text else ""
    return (
        f"{payload.family}:{payload.op}"
        f"{tags}{value_part}{range_part}{details_part}{text_part}"
    )


def _humanize(value: str) -> str:
    return value.replace("_", " ")


def _details_dict(payload: EffectPayload) -> dict[str, str]:
    return {key: value for key, value in payload.details}


def _format_ratio(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value * 100:g}%"
    return str(value)


def _payload_to_markdown(payload: EffectPayload) -> str:
    details = _details_dict(payload)

    if payload.family == "damage":
        if payload.op == "regular":
            return "Deals regular damage"
        if payload.op == "ohko":
            return "Knocks out target in one hit"
        if payload.op == "fixed":
            source = details.get("fixed_source")
            if source:
                return f"Deals fixed damage based on {_humanize(source)}"
            return "Deals fixed damage"
        if payload.op == "scaling":
            rule = details.get("scaling_rule")
            if rule:
                return f"Deals scaling damage ({_humanize(rule)})"
            return "Deals scaling damage"

    if payload.family == "status" and payload.op == "inflict":
        if payload.tags:
            labels = ", ".join(_humanize(tag) for tag in payload.tags)
            return f"Inflicts {labels}"
        return "Inflicts status"

    if payload.family == "stat" and payload.op == "stage_change":
        stat = _humanize(payload.tags[0]) if payload.tags else "stat"
        delta = payload.value if isinstance(payload.value, int) else 0
        if delta > 0:
            return f"Raises {stat} by {delta} stage(s)"
        if delta < 0:
            return f"Lowers {stat} by {abs(delta)} stage(s)"
        return f"Modifies {stat} stage"

    if payload.family == "hp":
        ratio = _format_ratio(payload.value)
        unit = _humanize(payload.unit or "")
        if payload.op == "heal":
            return f"Heals {ratio} of {unit}" if ratio and unit else "Heals HP"
        if payload.op == "drain":
            if ratio and unit:
                return f"Drains {ratio} of {unit} as HP"
            return "Drains HP"
        if payload.op == "recoil":
            if ratio and unit:
                return f"Takes recoil equal to {ratio} of {unit}"
            return "Takes recoil damage"
        if payload.op == "faint":
            return "Causes a Pokemon to faint"

    if payload.family == "timing":
        if payload.op == "hit_fixed" and payload.value is not None:
            return f"Hits {int(payload.value)} time(s)"
        if payload.op == "hit_range":
            if payload.min_value is not None and payload.max_value is not None:
                return f"Hits {payload.min_value}-{payload.max_value} times"
        if payload.op == "turn_pattern":
            parts: list[str] = []
            if "charge_turns" in details:
                parts.append(f"charge {details['charge_turns']} turn(s)")
            if "recharge_turns" in details:
                parts.append(f"recharge {details['recharge_turns']} turn(s)")
            if "delay_turns" in details:
                parts.append(f"delay {details['delay_turns']} turn(s)")
            if "active_turns_min" in details and "active_turns_max" in details:
                parts.append(
                    f"active {details['active_turns_min']}-{details['active_turns_max']} turns"
                )
            if parts:
                return "Timing: " + ", ".join(parts)
            return "Timing pattern"

    summary = f"{_humanize(payload.family)}: {_humanize(payload.op)}"
    extras: list[str] = []
    if payload.tags:
        extras.append("tags=" + ", ".join(_humanize(tag) for tag in payload.tags))
    if payload.value is not None:
        value = _format_value(payload.value)
        if payload.unit:
            value = f"{value} {_humanize(payload.unit)}"
        extras.append(f"value={value}")
    if payload.min_value is not None and payload.max_value is not None:
        unit = f" {_humanize(payload.unit)}" if payload.unit else ""
        extras.append(f"range={payload.min_value}-{payload.max_value}{unit}")
    if payload.details:
        detail_text = ", ".join(
            f"{_humanize(key)}={_humanize(value)}" for key, value in payload.details
        )
        extras.append(detail_text)
    if payload.text:
        extras.append(f"text={payload.text}")

    if extras:
        return f"{summary} ({'; '.join(extras)})"
    return summary


def serialize_component(component: EffectComponent) -> dict[str, object]:
    """Serialize one component to a dict for tabular outputs."""

    payload = asdict(component.payload)
    payload["payload_type"] = f"{component.payload.family}:{component.payload.op}"
    payload["tags"] = list(payload["tags"])
    payload["details"] = [list(item) for item in payload["details"]]
    return {
        "target": component.target,
        "chance": component.chance,
        "payload": payload,
    }
