"""The fixed list of signs the extractor is allowed to return."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Sign:
    sign_id: str
    description: str


# Order is the order findings are returned in. Descriptions are the brief, not a wider clinical list.
SIGNS: tuple[Sign, ...] = (
    Sign("reduced_appetite", "eating less, or has stopped eating or grazing"),
    Sign("weakness", "dull, weak, lying down more than usual"),
    Sign(
        "swelling",
        "a lump below the ear, under the jaw, or in front of the shoulder "
        "(farmers may call it mbandabvu)",
    ),
    Sign("rapid_breathing", "breathing fast while resting"),
    Sign("cloudy_eye", "the eye looks white, grey or milky"),
    Sign("froth_nose_mouth", "foam or froth from the nose or mouth"),
    Sign("red_urine", "urine that is red, brown, or the colour of strong tea"),
    Sign("unable_to_stand", "cannot get up or stay standing"),
    Sign("ticks", "ticks can be seen on the animal"),
)

SIGN_IDS: frozenset[str] = frozenset(sign.sign_id for sign in SIGNS)

# Policy from the brief, applied in code. Other signs, however serious they sound, do not escalate.
ESCALATION_SIGN_IDS: frozenset[str] = frozenset(
    {"froth_nose_mouth", "unable_to_stand", "cloudy_eye"}
)

VALUES: frozenset[str] = frozenset({"yes", "no", "not_sure"})

# When the model emits the same sign twice, keep the more serious value.
VALUE_RANK: dict[str, int] = {"no": 0, "not_sure": 1, "yes": 2}
