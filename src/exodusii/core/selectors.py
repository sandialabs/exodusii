# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Selection helpers for Exodus variables and entities."""

from collections.abc import Iterable
from dataclasses import dataclass

from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.errors import ExodusInvalidEntityError

#: Unqualified selector names that expand to per-axis spatial columns
#: (e.g. ``COORDX``/``COORDY``/``COORDZ``).  These are entity-agnostic: they
#: may be combined with node *or* element variable selectors and are resolved
#: at node positions or element centers accordingly.
SPECIAL_SELECTOR_NAMES = frozenset({"coordinates", "displacements"})


def is_special_selector(value: "VariableSelector | str") -> bool:
    """Return true if *value* names a special spatial pseudo-variable."""

    name = value.name if isinstance(value, VariableSelector) else value
    return isinstance(name, str) and name.strip().lower() in SPECIAL_SELECTOR_NAMES


@dataclass(frozen=True, slots=True)
class VariableSelector:
    """A normalized Exodus variable selector.

    Parameters
    ----------
    name
        Variable name as stored in, or requested from, the Exodus database.
    entity
        Variable location. User-facing code may pass strings such as ``"node"``,
        ``"n"``, ``"element"``, or ``"e"``.
    original
        Optional original selector string.
    """

    name: str
    entity: Entity | str
    original: str | None = None

    def __post_init__(self) -> None:
        normalized_entity = entity(self.entity)
        if not normalized_entity.is_variable_location:
            raise ExodusInvalidEntityError(
                f"{normalized_entity.value!r} is not a valid variable location"
            )

        name = self.name.strip()
        if not name:
            raise ValueError("variable selector name cannot be empty")

        object.__setattr__(self, "entity", normalized_entity)
        object.__setattr__(self, "name", name)

    @property
    def legacy(self) -> str:
        """Return a legacy compact selector such as ``n/DISPLX``."""

        return f"{entity(self.entity).short_name}/{self.name}"

    @property
    def qualified(self) -> str:
        """Return a long-form selector such as ``node/DISPLX``."""

        return f"{entity(self.entity).value}/{self.name}"

    @property
    def is_global(self) -> bool:
        """Return true if this selects a global variable."""

        return entity(self.entity) is Entity.GLOBAL

    @property
    def is_spatial(self) -> bool:
        """Return true if this selects a non-global variable."""

        return not self.is_global


def parse_variable_selector(
    value: VariableSelector | str, *, default_entity: Entity | str | None = None
) -> VariableSelector:
    """Parse a variable selector.

    Parameters
    ----------
    value
        A :class:`VariableSelector` or string selector. String selectors may be
        qualified with an entity prefix, e.g. ``"n/DISPLX"`` or
        ``"element/ENERGY_1"``.
    default_entity
        Entity to use when ``value`` is an unqualified variable name.

    Returns
    -------
    VariableSelector
        Normalized selector.

    Raises
    ------
    ValueError
        If the selector format is invalid.
    ExodusInvalidEntityError
        If the entity prefix is invalid or not a variable location.
    """

    if isinstance(value, VariableSelector):
        return value

    if not isinstance(value, str):
        raise TypeError(
            f"variable selector must be a string or VariableSelector, got {type(value).__name__}"
        )

    text = value.strip()
    if not text:
        raise ValueError("variable selector cannot be empty")

    if "/" in text:
        entity_part, name = _split_qualified_selector(text)
        return VariableSelector(name=name, entity=entity_part, original=value)

    if text.lower() in SPECIAL_SELECTOR_NAMES:
        # Special spatial pseudo-variables are entity-agnostic; bind them to a
        # placeholder NODE entity so they parse, and let the query layer resolve
        # them at the appropriate positions (nodes or element centers).
        return VariableSelector(name=text, entity=Entity.NODE, original=value)

    if default_entity is None:
        raise ValueError(f"unqualified variable selector {value!r} requires a default_entity")

    return VariableSelector(name=text, entity=default_entity, original=value)


def parse_variable_selectors(
    values: Iterable[VariableSelector | str],
    *,
    default_entity: Entity | str | None = None,
    require_same_entity: bool = False,
) -> tuple[VariableSelector, ...]:
    """Parse multiple variable selectors.

    Parameters
    ----------
    values
        Iterable of selectors.
    default_entity
        Entity to use for unqualified selector strings.
    require_same_entity
        If true, all parsed selectors must have the same entity.

    Returns
    -------
    tuple[VariableSelector, ...]
        Parsed selectors.
    """

    selectors = tuple(
        parse_variable_selector(value, default_entity=default_entity) for value in values
    )

    if require_same_entity and selectors:
        # Special spatial pseudo-variables (coordinates/displacements) are
        # entity-agnostic and excluded from the same-entity requirement.
        real = [selector for selector in selectors if not is_special_selector(selector)]
        if real:
            first = entity(real[0].entity)
            different = [selector for selector in real if entity(selector.entity) is not first]
            if different:
                entities = ", ".join(sorted({entity(selector.entity).value for selector in real}))
                raise ValueError(f"variable selectors must have the same entity; got {entities}")

    return selectors


def _split_qualified_selector(value: str) -> tuple[str, str]:
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError(f"invalid variable selector {value!r}; expected format ENTITY/NAME")

    entity_part, name = (part.strip() for part in parts)
    if not entity_part:
        raise ValueError(f"invalid variable selector {value!r}; entity is empty")
    if not name:
        raise ValueError(f"invalid variable selector {value!r}; name is empty")

    return entity_part, name


__all__ = [
    "SPECIAL_SELECTOR_NAMES",
    "VariableSelector",
    "is_special_selector",
    "parse_variable_selector",
    "parse_variable_selectors",
]
