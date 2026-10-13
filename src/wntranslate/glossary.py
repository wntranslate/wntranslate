from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from ._ids import new_id
from .enums import NameType


@dataclass
class Alias:
    """An alternate form of a term and the output it maps to.

    Args:
        original: Source-language form, possibly partial such as a cut-off name.
        translation: Output form to render.
        context: Optional note describing when this alias applies.
    """

    original: str
    translation: str = ""
    context: str | None = None
    _id: str = field(default_factory=new_id, init=False, repr=False)

    @property
    def id(self) -> str:
        """Stable unique identifier, generated once and never settable."""
        return self._id

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"id": self._id, "original": self.original, "translation": self.translation, "context": self.context}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Alias:
        """Rebuild an alias, preserving its id.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed alias.
        """
        alias = cls(original=data["original"], translation=data.get("translation", ""), context=data.get("context"))
        if "id" in data:
            alias._id = data["id"]
        return alias


@dataclass
class GlossaryEntry:
    """A glossary term with its original name, aliases, and definition.

    Args:
        name: Output name of the term.
        original_name: Source-language name of the term.
        definition: Explanation of the term.
        aliases: Alternate forms of the term.
    """

    name: str
    original_name: str = ""
    definition: str = ""
    aliases: list[Alias] = field(default_factory=list)
    _id: str = field(default_factory=new_id, init=False, repr=False)

    type: ClassVar[NameType] = NameType.OTHER

    @property
    def id(self) -> str:
        """Stable unique identifier, generated once and never settable."""
        return self._id

    def get_alias(self, alias_id: str) -> Alias | None:
        """Find an alias by its id.

        Args:
            alias_id: The alias id.

        Returns:
            The matching alias, or None.
        """
        for alias in self.aliases:
            if alias.id == alias_id:
                return alias
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict including id and name type.

        Returns:
            A JSON-compatible representation.
        """
        data: dict[str, Any] = {"id": self._id, "type": self.type.value}
        for f in fields(self):
            if f.name == "_id":
                continue
            if f.name == "aliases":
                data["aliases"] = [alias.to_dict() for alias in self.aliases]
            else:
                data[f.name] = getattr(self, f.name)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlossaryEntry:
        """Rebuild the correct entry subclass, preserving its id.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed entry.
        """
        target = _ENTRY_TYPES[NameType(data["type"])]
        valid = {f.name for f in fields(target) if f.init}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in valid}
        if "aliases" in data:
            kwargs["aliases"] = [Alias.from_dict(alias) for alias in data["aliases"]]
        entry = target(**kwargs)
        if "id" in data:
            entry._id = data["id"]
        return entry


@dataclass
class PersonEntry(GlossaryEntry):
    """A glossary entry for a person.

    Args:
        pronouns: Pronouns used for the person.
    """

    pronouns: str = ""

    type: ClassVar[NameType] = NameType.PERSON


@dataclass
class ItemEntry(GlossaryEntry):
    """A glossary entry for an item or object."""

    type: ClassVar[NameType] = NameType.ITEM


@dataclass
class OrganizationEntry(GlossaryEntry):
    """A glossary entry for an organization."""

    type: ClassVar[NameType] = NameType.ORGANIZATION


@dataclass
class CreativeWorkEntry(GlossaryEntry):
    """A glossary entry for a creative work such as a book or movie.

    Args:
        medium: The kind of work, e.g. book or movie.
    """

    medium: str = ""

    type: ClassVar[NameType] = NameType.CREATIVE_WORK


@dataclass
class OtherEntry(GlossaryEntry):
    """A glossary entry whose type is specified by the AI.

    Args:
        specified_type: The free-form type assigned to the term.
    """

    specified_type: str = ""

    type: ClassVar[NameType] = NameType.OTHER


_ENTRY_TYPES: dict[NameType, type[GlossaryEntry]] = {
    NameType.PERSON: PersonEntry,
    NameType.ITEM: ItemEntry,
    NameType.ORGANIZATION: OrganizationEntry,
    NameType.CREATIVE_WORK: CreativeWorkEntry,
    NameType.OTHER: OtherEntry,
}


@dataclass
class Glossary:
    """A collection of glossary entries.

    Args:
        entries: The stored entries.
    """

    entries: list[GlossaryEntry] = field(default_factory=list)

    def add(self, entry: GlossaryEntry) -> None:
        """Append an entry to the glossary.

        Args:
            entry: The entry to add.
        """
        self.entries.append(entry)

    def get(self, entry_id: str) -> GlossaryEntry | None:
        """Find an entry by its id.

        Args:
            entry_id: The entry id.

        Returns:
            The matching entry, or None.
        """
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def remove(self, entry_id: str) -> bool:
        """Delete the entry with the given id.

        Args:
            entry_id: The entry id.

        Returns:
            Whether an entry was removed.
        """
        for index, entry in enumerate(self.entries):
            if entry.id == entry_id:
                del self.entries[index]
                return True
        return False

    def replace(self, entry_id: str, entry: GlossaryEntry) -> bool:
        """Replace an entry, keeping its id and matching alias ids by original form.

        Args:
            entry_id: The id of the entry to replace.
            entry: The replacement entry.

        Returns:
            Whether an entry was replaced.
        """
        for index, existing in enumerate(self.entries):
            if existing.id == entry_id:
                entry._id = entry_id
                kept = {alias.original: alias.id for alias in existing.aliases}
                for alias in entry.aliases:
                    if alias.original in kept:
                        alias._id = kept[alias.original]
                self.entries[index] = entry
                return True
        return False

    def lookup(self, name: str) -> GlossaryEntry | None:
        """Find an entry by name, original name, or alias, case-insensitively.

        Args:
            name: The text to search for.

        Returns:
            The matching entry, or None.
        """
        key = name.casefold()
        for entry in self.entries:
            candidates = [entry.name, entry.original_name]
            for alias in entry.aliases:
                candidates.append(alias.original)
                candidates.append(alias.translation)
            if any(key == candidate.casefold() for candidate in candidates if candidate):
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Glossary:
        """Rebuild a glossary from its dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed glossary.
        """
        return cls(entries=[GlossaryEntry.from_dict(entry) for entry in data.get("entries", [])])

    def __iter__(self) -> Iterator[GlossaryEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)
