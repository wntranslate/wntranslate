from enum import Enum, IntEnum


class ContextType(IntEnum):
    NONE = 0
    ROLLING_SUMMARY = 1


class Creativity(IntEnum):
    VERY_LOW = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


class NameType(Enum):
    PERSON = "person"
    ITEM = "item"
    ORGANIZATION = "organization"
    CREATIVE_WORK = "creative_work"
    OTHER = "other"


class NodeType(Enum):
    TEXT = "text"
    TERM = "term"
    FOOTNOTE = "footnote"


class ChapterKind(Enum):
    CHAPTER = "chapter"
    ANNOUNCEMENT = "announcement"
    POST = "post"
    OTHER = "other"


class RequestKind(Enum):
    SYNOPSIS = "synopsis"
    ORIGINAL_TITLE = "original_title"
    ORIGINAL_SYNOPSIS = "original_synopsis"
    CHAPTER_LIST = "chapter_list"
    CHAPTER_SUMMARY = "chapter_summary"
    CHAPTER_TEXT = "chapter_text"
    CHAPTER_ORIGINAL_TITLE = "chapter_original_title"
    CHAPTER_ORIGINAL_CONTENT = "chapter_original_content"
    GLOSSARY_ENTRY = "glossary_entry"
    STORY_SUMMARY = "story_summary"
    OTHER = "other"


class ResponseType(Enum):
    REQUEST = "request"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"


class GlossaryOp(Enum):
    NEW = "new"
    EDIT = "edit"
    DELETE = "delete"
