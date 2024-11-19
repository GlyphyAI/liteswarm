import re

import orjson
from partial_json_parser.core.api import JSON, parse_json

from .types import CodeBlock


def extract_code_block(content: str) -> CodeBlock:
    r"""Extract the content from a code block.

    Args:
        content: The content to extract the code block from.

    Returns:
        CodeBlock containing the content and optional language.

    Example:
        >>> extract_code_block("```json\\n{\"key\": \"value\"}\\n```")
        CodeBlock(content='{\"key\": \"value\"}', language='json')
    """
    pattern = r"```(\w+)?\n?(.*?)```"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return CodeBlock(content=content.strip())

    language = match.group(1)
    content = match.group(2).strip()

    return CodeBlock(
        content=content,
        language=language,
    )


def load_partial_json(content: str) -> JSON:
    return parse_json(content, parser=orjson.loads)


def dump_partial_json(data: JSON) -> str:
    return orjson.dumps(data).decode()


def dump_json(data: JSON) -> str:
    return orjson.dumps(data).decode()


def load_json(data: str) -> JSON:
    return orjson.loads(data)
