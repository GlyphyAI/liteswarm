# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import re

import orjson
from json_repair import repair_json

from liteswarm.types.misc import JSON
from liteswarm.utils.misc import find_tag_content

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


def dump_json(data: JSON, indent: bool = False) -> str:
    option = orjson.OPT_INDENT_2 if indent else None
    return orjson.dumps(data, option=option).decode()


def load_json(data: str) -> JSON:
    return orjson.loads(data)


def find_json_tag(text: str, tag: str) -> JSON:
    tag_content = find_tag_content(text, tag)
    if not tag_content:
        return None

    return repair_json(tag_content, return_objects=True)  # type: ignore
