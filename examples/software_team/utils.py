import re

from .types import CodeBlock


def extract_code_block(content: str) -> CodeBlock:
    r"""Extract the content from a code block, removing language specification if present.

    Args:
        content: The content to extract the code block from.

    Returns:
        The content from the code block (without language spec) or the original content
        if no code block is found.

    Example:
        >>> extract_code_block("```json\\n{\"key\": \"value\"}\\n```")
        '{\"key\": \"value\"}'
        >>> extract_code_block("```{\"key\": \"value\"}```")
        '{\"key\": \"value\"}'
    """
    language: str | None = None
    filepath: str | None = None
    match = re.search(r"```(?:\w+\n|\n)?(.*)```", content, re.DOTALL)

    if match:
        annotations = match.group(0).split(":")
        language = annotations[0] if len(annotations) > 0 else None
        filepath = annotations[1] if len(annotations) > 1 else None
        content = match.group(1).strip()

    return CodeBlock(
        content=content,
        language=language,
        filepath=filepath,
    )
