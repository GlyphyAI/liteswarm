import re

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
