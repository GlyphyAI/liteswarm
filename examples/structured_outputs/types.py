# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Literal

from pydantic import BaseModel, Field


class Task(BaseModel):
    task_type: str
    title: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodingTask(Task):
    task_type: Literal["coding"]
    filepath: str
    code: str
    assigned_to: str | None = None
    state: Literal["pending", "in_progress", "completed", "failed"] = "pending"


class TestingTask(Task):
    task_type: Literal["testing"]
    test_report: str
    failed_reason: str | None = None


class Plan(BaseModel):
    tasks: list[CodingTask | TestingTask]


class Response(BaseModel):
    inner_monologue: str
    response: Plan
