# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    Unpack,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import FieldInfo, _FromFieldInfoInputs
from pydantic_core import PydanticUndefined

T = TypeVar("T", bound=BaseModel)
V = TypeVar("V", bound=BaseModel)

DEFAULT_VALUE_TYPE: TypeAlias = Literal["__DEFAULT_VALUE__"]
DEFAULT_VALUE_PLACEHOLDER: DEFAULT_VALUE_TYPE = "__DEFAULT_VALUE__"


@dataclass
class DefaultValueContainer:
    """Container to hold default value information for a model field.

    Attributes:
        value: The static default value
        factory: A callable that generates the default value
    """

    value: Any | None
    factory: Callable | None

    def get_default(self) -> Any:
        """Retrieve the default value.

        Returns:
            The default value, either static or generated by the factory
        """
        if self.value is not None:
            return self.value
        if self.factory is not None:
            return self.factory()
        return None

    def __hash__(self) -> int:
        return hash((self.value, self.factory))


def is_pydantic_model(model: Any) -> TypeGuard[type[BaseModel]]:
    """Check if the provided model is a subclass of Pydantic's BaseModel.

    Args:
        model: The model to check.

    Returns:
        TypeGuard[type[BaseModel]]: True if it's a Pydantic BaseModel subclass, False otherwise.
    """
    return (
        model is not None
        and not get_origin(model)
        and isinstance(model, type)
        and issubclass(model, BaseModel)
    )


def copy_field_info(
    field_info: FieldInfo,
    default: Any = PydanticUndefined,
    **overrides: Any,
) -> FieldInfo:
    """Copy a FieldInfo instance, optionally overriding attributes.

    Args:
        field_info: The FieldInfo instance to copy
        default: The default value to use for the field
        **overrides: Keyword arguments to override attributes

    Returns:
        A new FieldInfo instance with the specified overrides
    """
    field_kwargs: dict[str, Any] = {}
    for attr_name in _FromFieldInfoInputs.__annotations__.keys():
        if attr_value := getattr(field_info, attr_name, None):
            field_kwargs[attr_name] = attr_value

    field_kwargs.pop("annotation")
    field_kwargs.update(overrides)

    return field_info.from_field(default=default, **field_kwargs)


def _unwrap_pydantic_type(model_type: type[Any] | None) -> Any:
    """Recursively unwrap Pydantic model types to ensure compatibility.

    Args:
        model_type: The type to unwrap

    Returns:
        The unwrapped type
    """
    if model_type is None:
        return type(None)

    origin = get_origin(model_type)
    args = get_args(model_type)

    if origin is list:
        return list[_unwrap_pydantic_type(args[0])]  # type: ignore

    if origin is dict:
        return dict[_unwrap_pydantic_type(args[0]), _unwrap_pydantic_type(args[1])]  # type: ignore

    if origin in (Union, UnionType):
        return Union[tuple(_unwrap_pydantic_type(arg) for arg in args)]  # noqa: UP007

    result: Any = None
    if is_pydantic_model(model_type):
        result = remove_default_values(model_type)
    else:
        result = copy(model_type)

    return result


def _extract_annotations(model_type: type[Any]) -> tuple[type[Any], Sequence[Any]]:
    """Extract the base type and annotations from an Annotated type.

    Args:
        model_type: The type to extract from.

    Returns:
        The base type and its annotations.
    """
    origin = get_origin(model_type)
    if origin is Annotated:
        args = get_args(model_type)
        return args[0], args[1:]
    return model_type, ()


def _apply_annotations(base_type: Any, *annotations: Any) -> Any:
    """Apply annotations to a base type using Annotated.

    Args:
        base_type: The base type to annotate.
        *annotations: Annotations to apply.

    Returns:
        The annotated type.
    """
    if annotations:
        return Annotated[base_type, *annotations]
    return base_type


def _replace_placeholder_with_default(instance: Any, field_name: str, field: FieldInfo) -> bool:
    """Replace the placeholder with the default value if present.

    Args:
        instance: The model instance.
        field_name: The name of the field.
        field: The field information.

    Returns:
        True if the placeholder was replaced, False otherwise.
    """
    if getattr(instance, field_name) != DEFAULT_VALUE_PLACEHOLDER:
        return False

    default_container = next(
        (metadata for metadata in field.metadata if isinstance(metadata, DefaultValueContainer)),
        None,
    )

    if default_container:
        setattr(instance, field_name, default_container.get_default())
        return True

    return False


def _restore_nested_models(field_type: Any, field_value: Any) -> Any:  # noqa: PLR0911
    """Restore default values for nested models, including handling Union types.

    Args:
        field_type: The type of the field.
        field_value: The current value of the field.

    Returns:
        The restored value
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    if isinstance(field_value, BaseModel):
        return restore_default_values(field_value, field_value.__class__)

    if origin is list and isinstance(field_value, list):
        item_type = args[0] if args else Any
        return [_restore_nested_models(item_type, item) for item in field_value]

    if origin is dict and isinstance(field_value, dict):
        value_type = args[1] if len(args) > 1 else Any
        return {
            key: _restore_nested_models(value_type, value) for key, value in field_value.items()
        }

    if origin in (Union, UnionType):
        for possible_type in args:
            try:
                if is_pydantic_model(possible_type):
                    model_instance = possible_type.model_validate(field_value)
                    return restore_default_values(model_instance, possible_type)
                else:
                    return field_value
            except (ValidationError, ValueError, TypeError):
                continue
        # If none of the types match, return the value as is
        return field_value

    else:
        # For other types, return the value as is
        return field_value


def remove_default_values(model: type[BaseModel]) -> type[BaseModel]:
    """Transform a Pydantic model by removing default values.

    Creates a new model where fields with default values are made required and
    use a placeholder value to indicate missing data. This is useful when working
    with systems that don't support default values in structural outputs.

    Args:
        model: The original Pydantic model

    Returns:
        A new Pydantic model with defaults removed
    """
    transformed_fields: dict[str, Any] = {}

    for field_name, field in model.model_fields.items():
        transformed_type = _unwrap_pydantic_type(field.annotation)

        if not field.is_required():
            # Remove default values
            updated_field = copy_field_info(
                field,
                metadata=[*copy(field.metadata)],
            )

            base_type, annotations = _extract_annotations(transformed_type)
            transformed_type = _apply_annotations(
                base_type | DEFAULT_VALUE_TYPE,
                *annotations,
                DefaultValueContainer(
                    value=field.default,
                    factory=field.default_factory,
                ),
            )
        else:
            # Preserve required fields
            updated_field = copy_field_info(field)

        transformed_fields[field_name] = (transformed_type, updated_field)

    try:
        transformed_model = create_model(
            f"{model.__name__}Transformed",
            __base__=model,
            **transformed_fields,
        )
    except TypeError as e:
        raise TypeError(
            f"Error creating transformed model '{model.__name__}Transformed': {e}"
        ) from e

    # Rebuild the original model to ensure consistency
    model.model_rebuild()

    return transformed_model


def restore_default_values(instance: T, target_model: type[V]) -> V:
    """Restore default values in a transformed model instance.

    Maps an instance of a transformed model (with placeholders) back to the
    original model by replacing placeholders with actual default values.

    Args:
        instance: The transformed model instance
        target_model: The original target Pydantic model

    Returns:
        An instance of the original target model with defaults restored
    """
    union_values: dict[str, Any] = {}

    for field_name, field in instance.model_fields.items():
        replaced = _replace_placeholder_with_default(instance, field_name, field)

        if not replaced:
            if field.annotation is not None:
                field_value = getattr(instance, field_name)
                restored_value = _restore_nested_models(field.annotation, field_value)
                setattr(instance, field_name, restored_value)
            else:
                raise ValueError(
                    f"Error restoring default values for model '{target_model.__name__}': "
                    f"field '{field_name}' has no annotation"
                )

    # Dump the instance to a dict without warnings because we've already handled
    # the placeholders and we don't want to see warnings about them
    dumped_data = instance.model_dump(warnings=False)

    # Correct union fields to use the actual member values
    for field_name, value in union_values.items():
        if isinstance(value, BaseModel):
            dumped_data[field_name] = value.model_dump()
        else:
            dumped_data[field_name] = value

    try:
        return target_model.model_validate(dumped_data)
    except Exception as e:
        raise ValueError(
            f"Error restoring default values for model '{target_model.__name__}': {e}"
        ) from e


def change_field_type(  # noqa: PLR0913
    model_type: type[T],
    field_name: str,
    new_type: Any,
    new_model_type: type[T] | None = None,
    new_model_name: str | None = None,
    default: Any = PydanticUndefined,
    **kwargs: Unpack[_FromFieldInfoInputs],
) -> type[T]:
    """Create a new Pydantic model with a modified or added field.

    Creates a copy of the original model with one field modified or added,
    preserving all other fields and model configuration. The new model can:
    - Modify existing field types
    - Add new fields
    - Change base model type
    - Customize field validation

    Args:
        model_type: The original Pydantic model to modify
        field_name: Name of the field to modify or add
        new_type: New type for the field
        new_model_type: Optional new base model type (defaults to original model)
        new_model_name: Optional name for the new model (defaults to "Updated" + original name)
        default: Optional default value for the field
        **kwargs: Additional field configuration (validation rules, descriptions, etc.)

    Returns:
        A new Pydantic model class with the modified or added field

    Raises:
        TypeError: If the default value doesn't match the new field type
        ValidationError: If default value validation fails when validate_default=True

    Example:
    ```python
    class User(BaseModel):
        id: int
        name: str

    # Modify existing field
    UserStr = change_field_type(
        model_type=User,
        field_name="id",
        new_type=str,
        new_model_name="UserStr"
    )

    # Add new field
    UserWithAge = change_field_type(
        model_type=User,
        field_name="age",
        new_type=int,
        default=0,
        ge=0,
        description="User's age in years"
    )

    # Change base model and add validation
    class ValidatedModel(BaseModel):
        model_config = ConfigDict(validate_default=True)

    UserValidated = change_field_type(
        model_type=User,
        field_name="name",
        new_type=str,
        new_model_type=ValidatedModel,
        min_length=1,
        max_length=100
    )
    ```
    """
    fields: dict[str, Any] = {}
    if field := model_type.model_fields.get(field_name):
        field_info = copy_field_info(field, default=default, **kwargs)
        fields[field_name] = (new_type, field_info)
    else:
        fields[field_name] = (new_type, FieldInfo(default=default, **kwargs))

    for name, field_info in model_type.model_fields.items():
        if name != field_name:
            fields[name] = (field_info.annotation, field_info)

    if default is not PydanticUndefined and kwargs.get("validate_default"):
        try:
            fields = {field_name: (new_type, default)}
            temp_model = create_model("TempModel", **fields)
            temp_model(**{field_name: default})
        except ValidationError as e:
            raise TypeError(
                f"Default value {default!r} is not valid for field '{field_name}' of type {new_type}"
            ) from e

    updated_model_name = new_model_name or f"Updated{model_type.__name__}"

    new_model = create_model(
        updated_model_name,
        __base__=new_model_type or model_type,
        **fields,
    )

    return new_model
