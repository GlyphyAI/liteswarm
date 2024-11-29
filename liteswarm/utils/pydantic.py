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
    TypeVar,
    Union,
    Unpack,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Discriminator, ValidationError, create_model
from pydantic.fields import FieldInfo, _FromFieldInfoInputs
from pydantic_core import PydanticUndefined

from liteswarm.utils.typing import is_subtype, union

T = TypeVar("T", bound=BaseModel)
V = TypeVar("V", bound=BaseModel)

PLACEHOLDER_TYPE: TypeAlias = Literal["___UNKNOWN___"]
PLACEHOLDER_VALUE: PLACEHOLDER_TYPE = "___UNKNOWN___"


@dataclass
class DefaultValueContainer:
    """Container for field default value information.

    Stores either a static default value or a factory function
    that generates default values on demand.

    Attributes:
        value: Static default value.
        factory: Function that generates default values.

    Examples:
        Static value:
            ```python
            container = DefaultValueContainer(
                value=42,
                factory=None
            )
            assert container.get_default() == 42
            ```

        Factory function:
            ```python
            container = DefaultValueContainer(
                value=None,
                factory=lambda: list()
            )
            result = container.get_default()  # New list
            ```
    """

    value: Any | None
    factory: Callable[..., Any] | None

    def get_default(self) -> Any:
        """Get the default value.

        Returns static value if set, otherwise calls factory
        function if available. Returns None if neither is set.

        Returns:
            Default value from static value or factory.

        Examples:
            Static value:
                ```python
                container = DefaultValueContainer(42, None)
                assert container.get_default() == 42
                ```

            Factory:
                ```python
                import random
                container = DefaultValueContainer(
                    None,
                    lambda: random.randint(1, 10)
                )
                value = container.get_default()  # Random int
                ```
        """
        if self.value not in (None, PydanticUndefined):
            return self.value

        if self.factory is not None:
            return self.factory()

        return None

    def __hash__(self) -> int:
        return hash((self.value, self.factory))


def copy_field_info(
    field_info: FieldInfo,
    exclude_attributes: Sequence[str] = (),
    make_required: bool = False,
    **overrides: Any,
) -> FieldInfo:
    """Create a copy of a Pydantic field with modifications.

    Creates a new FieldInfo instance with optional changes:
    - Exclude specific attributes
    - Make field required
    - Override field properties

    Args:
        field_info: Original field to copy.
        exclude_attributes: Attributes to omit from copy.
        make_required: Whether to remove default value.
        **overrides: New values for field attributes.

    Returns:
        New FieldInfo instance with specified modifications.

    Examples:
        Basic copy:
            ```python
            class Model(BaseModel):
                field: int = Field(
                    default=0,
                    description="A number"
                )

            # Copy with new description
            new_field = copy_field_info(
                Model.model_fields["field"],
                description="Updated description"
            )
            ```

        Make required:
            ```python
            # Remove default value
            required_field = copy_field_info(
                optional_field,
                make_required=True
            )
            assert required_field.is_required()
            ```

        Exclude attributes:
            ```python
            # Copy without validation rules
            basic_field = copy_field_info(
                field_info,
                exclude_attributes=["gt", "lt"]
            )
            ```
    """
    field_kwargs: dict[str, Any] = {}
    for attr_name in _FromFieldInfoInputs.__annotations__.keys():
        if attr_name in exclude_attributes:
            continue

        attr_value = getattr(field_info, attr_name, None)
        if attr_value is not None:
            field_kwargs[attr_name] = attr_value

    field_kwargs.pop("annotation", None)
    field_kwargs.update(overrides)

    if make_required:
        field_kwargs.pop("default", None)
        field_kwargs.pop("default_factory", None)

    return field_info.from_field(**field_kwargs)


def _unwrap_pydantic_type(model_type: type[Any] | None) -> type[Any]:  # noqa: PLR0911, PLR0912
    """Recursively unwrap complex type annotations.

    Processes type annotations to handle:
    - Annotated types with metadata
    - Container types (list, dict)
    - Union types with deduplication
    - Pydantic models with default removal

    Args:
        model_type: Type annotation to unwrap.

    Returns:
        Unwrapped and processed type annotation.

    Examples:
        Basic types:
            ```python
            assert _unwrap_pydantic_type(str) == str
            assert _unwrap_pydantic_type(None) == type(None)
            ```

        Annotated types:
            ```python
            type_info = _unwrap_pydantic_type(
                Annotated[str, Field(min_length=1)]
            )
            assert type_info == str
            ```

        Container types:
            ```python
            assert _unwrap_pydantic_type(
                list[str]
            ) == list[str]

            assert _unwrap_pydantic_type(
                dict[str, int]
            ) == dict[str, int]
            ```

        Union types:
            ```python
            # Removes duplicates
            type_info = _unwrap_pydantic_type(
                Union[str, str, int]
            )
            assert type_info == Union[str, int]
            ```
    """
    if model_type is None:
        return type(None)

    origin = get_origin(model_type)
    args = get_args(model_type)

    if origin is Annotated:
        base_type, *annotations = args
        filtered_annotations = [
            annotation
            for annotation in annotations
            if not isinstance(annotation, FieldInfo | Discriminator)
        ]

        unwrapped_base = _unwrap_pydantic_type(base_type)
        if filtered_annotations:
            return Annotated[unwrapped_base, *filtered_annotations]  # type: ignore

        return unwrapped_base

    if origin is list:
        return list[_unwrap_pydantic_type(args[0])]  # type: ignore

    if origin is dict:
        return dict[  # type: ignore
            _unwrap_pydantic_type(args[0]),
            _unwrap_pydantic_type(args[1]),
        ]

    if origin in (Union, UnionType):
        # Flatten nested Unions
        flat_args: list[Any] = []
        for arg in args:
            processed_arg = _unwrap_pydantic_type(arg)
            if get_origin(processed_arg) in (Union, UnionType):
                flat_args.extend(get_args(processed_arg))
            else:
                flat_args.append(processed_arg)

        # Remove duplicates while preserving order
        unique_args: list[Any] = []
        seen: set[Any] = set()
        for arg in flat_args:
            if arg not in seen:
                unique_args.append(arg)
                seen.add(arg)

        return union(unique_args)

    if is_subtype(model_type, BaseModel):
        return remove_default_values(model_type)
    else:
        return copy(model_type)


def _replace_placeholder_with_default(
    instance: BaseModel,
    placeholder: Any,
    field_name: str,
    field_info: FieldInfo,
) -> bool:
    """Replace placeholder value with field's default.

    Checks if a field's value matches the placeholder and
    replaces it with the field's default value if available.

    Args:
        instance: Model instance to modify.
        placeholder: Placeholder value to check for.
        field_name: Name of field to process.
        field_info: Field metadata with default info.

    Returns:
        True if placeholder was replaced, False otherwise.

    Examples:
        Basic replacement:
            ```python
            class Model(BaseModel):
                value: int = Field(default=42)

            instance = Model(value="___UNKNOWN___")
            replaced = _replace_placeholder_with_default(
                instance,
                "___UNKNOWN___",
                "value",
                Model.model_fields["value"]
            )
            assert replaced is True
            assert instance.value == 42
            ```

        No replacement needed:
            ```python
            instance = Model(value=100)
            replaced = _replace_placeholder_with_default(
                instance,
                "___UNKNOWN___",
                "value",
                Model.model_fields["value"]
            )
            assert replaced is False
            assert instance.value == 100
            ```
    """
    if getattr(instance, field_name) != placeholder:
        return False

    default_container = next(
        (
            metadata
            for metadata in field_info.metadata
            if isinstance(metadata, DefaultValueContainer)
        ),
        None,
    )

    if default_container:
        setattr(instance, field_name, default_container.get_default())
        return True

    return False


def _restore_nested_models(field_type: Any, field_value: Any) -> Any:  # noqa: PLR0911
    """Recursively restore defaults in nested structures.

    Processes field values to restore defaults in:
    - Nested Pydantic models
    - Lists and dictionaries
    - Union type fields
    - Annotated types

    Args:
        field_type: Type annotation of the field.
        field_value: Current value to process.

    Returns:
        Processed value with defaults restored.

    Examples:
        Nested model:
            ```python
            class Inner(BaseModel):
                value: int = 0

            class Outer(BaseModel):
                inner: Inner = Inner()

            value = _restore_nested_models(
                Inner,
                {"value": "___UNKNOWN___"}
            )
            assert value.value == 0
            ```

        Collections:
            ```python
            # List of models
            value = _restore_nested_models(
                list[Inner],
                [{"value": "___UNKNOWN___"}]
            )
            assert value[0].value == 0

            # Dict with model values
            value = _restore_nested_models(
                dict[str, Inner],
                {"key": {"value": "___UNKNOWN___"}}
            )
            assert value["key"].value == 0
            ```
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is Annotated:
        base_type, *_annotations = args
        return _restore_nested_models(base_type, field_value)

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
                if is_subtype(possible_type, BaseModel):
                    model_instance = possible_type.model_validate(field_value)
                    return restore_default_values(model_instance, possible_type)
                else:
                    return field_value
            except (ValidationError, ValueError, TypeError):
                continue

        return field_value

    if isinstance(field_value, BaseModel):
        return restore_default_values(field_value, field_value.__class__)

    return field_value


def remove_default_values(model: type[BaseModel]) -> type[BaseModel]:
    """Create model variant with required fields.

    Transforms a Pydantic model by making all fields required
    and replacing default values with placeholders. Useful for
    systems that don't support optional fields.

    Args:
        model: Original Pydantic model class.

    Returns:
        New model class with all fields required.

    Raises:
        TypeError: If model transformation fails.

    Examples:
        Basic model:
            ```python
            class User(BaseModel):
                name: str
                age: int = 0
                tags: list[str] = []

            RequiredUser = remove_default_values(User)
            # All fields now required, defaults preserved
            # internally but not exposed in schema
            ```

        Nested models:
            ```python
            class Address(BaseModel):
                street: str = ""
                city: str

            class Contact(BaseModel):
                name: str
                address: Address = Address()

            RequiredContact = remove_default_values(Contact)
            # Both top-level and nested defaults handled
            ```
    """
    transformed_fields: dict[str, Any] = {}
    for field_name, field in model.model_fields.items():
        transformed_type: Any = _unwrap_pydantic_type(field.annotation)

        if not field.is_required():
            transformed_type = union([transformed_type, PLACEHOLDER_TYPE])

            updated_field = copy_field_info(
                field,
                exclude_attributes=("discriminator"),
                make_required=True,
            )

            updated_field.metadata = [
                *copy(field.metadata),
                DefaultValueContainer(value=field.default, factory=field.default_factory),
            ]

        else:
            updated_field = copy_field_info(
                field,
                exclude_attributes=("discriminator"),
            )

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


def restore_default_values(instance: T, target_model_type: type[V]) -> V:
    """Convert transformed model back to original form.

    Restores default values in a model instance that was previously
    transformed by remove_default_values(). Handles nested models
    and collections recursively.

    Args:
        instance: Transformed model instance.
        target_model_type: Original model class.

    Returns:
        Instance of original model with defaults restored.

    Raises:
        ValueError: If default value restoration fails.

    Examples:
        Basic restoration:
            ```python
            class User(BaseModel):
                name: str
                age: int = 0

            RequiredUser = remove_default_values(User)
            instance = RequiredUser(
                name="Alice",
                age="___UNKNOWN___"
            )

            original = restore_default_values(
                instance,
                User
            )
            assert original.age == 0
            ```

        Nested models:
            ```python
            class Settings(BaseModel):
                theme: str = "light"

            class App(BaseModel):
                settings: Settings = Settings()

            RequiredApp = remove_default_values(App)
            instance = RequiredApp(
                settings="___UNKNOWN___"
            )

            original = restore_default_values(
                instance,
                App
            )
            assert original.settings.theme == "light"
            ```
    """
    union_values: dict[str, Any] = {}
    for field_name, field in instance.model_fields.items():
        replaced = _replace_placeholder_with_default(
            instance=instance,
            placeholder=PLACEHOLDER_VALUE,
            field_name=field_name,
            field_info=field,
        )

        if not replaced:
            if field.annotation is not None:
                field_value = getattr(instance, field_name)
                restored_value = _restore_nested_models(field.annotation, field_value)
                setattr(instance, field_name, restored_value)
            else:
                raise ValueError(
                    f"Error restoring default values for model '{target_model_type.__name__}': "
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
        return target_model_type.model_validate(dumped_data)
    except Exception as e:
        raise ValueError(
            f"Error restoring default values for model '{target_model_type.__name__}': {e}"
        ) from e


def replace_default_values(
    instance: BaseModel,
    target_model_type: type[BaseModel] | None = None,
) -> BaseModel:
    """Replace field values matching defaults with placeholders.

    Processes a model instance to replace values that match their
    field defaults with placeholders. If target_model_type is not
    provided, creates a transformed version with removed defaults.

    Args:
        instance: Model instance to process.
        target_model_type: Optional transformed model class. If None,
            creates one using remove_default_values().

    Returns:
        New instance with defaults replaced by placeholders.

    Raises:
        TypeError: If resulting model is invalid.

    Examples:
        Basic replacement:
            ```python
            class User(BaseModel):
                name: str
                age: int = 0
                active: bool = True

            user = User(name="Alice", age=0)

            # Using automatic transformation
            processed = replace_default_values(user)
            assert processed.age == "___UNKNOWN___"
            assert processed.active == "___UNKNOWN___"

            # Using pre-transformed model
            RequiredUser = remove_default_values(User)
            processed = replace_default_values(
                user,
                RequiredUser
            )
            ```

        Collections:
            ```python
            class Item(BaseModel):
                tags: list[str] = []
                meta: dict[str, str] = {}

            item = Item(tags=[], meta={})
            processed = replace_default_values(item)
            assert processed.tags == "___UNKNOWN___"
            assert processed.meta == "___UNKNOWN___"
            ```

        Nested models:
            ```python
            class Address(BaseModel):
                street: str = ""

            class Contact(BaseModel):
                name: str
                address: Address = Address()

            contact = Contact(
                name="Alice",
                address=Address(street="")
            )
            processed = replace_default_values(contact)
            assert processed.address == "___UNKNOWN___"
            ```
    """
    target_model_type = target_model_type or remove_default_values(instance.__class__)
    target_model = target_model_type.model_validate_json(instance.model_dump_json())

    def replace_placeholders(model: BaseModel) -> BaseModel:
        """Recursively replace default values with placeholders in the model.

        Args:
            model: The model instance to process.

        Returns:
            BaseModel: A new model instance with placeholders replacing default values.
        """
        new_model_data: dict[str, Any] = {}
        for field_name, field_info in model.model_fields.items():
            field_value = getattr(model, field_name, None)
            default_container = next(
                (
                    metadata
                    for metadata in field_info.metadata
                    if isinstance(metadata, DefaultValueContainer)
                ),
                None,
            )

            new_value: Any
            match field_value:
                case BaseModel():
                    new_value = replace_placeholders(field_value)
                case list():
                    new_value = [
                        replace_placeholders(item) if isinstance(item, BaseModel) else item
                        for item in field_value
                    ]
                case dict():
                    new_value = {
                        key: replace_placeholders(value) if isinstance(value, BaseModel) else value
                        for key, value in field_value.items()
                    }
                case set():
                    new_value = {
                        replace_placeholders(item) if isinstance(item, BaseModel) else item
                        for item in field_value
                    }
                case tuple():
                    new_value = tuple(
                        replace_placeholders(item) if isinstance(item, BaseModel) else item
                        for item in field_value
                    )
                case _:
                    new_value = field_value

            if default_container and default_container.get_default() == field_value:
                new_value = PLACEHOLDER_VALUE

            new_model_data[field_name] = new_value

        return model.__class__.model_validate(new_model_data)

    target_model_with_placeholders = replace_placeholders(target_model)
    if not isinstance(target_model_with_placeholders, target_model_type):
        raise TypeError(
            f"Error replacing default values for model '{target_model_type.__name__}': "
            "transformed model is not a subclass of the target model"
        )

    return target_model_with_placeholders


def change_field_type(  # noqa: PLR0913
    model_type: type[T],
    field_name: str,
    new_type: Any,
    new_model_type: type[T] | None = None,
    new_model_name: str | None = None,
    default: Any = PydanticUndefined,
    **kwargs: Unpack[_FromFieldInfoInputs],
) -> type[T]:
    r"""Create model variant with modified field type.

    Creates a new model class based on an existing one, with one
    field modified or added. Preserves all other fields and model
    configuration.

    Args:
        model_type: Base model class to modify.
        field_name: Name of field to change or add.
        new_type: New type for the field.
        new_model_type: Optional new base class.
        new_model_name: Optional name for new model.
        default: Optional default value.
        **kwargs: Additional field configuration.

    Returns:
        New model class with modified field.

    Raises:
        TypeError: If default value is invalid.
        ValidationError: If default validation fails.

    Examples:
        Modify existing field:
            ```python
            class User(BaseModel):
                id: int
                name: str

            UserWithStringId = change_field_type(
                model_type=User,
                field_name="id",
                new_type=str,
                new_model_name="UserWithStringId"
            )
            ```

        Add new field:
            ```python
            UserWithEmail = change_field_type(
                model_type=User,
                field_name="email",
                new_type=str,
                # FieldInfo kwargs
                pattern=r"[^@]+@[^@]+\.[^@]+",
                description="Valid email address"
            )
            ```

        Validation rules:
            ```python
            UserWithAge = change_field_type(
                model_type=User,
                field_name="age",
                new_type=int,
                default=0,
                # FieldInfo kwargs
                ge=0,
                le=120,
                description="User age in years"
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
            temp_fields: dict[str, Any] = {field_name: (new_type, default)}
            temp_model = create_model("TempModel", **temp_fields)
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
