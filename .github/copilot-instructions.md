# Instructions for AI Agents


### Environment Notes
See [.github/AGENT_NOTES.md](.github/AGENT_NOTES.md).

### Debugging Notes
See [.github/AGENT_DEBUGGING.md](.github/AGENT_DEBUGGING.md).

## Code Writing Practices

### Readability

#### Prefer Enums and Literals
Prefer expressive control-state types (`Enum` or `Literal`) over ambiguous `bool` flags when code flow depends on named states (for example, movement direction).

#### Split Long Functions into Smaller Ones
When independent parts of a logic block can be extracted, split them into small, named sub-functions instead of keeping one long function.

#### Type-hinting:
Use type hints for all functions, including return types. This improves readability and linting and helps catch bugs.

- Avoid using `Any` as a type hint. Instead, use more specific types or create custom types if necessary.
- If many outputs are needed from a function, consider using a `TypedDict` or a `dataclass` to return a structured object instead of a general tuple\dict.