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
