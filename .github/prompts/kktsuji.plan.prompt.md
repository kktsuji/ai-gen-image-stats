---
description: Plan Command
argument-hint: "What you want to implement."
---

# Plan Command

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding.

## Command Usage

`/kktsuji.plan <what you want to implement>`

Take the user input and create a plan to implement it.

## Outline

1. **Understand the User Input**: Analyze the user input to grasp the requirements and objectives of the implementation.
2. **Read Documentation**: If necessary, read relevant documentation or resources to gather information that will aid in creating the plan.

- `docs/standards/architecture.md`: This document provides an overview of the system architecture.
- `README.md`: This document contains general information about the project.

3. **Investigate the Codebase**: Investigate the codebase to understand the existing structure and identify where changes or additions may be needed to implement the desired functionality. This may involve reviewing relevant files, modules, or components that are related to the implementation.
4. **Create a Plan**: Based on the understanding of the user input and the information gathered from the documentation and codebase, create a detailed plan outlining the steps required to implement the desired functionality.
5. **Create New Documentation**: Create a new file named `yyyymmdd_<title>.md` in the `docs/features/` directory, where `yyyymmdd` is the current date and `<title>` is a concise title summarizing the implementation.
6. **Write the Plan**: Write the plan in the newly created documentation file, ensuring that it is clear, concise, and well-structured for easy understanding and implementation by developers. Follow the format:

```markdown
# <Title>

## Overview

You can write an overview of the implementation here. This section should provide a high-level summary of what the implementation is about and its purpose. (e.g., objective, design, goals, expected outcomes, new architecture, file changes, time estimates).

Note: Keep this section concise and focused on the key points to provide a clear understanding of the implementation at a glance.

## Implementation Checklist

- [ ] Phase 1: XXX
  - [ ] Task 1.1: YYY
  - [ ] Task 1.2: ZZZ
- [ ] Phase 2: AAA
  - [ ] Task 2.1: BBB
  - ...

Note:

- Add the tests for implementation phases to ensure that the new functionality works as expected and does not introduce regressions.
- Add the tests phase to validate that all tests pass after all the implementation phases are completed.
- If necessary, add the documentation phase to update or create new documentation related to the implementation (e.g., `README.md`, `docs/standards/architecture.md`).

## Phase Details

### Phase 1: XXX

...
```
