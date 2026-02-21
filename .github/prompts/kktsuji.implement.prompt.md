---
description: Implementation Command
argument-hint: "#file:implementation-plan.md"
---

# Implementation Command

## Instruction

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding.

## Command Usage

`/kktsuji.implement #file:implementation-plan.md`

Take the user input and implement following the plan in it.

## Outline

1. **Understand the User Input**: Analyze the user input to grasp the requirements and objectives of the implementation. User input has a TODO list of implementation phases and tasks. Each phase may have multiple tasks that need to be completed to achieve the overall implementation goals.
2. **Read Documentation**: If necessary, read relevant documentation or resources to gather information that will aid in creating the plan.

- `docs/standard/architecture.md`: This document provides an overview of the system architecture.
- `README.md`: This document contains general information about the project.
- `.github/prompts/kktsuji.venv.prompt.md`: This document provides guidance for using the Python virtual environment.

3. **Implement the Plan**: Follow the tasks and execute the implementation step-by-step according to the plan outlined in the user input. This may involve writing code, creating new files, modifying existing files, and performing necessary actions to complete each task in the implementation plan.
4. **Mark Tasks as Completed**: As you complete each task, mark it as completed in the implementation plan. This helps to keep track of the progress and ensures that all tasks are addressed systematically.
5. **Commit Message**: After completing the implementation, generate a commit message like the following format `feat: <title><blank line><description>`. The title should be a concise summary of the changes made, and the description should provide more details about the implementation. You do NOT need to execute the `git add` or `git commit` commands, just generate the commit message text.
