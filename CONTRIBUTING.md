# Contributing to telekinesis-illusion

`telekinesis-illusion` generates synthetic datasets for robotics and computer
vision. Contributions can improve generation, annotations, performance,
examples, or documentation.

Read the [README](README.md) for usage and the
[development guide](DEVELOPMENT.md) for environment setup, repository layout,
coding conventions, and validation commands.

## Report a bug

Search the [existing issues](https://github.com/telekinesis-ai/telekinesis-illusion/issues)
before opening a new one. Include:

- The expected behavior and what actually happened.
- A minimal script or YAML configuration and the exact command you ran.
- The commit or package version, OS, Python and `bpy` versions, and rendering
  device. For GPU problems, include the GPU model and driver version.
- The full traceback and relevant logs as text. For rendering or annotation
  problems, include a small image or mask showing the discrepancy.
- The random seed, model scale, and relevant assets or instructions for obtaining
  them. Share only assets you are allowed to redistribute.

Remove credentials and private data from reports. A reproduction using the
repository's sample assets is easier for others to investigate.

## Propose a change

For a substantial feature, dependency, architecture change, or breaking API
change, open an issue describing the use case and proposed behavior before
starting implementation. Explain why the current API cannot meet the need.
Small fixes and documentation corrections can go directly to a pull request.

## Submit a pull request

1. Fork the repository if you do not have write access. Create a focused branch
   from the current `main`, following the [branch naming guide](#branch-names).
2. Set up the project using [DEVELOPMENT.md](DEVELOPMENT.md#get-started).
3. Make the smallest complete change. Inspect affected callers and keep
   unrelated refactoring and formatting out of the diff.
4. Validate the behavior using the
   [relevant checks](DEVELOPMENT.md#validate-changes). Add regression coverage
   for behavior changes and update affected examples and documentation.
5. Open a pull request against `main`. Link a related issue when one exists.
   A draft pull request is useful when feedback is needed before completion.

Follow the [commit message guide](#commit-messages) to describe each change.

The pull request description should state:

- **Problem:** the concrete failure or use case.
- **Change:** the resulting behavior and any compatibility or migration impact.
- **Validation:** commands run, results, and relevant hardware or assets. State
  which checks were skipped and why.

For rendering changes, include comparable before/after images with the same
configuration and seed. For performance changes, include the workload, device,
and measured timings or memory use.

## Style guides

### Commit messages

Use this format, omitting the body and footer when they are unnecessary:

```text
<type>(<scope>): <subject>

<body>

<footer>
```

Choose a lowercase type that describes the main purpose of the change:

| Type | Purpose |
| --- | --- |
| `doc` | Update user or developer documentation. |
| `feat` | Add, change, or remove a feature. |
| `fix` | Correct a bug. |
| `refactor` | Restructure code without changing its behavior. |
| `revert` | Undo all or part of an earlier commit. |
| `style` | Change code formatting without changing behavior. |
| `chore` | Maintain tooling or repository configuration without changing runtime behavior. |

- **Scope:** name the affected component in lowercase, such as `scene`,
  `camera`, or `writer`.
- **Subject:** use a short, lowercase imperative phrase, such as
  `validate image dimensions`. Omit the trailing period.
- **Body:** explain what changed and why when the subject needs context.
- **Footer:** reference related issues using lowercase keywords, such as
  `closes #999` when the change resolves that issue.

For example:

```text
fix(camera): validate image dimensions

Reject non-positive dimensions before rendering.

closes #999
```

### Branch names

Include the issue number and a short, lowercase description, separated by
underscores:

```text
iss_<issue-number>_<short_description>
```

For example, a branch for issue #999 could be:

```text
iss_999_validate_camera_dimensions
```

## Before requesting review

- [ ] The change is focused and follows the
  [coding conventions](DEVELOPMENT.md#coding-conventions).
- [ ] Affected callers, configuration files, examples, and public documentation
  agree with the new behavior.
- [ ] Relevant tests and lint checks are reported, including any failures or
  checks that could not run. Documentation-only changes have been checked for
  accurate commands, working links, and Markdown rendering.
- [ ] No credentials, machine-specific paths, generated datasets, build output,
  or unrelated large assets are included.
- [ ] Added assets and adapted code retain their source and license information.

Review comments should focus on behavior, evidence, and maintainability. Keep
discussion respectful and explain tradeoffs when an alternative is proposed.

## License and attribution

The project license is recorded in [LICENSE](LICENSE). Preserve existing
copyright and license notices. For changes to adapted rendering code, consult
the [attribution notice](src/telekinesis/illusion/blender/NOTICE.md) and
[upstream license](src/telekinesis/illusion/blender/LICENSE.upstream).
Include provenance and license information with any new third-party code or
assets.
