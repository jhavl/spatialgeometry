Thanks for contributing to spatialgeometry!

## Summary

<!-- What does this PR do, and why? -->

## Related issue

<!-- Fixes #123 / Closes #123 -- if applicable -->

## Checklist

Nothing below is checked automatically -- this is a self-check for you before requesting review.

- [ ] PR title follows [Conventional Commits](https://www.conventionalcommits.org/) (`type: description`)
- [ ] Tests pass locally (`pip install .[dev,collision] && pytest`)
- [ ] Added/updated tests for this change, if applicable
- [ ] New/changed code is type-hinted with modern syntax (`X | Y`, `list[X]`, not `Union`/`Optional`/`List`)
- [ ] Docstrings updated (reST style: `:param:`, `:returns:`; type hints in the signature cover types now, `:type:`/`:rtype:` are rarely needed)
- [ ] If the C++ extension changed (`src/spatialgeometry/cpp-extension/`, `CMakeLists.txt`): the pure-Python fallback (`scene.py`) still matches -- it's what Pyodide/JupyterLite actually runs, since nothing gets compiled there
- [ ] PR is as small/focused as practical -- if it tackles several unrelated things, consider splitting it so each can be reviewed and accepted independently

<!-- Target branch is `main`. -->
