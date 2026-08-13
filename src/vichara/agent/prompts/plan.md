Plan how to answer the question below.

Produce between {min_steps} and {max_steps} steps. Each step names one intended
tool and the criterion that would tell you the step succeeded.

Judge three things before planning:

- **Answerable at all?** If the question rests on something that does not
  exist, asks for information no tool of yours could reach, or needs a
  capability you do not have, set `answerable: false` and explain why in one
  sentence. Do not plan steps for a question you cannot answer -- refusing on
  step one is correct behaviour, refusing on step twelve is a failure.
- **Ambiguous?** If it could reasonably mean two different things and the
  answers differ, set `needs_clarification: true` and give the single question
  that would resolve it.
- **Otherwise**, plan the shortest path that produces a cited answer.

The plan is advisory. You may deviate later if what you find warrants it.

Tools available this session: {tools}
{capability_notice}

Question: {task}
