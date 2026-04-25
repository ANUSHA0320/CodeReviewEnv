## Plan: Multi-Agent + Self-Improvement Upgrade

You want a “winning” next phase, but with a tight 24-hour scope. Given your choices (Multi-Agent Interactions + Self-Improvement), the fastest high-impact path is to: (1) turn the environment into a reviewer–author interaction loop (author is a simulated agent that responds to reviewer actions), and (2) add a simple self-improvement curriculum that adapts difficulty based on agent performance. This preserves your existing core while making the environment clearly multi-agent and adaptive, which aligns with judging criteria on innovation, reward design, and training evidence.

Steps
1. Extend environment state to track a review dialogue and author responses. Add fields like `conversation_history`, `author_reply`, and `iteration` in code_review_env/state.py, and include them in the observation built in code_review_env/env.py.
2. Implement a lightweight “AuthorAgent” inside the environment to respond to reviewer actions. For example, if the reviewer uses `request_changes`, the author can respond with a patch hint, re-test results, or a new diff. Implement the author logic and state updates in code_review_env/env.py.
3. Add curriculum-based self-improvement: after each episode, adjust difficulty sampling (easy → medium → hard) based on rolling performance, or increase PR complexity by sampling “harder” templates. Track performance and sampling logic in code_review_env/env.py and persist stats in code_review_env/state.py.
4. Update reward shaping to incentivize effective multi-agent negotiation and successful resolution (e.g., positive reward for reviewer actions that lead to author fixes, penalties for endless back-and-forth). Update logic in code_review_env/reward.py and align scoring in tasks/easy_task.py, tasks/medium_task.py, tasks/hard_task.py.
5. Update API and UI so users can see the conversation and evolving state. Add observation fields to responses in app/main.py and show author responses + conversation history in gradio_app.py.
6. Ensure OpenEnv compliance: add the new observation fields and any changes to the step response in openenv.yaml; verify action/obs schema consistency.
7. Update README to describe the new multi-agent loop + self-improvement curriculum, and add a short “how training improves behavior” section with a placeholder for plots/results in README.md.

Verification
- Run unit tests: pytest
- Manual check: run python gradio_app.py and verify conversation history, author replies, and curriculum progression.
- API check: call /reset and /step, confirm observation includes new dialogue fields and updated state.

Decisions
- Multi-agent interactions are simulated via an “AuthorAgent” inside the environment (fast to implement, fits 24-hour scope).
- Self-improvement is implemented as adaptive difficulty/curriculum rather than full self-play generation (faster, still aligned with theme).
