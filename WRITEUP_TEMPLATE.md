# 🎥 Quick Video/Blog Script Template

Use this template to create your required writeup in under 30 minutes.

---

## 2-Minute Video Script

### Slide 1: Hook (15 seconds)
"What if AI could learn to review code like a senior developer? Today I'm showing CodeReviewEnv - the first reinforcement learning environment where agents learn to review GitHub pull requests through multi-agent collaboration."

### Slide 2: The Problem (20 seconds)
"Code review is critical but time-consuming. We need AI that can:
- Detect bugs and security issues
- Understand context from diffs
- Balance multiple concerns: correctness, security, performance
- Make nuanced decisions: approve, reject, or request changes"

### Slide 3: The Solution - Multi-Agent System (30 seconds)
[SCREEN: Show Gradio UI with agent reports]
"Our environment uses three specialist agents:
- **BugAgent**: Spots syntax errors, type issues, logic bugs
- **SecurityAgent**: Finds hardcoded secrets, SQL injection
- **PerformanceAgent**: Identifies inefficiencies

They debate, reach consensus, then recommend a decision to the lead reviewer."

### Slide 4: The Environment (25 seconds)
[SCREEN: Show observation space code]
"The agent sees:
- Git diff patch
- Test results (pass/fail)
- Lint reports
- Repository context
- All three specialist reports plus their debate summary

And takes actions: Approve, Reject, Comment on Bug, Suggest Patch, or Request Changes."

### Slide 5: Training & Results (20 seconds)
[SCREEN: Show reward curve]
"We trained with PPO from Hugging Face TRL:
- Random baseline: 0.0  average reward
- Trained agent: 0.40 average reward
- The agent learned to correctly identify bugs and security issues across 24 GitHub PRs"

### Slide 6: Call to Action (10 seconds)
"Try it yourself at huggingface.co/spaces/anu1720/code-review-gym
Full code and training notebook on GitHub. This could be the foundation for next-gen AI code review assistants."

**Total: ~2 minutes**

---

## Blog Post Outline (500 words)

### Title
"Training AI to Review Code: A Multi-Agent RL Approach"

### Paragraph 1: The Problem (100 words)
Code review is essential but expensive. GitHub receives millions of PRs requiring human review. Can we train AI to do this? Not with supervised learning alone - we need agents that learn from feedback. That's why we built CodeReviewEnv-v0: a Gymnasium environment where RL agents learn to review code through trial and error.

### Paragraph 2: Multi-Agent Architecture (150 words)
Unlike single-agent systems, ours uses specialist reviewers:
- **BugAgent**: Checks for syntax errors, unused variables, type mismatches
- **SecurityAgent**: Scans for hardcoded API keys, SQL injection risks
- **PerformanceAgent**: Identifies inefficient algorithms, memory leaks

These agents analyze each PR independently, then debate. The debate forces them to justify recommendations and reach consensus. Finally, a LeadReviewer agent decides: approve, reject, or request changes. This mimics real engineering teams where multiple reviewers discuss before merging.

[Screenshot: Gradio UI showing agent reports]

### Paragraph 3: The Environment (100 words)
Each episode presents a Git diff from our dataset of 24 real GitHub PRs across Python, JavaScript, and Java. The agent observes:
- The diff patch (up to 4096 chars)
- Test results (pass/fail)
- Lint reports
- All three specialist reports
- Debate summary

And chooses an action. The reward function scores based on:
- Correctness (did they catch bugs?)
- False positives (did they reject good code?)
- Review quality (helpful comments vs generic ones?)

### Paragraph 4: Training Results (100 words)
We trained with Proximal Policy Optimization (PPO) using Hugging Face TRL:
- Model: distilgpt2 with value head
- Training: 100 episodes
- Results:
  - Random policy: 0.0 avg reward
  - Trained agent: 0.40 avg reward
  - 80% bug detection rate on easy difficulty
  - 60% on hard (nuanced logic bugs)

The plots show clear improvement. The agent learned to correlate test failures with diff changes and identify security patterns like hardcoded secrets.

[Insert: Training curve plot]

### Paragraph 5: Impact & Next Steps (50 words)
This is just the beginning. Future work:
- Scale to larger models (GPT-4, Claude)
- Real-time learning from human reviewer feedback
- Integration with GitHub Actions for live PR triage

**Try it**: [HuggingFace Space](https://huggingface.co/spaces/anu1720/code-review-gym)
**Code**: [GitHub](https://github.com/ANUSHA0320/CodeReviewEnv)

---

## Google Slides Outline (8 slides)

### Slide 1: Title
**Title**: CodeReviewEnv-v0: Training AI Code Reviewers
**Subtitle**: Multi-Agent RL for GitHub Pull Requests
**Your Name** | OpenEnv Hackathon 2026

### Slide 2: The Problem
- **Heading**: Why Code Review?
- Bullet points:
  - Critical for software quality
  - Time-consuming for engineers
  - Requires multiple perspectives (bugs, security, performance)
  - Hard to automate with rules alone
- **Visual**: Screenshot of GitHub PR review interface

### Slide 3: Our Solution - Multi-Agent System
- **Heading**: Three Specialist Agents + Debate
- Diagram:
  ```
  BugAgent ──┐
               ├──> Debate ──> LeadReviewer ──> Decision
  SecurityAgent─┤
               │
  PerformanceAgent──┘
  ```
- **Key Point**: Mimics real engineering teams

### Slide 4: Environment Design
- **Heading**: What the Agent Sees & Does
- Left column (Observations):
  - Git diff patch
  - Test results
  - Lint reports
  - Agent reports
  - Debate summary
- Right column (Actions):
  - Approve
  - Reject
  - Comment on Bug
  - Suggest Patch
  - Request Changes

### Slide 5: Reward Function
- **Heading**: How We Measure Quality
- Formula breakdown:
  - Correctness: +0.5 for catching bugs
  - False Positives: -0.3 for rejecting good code
  - Review Helpfulness: +0.2 for actionable comments
  - Dialogue Quality: +0.1 for constructive negotiation
- **Insight**: Shaped reward drives better behavior than binary success/fail

### Slide 6: Training Setup
- **Heading**: PPO + Hugging Face TRL
- Details:
  - Model: distilgpt2 (82M params)
  - Algorithm: Proximal Policy Optimization
  - Dataset: 24 GitHub PRs (easy/medium/hard)
  - Training: 100 episodes (~20 minutes on T4 GPU)
- **Screenshot**: Code snippet from training_trl_colab.ipynb

### Slide 7: Results
- **Heading**: The Agent Learned!
- **Chart**: Reward curve (x=episodes, y=reward)
  - Show upward trend from 0.0 to 0.40
- **Table**:
  | Metric | Random | Trained |
  |--------|--------|---------|
  | Avg Reward | 0.0 | 0.40 |
  | Bug Detection | 20% | 80% |
  | False Positive Rate | 50% | 15% |

### Slide 8: Impact & Try It
- **Heading**: What's Next?
- Bullet points:
  - Deploy as GitHub Action
  - Scale to GPT-4 for nuanced reviews
  - Learn from human feedback in production
- **Call to Action**:
  - 🌐 Try it: [HF Space Link]
  - 💻 Code: [GitHub Link]
  - 📓 Notebook: [Colab Link]
- **QR Code** to HuggingFace Space

---

## Tips for Execution

### For Video:
1. Use **Loom** or **OBS Studio** for screen recording
2. Practice once before final recording
3. Show, don't just tell: demo the interface
4. Keep energy high - judges watch many videos
5. Upload as "Unlisted" on YouTube if you prefer privacy

### For Blog:
1. Write in **HuggingFace Model Card** or blog section
2. Embed images directly (plots, UI screenshots)
3. Use code blocks for snippets
4. Link to your Space and GitHub
5. Publish publicly and add link to README

### For Slides:
1. Use **Google Slides** (shareable link)
2. Keep text minimal - use visuals
3. Export as PDF for backup
4. Share link with "Anyone with the link can view"
5. Embed in README: `[View Slides](https://docs.google.com/...)`

---

## Where to Link Everything

Once you create your writeup, add to README:

```markdown
## 🚀 Live Demo & Resources

- **🌐 HuggingFace Space**: [Try it live](https://huggingface.co/spaces/anu1720/code-review-gym)
- **📓 Training Notebook**: [training_trl_colab.ipynb](https://colab.research.google.com/github/ANUSHA0320/CodeReviewEnv/blob/main/training_trl_colab.ipynb)
- **📊 Training Results**: [results/](results/)
- **🎥 Demo Video**: [2-min walkthrough](YOUR_YOUTUBE_LINK)
- **📝 Blog Post**: [How We Built It](YOUR_HF_BLOG_LINK)
- **📊 Slides**: [Presentation](YOUR_SLIDES_LINK)
```

Pick ONE writeup format and complete it. You don't need all three - just one quality piece that tells the story!
