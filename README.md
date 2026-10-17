# agents_mechanism



 history = copy(chat_history)
        │
        ▼
 ┌─ 1. ask_clarifying_questions ───────────────┐   skipped if skip_clarification
 │  LLM(tools=[generate_plan], choice=auto)    │
 └─────────────────────────────────────────────┘
        │ called generate_plan          │ replied with text
        ▼                               └──► return ClarificationRequest(questions)
 ┌─ 2. generate_research_plan ─────────────────┐
 │  LLM(no tools, streamed)                    │   research_plan: fixed string,
 │  → "1. ...\n2. ...\n..."                    │   never updated afterwards
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─ 3. run_orchestrator  (diagram 3) ──────────┐   appends tool calls + agent
 │                                             │   reports to history
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─ 4. generate_final_report ──────────────────┐
 │  LLM(FINAL_REPORT_PROMPT + history          │
 │      + reminder containing the plan)        │
 └─────────────────────────────────────────────┘
        │
        ▼
 return ResearchReport(plan, report, sources cited in report)




<img width="1900" height="1600" alt="image" src="https://github.com/user-attachments/assets/88f2e0bb-595f-4eed-a198-2a580c3f93a0" />
