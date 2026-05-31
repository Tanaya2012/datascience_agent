# 📊 **Interactive Data Cleaning Agent**

## 🎯 **What It Is**

Your agent is an **interactive, intelligent, LLM-guided data cleaning assistant** that helps users take raw tabular datasets (CSV/Excel or Kaggle) and transform them into structured, analysis-ready data while keeping a clear cleaning audit trail.

At its core, this agent:

* **Understands user intent in natural language**
* **Asks clarifying questions when needed**
* **Breaks down the data cleaning problem into ordered steps**
* **Executes validated cleaning operations**
* **Generates outputs (cleaned dataset, logs, and quality reports)**

It’s designed as a **modular, extendable MVP** — tomorrow you can add DAG planning, reflection loops, memory, schema drift detection, etc.

---

## 🤖 **Powered by Google’s ADK**

Your agent uses **Google’s Agent Development Kit (ADK)** as the backbone for interactive behavior. ADK provides:

✔ A framework to define **stateful conversational agents** that can reason, use tools, and remember context in a session. ([google.github.io][1])
✔ A model-agnostic way to integrate LLMs (Gemini/other). ([google.github.io][1])
✔ Support for tools — Python functions the agent can call during a conversation. ([Medium][2])
✔ Session state management, keeping lightweight context across turns. ([google.github.io][3])

Rather than being a *static script*, your agent becomes a **conversational orchestrator**, reasoning and acting step-by-step with LLM help.

---

## 🧠 **How It Thinks and Acts**

### 1️⃣ **Natural Language Problem Intake**

The user describes what they want, e.g.:

> “Clean my sales data, handle missing values, and merge with customer info.”

The agent then:

* Validates user intent
* Asks follow-ups (e.g., dataset source, merge key)
* Stores responses in session state

This keeps the experience conversational and interactive, not a one-shot CLI.

---

### 2️⃣ **LLM-Driven Task Planning**

Instead of hard-coding a task sequence, the agent uses LLM reasoning to **generate a task plan**:

```
[
  {task_type: "load_data", ...},
  {task_type: "profile_data"},
  {task_type: "clean_missing"},
  ...
]
```

But the list is validated against an allowed enum so the agent can’t invent unsupported operations. This ensures robustness while retaining intelligence.

---

### 3️⃣ **Plan Confirmation**

Before executing, it *presents the plan back to the user*:

> “I’ll load your dataset, profile columns, handle missing values with medians, standardize dates, dedupe, and merge on key X. Proceed?”

This confirmation step gives transparency and control.

---

### 4️⃣ **Task Execution with Tools**

Each task (load, profile, clean, dedupe, merge, validate) runs via a **tool** — a wrap-around Python function the agent calls.

Tools:

* Return structured output (data, metadata, logs, confidence)
* Update the **execution state**
* Don’t store large data in session state (artifacts are used for that)

This pattern keeps your agent scalable and safe.

---

### 5️⃣ **Transformation History**

Rather than storing full dataframes in memory every step, the system logs:

* Shapes before/after
* Columns affected
* Rows removed
* Checksum hashes

These structured transformation records provide traceability without bloat.

---

## 📦 **Component Summary**

### 📍 **Interface & Interaction**

Interactive multi-turn conversation using ADK session state.

### 🧠 **Task Decomposer**

LLM generates a **validated task list** from user intent within allowed operations.

### 📆 **Planner**

Constructs linear, ordered task objects (no graph yet, just sequence), with defaults from config.

### ⚙️ **Execution Engine**

Executes tasks sequentially via tools, manages state, and logs detailed transformation metadata.

### 📊 **Cleaning Modules**

Each cleaning tool returns:

```python
{
  "data": df, "metadata": {...},
  "logs": [...],
  "confidence": float
}
```

This consistent contract allows future expandability.

### 📋 **Validator**

Produces a quality score (0–100) based on missingness, duplicates, and merge issues.

### 📤 **Output Generator**

Returns in-memory bytes for:

* Cleaned CSV
* Cleaning log
* Data quality report

Users can download these as needed.

---

## 💡 **Why This Approach Works**

### 🧠 **Hybrid Intelligence**

LLM reasoning + deterministic execution tools gives you:

* Clarity (explainable steps)
* Adaptability (LLM reasoning)
* Reliability (validated task lists)

This sets you up for future upgrades like DAG planning or reflection loops.

---

## 🛠️ **Built for Growth**

This MVP:

✔ Is modular and extendable
✔ Has clear tool boundaries
✔ Keeps planning separate from execution
✔ Uses config-driven defaults
✔ Stores artifacts efficiently via ADK
✔ Preserves conversational context and state

Most importantly, it builds the foundation for a *full-blown agentic data assistant* without premature complexity.

---

## 📌 Final Thought

This is not a script — it’s a **stateful, interactive, LLM-orchestrated agent** built with a robust agent framework. It listens, reasons, plans, confirms, executes, explains, and outputs — exactly the kind of tool needed for real-world data workflows.

---
