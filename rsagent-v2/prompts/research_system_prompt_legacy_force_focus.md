> **Deprecated — do not use for new runs.**  
> The repository default is `prompts/research_system_prompt.txt` (**plan-first**, no global mandatory tool order).  
> Earlier versions of this file required calling `focus_select_keyframes` before web search; that **global** requirement has been **removed**. Configure tools only through your Research Plan and the host (FOCUS enabled or not).

Copy `prompts/research_system_prompt.txt` (or set `RESEARCH_SYSTEM_PROMPT_PATH`) and adjust wording there if you need stricter per-run behavior.
