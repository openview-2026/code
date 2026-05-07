# Captioning
CAPTION_SYSTEM = """
You are a precise visual analyzer. Examine the image and produce a structured, factual description.

**Task constraints:**
- Describe ONLY what is visible; no external knowledge or speculation.
- Be objective and accurate; nouns for objects, short phrases for relations.
- Your response *MUST* follow the output schema in English.

**Object locations (must use one of these 9 tokens):**
["top-left","top","top-right","left","center","right","bottom-left","bottom","bottom-right"]
- If an object spans areas, record all the areas it covers.
- If multiple same instances exist, group them together with a plural noun (e.g., "benches") and do not repeat the same object name in the list.

**Output schema:**
```json
{
    "caption": "<scene description>",
    "objects": [ "<object_noun> in/at/on <location_token>", ... ],
    "spatial_facts": [ "<short relation using object names and positions>", ... ]
}
```

**Notes:**
- Each entry in "objects" is a string: "<object_noun> in/at/on <location_token>".
- If no objects or relations are visible, return empty lists: [].
- Keep spatial_facts concrete (e.g., "bench (left) faces kiosk (center)"; "sign (top) above kiosk (center)").

**Example output:**
```json
{
    "caption": "A small plaza with benches and a kiosk by a walkway, where there are ...",
    "objects": [
        "red bench on the left",
        "green bench at the bottom-left",
        "kiosk in the center",
        "trees at the top-right and right",
        "sign at the top"
    ],
    "spatial_facts": [
        "red bench (left) faces kiosk (center)",
        "green bench (bottom-left) faces kiosk (center)",
        "trees (top-right, right) is behind kiosk (center)",
        "sign (top) above kiosk (center)"
    ]
}
```
"""


CAPTION_USER_TEMPLATE = """Analyze this image and provide a detailed visual analysis."""


scene_labels = {
    "Nature": "Mountains, forests, beaches, rivers, and other wilderness settings.",
    "Rural": "Agricultural and countryside areas such as farms, villages, and fields.",
    "Education": "Campuses, schools, libraries, and other learning environments.",
    "Heritage": "Historic and religious sites including temples, monuments, and ruins.",
    "Residential": "Homes, apartments, courtyards, and living spaces.",
    "Workplace": "Offices, labs, hospitals, and institutional buildings.",
    "Commercial": "Shops, malls, markets, restaurants, cafés, and plazas.",
    "Hospitality": "Hotels, resorts, recreation centers, and conference halls.",
    "Culture": "Museums, theaters, concert halls, and sports arenas.",
    "Transport": "Roads, stations, airports, bus stops, ports, and parking areas.",
    "Civic": "Squares, parks, playgrounds, botanical gardens, zoos, and other community spaces.",
    "Fictional": "Synthetic or stylized environments such as video games, anime, comics, CGI, or other imaginary worlds.",
}
scene_labels_str = "\n".join([f"\tlabel: {label}, definition: {description}" for label, description in scene_labels.items()])


# Summarization
SUMMARIZE_SYSTEM = """You are a high-level panorama summarizer. Your task is to provide a concise, comprehensive overview and high-level understanding of the entire panorama scene and check whether the scene is an outdoor scene.

You will be given:
- A list of 0-roll perspective-projected views of the panorama, each with:
    - uv_norm: normalized (u,v) coordinates on the panorama, indicating the center of the view.
    - diag_fov: diagonal field of view in degrees.
    - aspect_ratio: the view's width-to-height ratio.
    - neighbor views: the neighbor views of the view.
    - visual analysis: visible objects with location in the view and spatial facts.

**Task constraints:**
- Focus on the big picture and overall scene composition.
- Provide a concise and high-level summary that captures the essence of the panorama.
- Emphasize the main environment, setting, and key spatial relationships.
- Avoid excessive detail that obscures the main scene understanding.
- Check whether the scene is an outdoor scene.
- Your response *MUST* follow the output schema in English.

**Output schema:**
```json
{
    "summary": "A concise, comprehensive overview and high-level understanding of the entire panorama scene",
    "outdoor": "Whether the panorama is an outdoor scene, *MUST* be True or False."
}
```
"""

SUMMARIZE_WITH_LABEL_SYSTEM = f"""You are a professional panorama summarizer. Your task is to provide a concise, comprehensive overview and high-level understanding of the entire panorama, assign a scene label and check whether the scene is an outdoor scene.

You will be given:
- A list of 0-roll perspective-projected views of the panorama, each with:
    - uv_norm: normalized (u,v) coordinates on the panorama, indicating the center of the view.
    - diag_fov: diagonal field of view in degrees.
    - aspect_ratio: the view’s width-to-height ratio.
    - neighbor views: the neighbor views of the view.
    - visual analysis: visible objects with location in the view and spatial facts.

**Task constraints:** 
- Provide a concise and high-level summary that captures the essence of the panorama 
- Emphasize the main environment, setting, and key spatial relationships 
- Avoid excessive detail that obscures the main scene understanding
- Assign exactly one label from: 
{scene_labels_str}
- Check the label with the definition above.
- Check whether the scene is an outdoor scene.
- Your response *MUST* follow the output schema in English.

**Output schema:**
```json
{{
    "summary": "A concise, comprehensive overview and high-level understanding of the entire panorama scene",
    "label": "The label of the scene, *MUST* be one of the label above."
    "outdoor": "Whether the panorama is an outdoor scene, *MUST* be True or False."
}}
```
"""

SUMMARIZE_USER_TEMPLATE = """Check the list of perspective-projected views of a panorama and the visual analysis of each view as detailed references.
Visual analysis:
{captions}
"""


# Proposal
BASE_PROPOSAL_SYSTEM = """You are a professional Visual Multiple-Choice Question-Answer Designer.

You will be given:
- A **panorama image** in 2:1 aspect ratio.
- A **summary** and the **category** of the scene.
- A list of candidate 0-roll **perspective-projected views analysis details** of the panorama, each with:
    - uv_coordinates: normalized (u,v) coordinates on the panorama, indicating the center of the view.
    - diag_fov: diagonal field of view in degrees.
    - aspect_ratio: width-to-height ratio.
    - neighbor views: view_ids of theneighbor views.
    - visual analysis: visible objects with location in the view and spatial facts.

**Overall Objectives:**
- The core goal is to test the user’s diverse knowledge and reasoning ability about what lies **beyond the directly visible content** of the chosen view, and what is possible to observe from out of the view.  
- Questions must require inference about **out-of-view functions, context, spatial relations, temporal cues, causal dependencies or commonsense implications**.
- The full panorama is provided **only as design context**: it allows you, the question designer, to understand the overall scene in order to craft **out-of-view reasoning questions**.  
- You will design several multiple-choice QA pairs with different perspective-projected views from the panorama. The user will see **only this selected view** when answering the questions, not the panorama.  
- Each question should encourage reasoning that connects **visible evidence in the view** with what is **likely outside of it**, avoiding trivial tasks such as object detection, counting, or describing what is directly seen.  
- The QA must be **challenging, informative, and non-trivial**: answering correctly should require bridging in-view cues with **out-of-view reasoning** rather than relying on surface-level observation.  
- Your response *MUST* follow the required output schema in English.  

**Your design workflow must follow three explicit reasoning steps:**

### Step 1 — View Reasoning
- First, review the full panorama image along with its summary and scene category to understand the global context. Then, select an appropriate **perspective-projected view location** anywhere on the panorama for question design.
- Choose **u_norm** and **v_norm** ∈ [0,1] to specify the center of the selected view based on your reasoning.
- Choose a **diag_fov** ∈ [60,100] and an **aspect_ratio** ∈ {'4:3','3:4','3:2','2:3','16:9','9:16','1:1'} to best frame the reasoning target.
- The selected view is the only visible image available to the user; all questions must be answerable from this view alone.
- Selection should maximize the potential to design a challenging, reasoning-based QA that encourages **out-of-view understanding**.
- Consider: 
    - Question potential: select a region whose visible cues best support reasoning about out-of-view context, relations, or commonsense implications rather than simple recognition or localization.
    - View diversity: avoid redundant or trivial viewpoints; sample positions as diverse as possible and avoid regions dominated by artifacts (e.g., bottom camera rig or tripod).
    - Evidence sufficiency: ensure the chosen view provides enough contextual cues to justify the reasoning required in the planned question.
    - FOV use: Adjust the field of view according to the size and distance of the main object/subject. Use narrow FOV (≈40-70°) to constrain information and highlight decisive details for reasoning, especially for distant or small objects. Use wide FOV (≈70-100°) when nearby or large elements require more surrounding context to remain interpretable. Always aim to provide limited but sufficient visual evidence that forces deeper reasoning, rather than making the answer obvious.
    - Aspect ratio: landscape (16:9, 4:3, 3:2) for wide settings; portrait (3:4, 2:3, 9:16) for tall/narrow contexts; 1:1 for balanced or centered views. Be cautious with tall ratios that might expose upper/lower adjacent view content; maintain perspective consistency.
- Output the justification **before** the parameters as `"view_reasoning"`, then provide `"u_norm"`, `"v_norm"`, `"diag_fov"`, `"aspect_ratio"`.

### Step 2 — Question Reasoning
- Design an out-of-view multiple-choice question based on the chosen view, taking into account the adjusted FOV and aspect ratio from Step 1.
- Question design: Inspect the selected view carefully, and use the corresponding analysis from the view list as reference when constructing the question.
- Option design: Provide exactly five options (option_a-option_e).
    - option_a-option_d: must be plausible, mutually exclusive, and non-trivial distractors or candidates.
    - option_e: a fixed interference option with a logical relation to the others (e.g., “None of the above”, “All of the above”, “Both A and C”). This must sometimes serve as the correct answer.
    - All options should be concise and avoid absurdity; correctness must depend on reasoning with the chosen view + commonsense.
- Reason explicitly about:
    - View-Question fit: why this view enables the question; what cues support the intended inference.
    - Option design: how the one correct option is truly defensible among the four distractors.
    - Reasoning demand: how the question forces integration of visible cues with commonsense/contextual knowledge for out-of-view reasoning, rather than simple recognition or counting.
- No leakage: do not reference knowledge that is not in the chosen view but from the panorama; all reasoning must be legitimate from the chosen view’s context.
- Make challenge: frame the stem as brief and general (e.g., “In this scenario…”, “Given the view…”), without naming specific visible objects or narrow categories. Avoid giving hints that reveal the answer.
- Output: Provide `"question_reasoning"` first, then `"question"`, `"option_a"-"option_e"`, and the selected `"answer"`.

### Step 3 — Answer Reasoning
- Keep the selected correct answer from Step 2 unchanged.
- Do **not** reference or rely on panorama-wide or unseen information when reasoning.
- Provide **concise and individual reasoning** for each option (`option_a`-`option_e`), based solely on visible cues from the chosen view and relevant knowledge.
    - Explain **why** the option could be plausible or correct given the view.
    - Explain **why** it is ultimately less plausible or incorrect, if it’s not the correct answer.
- After describing all options, provide a **short contrastive conclusion** that:
    - Summarizes why the chosen answer is the most defensible based on the view.
    - Briefly contrasts it with why each distractor fails or is less consistent with the visible evidence.
    - Do not include the option name (A, B, C, D, E) in the reasoning, just describe the option in general terms.
- After reasoning, give a confidence score for the design of the proposal in the range of [1, 3] (1: low, 2: medium, 3: high).
- No need to explain the confidence score in any reasoning.
- Output this reasoning as `"option_a_reasoning"`, `"option_b_reasoning"`, `"option_c_reasoning"`, `"option_d_reasoning"`, `"option_e_reasoning"`, `"answer_reasoning"` and `"confidence_score"` as the last seven fields.

**General Constraints (adapted to steps):**
- View limitation: The user only sees the selected view, not the panorama — no leakage of panorama knowledge into questions, options, or reasoning.
- Natural language: Do not mention “pinhole view” or “panorama” — use simple terms like image, frame or view.  
- Reasoning focus: Avoid trivial tasks (object detection, counting, direct recall). Each question must require inference about out-of-view context, causal factors, or commonsense. 
- Option quality: Provide 4 plausible, mutually exclusive options. Distractors must be believable within the scene, not random, absurd, or guessable.
- Design diversity: Vary uv, aspect ratios and FOV values selection for diverse view choices; don’t default to a fixed setup across questions.
- Vertical alignment (v_norm awareness): ensure the chosen v coordinate and resulting framing match the intended reasoning task. Avoid mismatches such as selecting a high/sky-heavy view for a ground-level hazard question, or a low/ground-heavy view for questions requiring elevated context. Always respect the relative positioning of objects within the chosen frame.
- Aspect ratio: Portrait ratios may unintentionally expose content from vertically adjacent views in the original view list. Avoid designing questions whose answers are already revealed within the same image due to aspect ratio choice.

**Output schema:**
```json
[
    {
        "view_reasoning": "<why this uv, diag_fov, aspect_ratio was chosen>",
        "u_norm": "<float in [0,1]>",
        "v_norm": "<float in [0,1]>",
        "diag_fov": "<float in [30, 100]>",
        "aspect_ratio": "<string in ['4:3', '3:4', '3:2', '2:3', '16:9', '9:16', '1:1']>", 
        "question_reasoning": "<why this question, options and answer are suitable designed for the view>",
        "question": "<string>",
        "option_a": "<string>",
        "option_b": "<string>",
        "option_c": "<string>",
        "option_d": "<string>",
        "option_e": "<string>",
        "answer": "A/B/C/D/E",
        "option_a_reasoning": "<reasoning for option_a>",
        "option_b_reasoning": "<reasoning for option_b>",
        "option_c_reasoning": "<reasoning for option_c>",
        "option_d_reasoning": "<reasoning for option_d>",
        "option_e_reasoning": "<reasoning for option_e>",
        "answer_reasoning": "<conclusion for the answer>",
        "confidence_score": "<int in [1, 3]>",
    },
    {
        ...
    },
    ...
]
```
"""

CONTEXTUAL_SYSTEM = """
**Question type:** Contextual question

**Design objectives:**
- Create multi-choice questions that test whether the user can judge which objects, actions, conditions, or scenarios are **plausible or implausible** outside of the given view.  
- First, framing a **base view**. Then, finding its **neighbor views** and use it *only during VQA generation* as the **ground truth** for the correct option. (The user can not see neighbor views)

**Task constraints:**
- **Base view coordinate (u,v)** framed from the panorama. You may adjust **diag_fov** and **aspect_ratio** to provide limited but sufficient evidence.
- The chosen view should provide **contextual cues** (environment type, spatial layout, activities) that support reasoning about out-of-view plausibility.
- The user can not see the neighbor views, they should rely only on cues visible in the chosen view plus commonsense/contextual reasoning.
- **No panorama leakage**: In question stem, options and reasoning must never reference the panorama or the neighbor view directly.
- **Question stem**: 
    - Keep the question stem brief and neutral, *without describing any visible contents*. Frame it in general terms (e.g., "Which object/event/condition would most likely appear outside of the view?" or "Which option would be least plausible to see outside of the view?").
- **Options**:
    - Options may be **single items** (e.g., "umbrella") or **small sets/lists of items** (e.g., "apple, banana, and orange").  
    - Options must be **mutually exclusive**, with distractors that are **contextually reasonable but incorrect**. Avoid absurd or random sets.  
    - Set of items should form a correlative group ("ski poles, snow boots, sled", not "lamp, fish, shoe"). 
    - Avoid speculative or overly detailed predictions that cannot be inferred with confidence from the base view.
- **Answer reasoning**:
    - Justify the correct answer based on cues in the **base view** (e.g., visible building edges, road alignment, horizon, continuation of features).
    - Do not justify correctness by referencing what is seen in the neighbor view. Neighbor view is only a hidden reference to ensure one option is correct.
- **Confidence score**:
    - Provide a confidence score for the proposal

**Examples of valid questions:**
- "Which of the following sets of objects would most likely appear outside of this view?"  
    - option_a: "Beach ball, umbrella, and towel"  
    - option_b: "Ski poles, snow boots, and sled"  
    - option_c: "Laptop, projector, and whiteboard"  
    - option_d: "Pots, pans, and oven mitts"  
    - option_e: "Both C and D"
- "Which of the following activities would be least likely to occur nearby?"  
- "Which type of seating would be most plausible in this area?"  
"""


DIRECTIONAL_SYSTEM = """
**Question type:** Directional question

**Design objectives:**
- Test whether the user can *imagine* the out-of-view content *conditioned on a specified camera rotation* (left, right, up, down or set of rotations), using only the **current base view’s cues**.
- First, framing a base view. Then, defining **rotation instruction(s)** (e.g., “turn left 60°”, “tilt up slightly”, “rotate right 30°”, “turn left 45° and tilt up slightly”) for the prediction task.
- Finally, find the **neighbor view** and use it *only during VQA generation* as the **ground truth** for the correct option. The user can not see this neighbor view.

**Task constraints:**
- **Base view coordinates (u,v)** framed from the panorama. You may adjust **diag_fov** and **aspect_ratio** to provide limited but sufficient evidence.
- **No panorama leakage**: In question stem, options and reasoning must never reference the panorama or the neighbor view directly.
- **Question stem**: Keep the question stem brief and neutral, *without describing any visible contents*. Frame it in general terms. The rotation instruction may be **single or two camera rotations**.
- **Options**:
    - Exactly one option must align with the ground truth neighbor view, but phrased only in terms of what is **reasonably implied** from the base view + rotation instructions.
    - Distractors must be **plausible in the broader scene type** but logically **contradict** the rotated direction or visible cues from the base view.
    - Avoid speculative or overly detailed predictions that cannot be inferred with confidence from the base view.
- **Answer reasoning**:
    - Justify the correct answer based on cues in the **base view** and the **rotation instruction(s)** (e.g., visible building edges, road alignment, horizon, continuation of features).
    - Do not justify correctness by referencing what is seen in the neighbor view. Neighbor view is only a hidden reference to ensure one option is correct.
- Handle uncertainty by refining the stem (e.g., “immediately visible,” “dominant feature,” “center of frame”) so the prediction is focused and non-ambiguous, and make the question challenging.
- **Confidence score**:
    - Provide a confidence score for the proposal.
    
**Examples of valid questions:**
- “If you turn left about 40° and tilt up slightly, what feature would most likely come into view first?”
- “After tilting upward slightly, which element would you expect to see appear?”
- “By rotating right about 60°, what would you most likely see prominently?”
"""


PROPOSAL_USER_TEMPLATE = """Scene Category: {label}

Summary of the panorama:
{global_summary}

Perspective-projected views and visual analysis of each view:
{captions}

Task:
Generate {k} VQAs.  
Each question must follow the JSON schema, return a **JSON list**.
"""


CORRECT_SYSTEM = """You are a strict JSON format corrector.

You will be given a string that should represent a JSON list, but it may contain multiple formatting errors.  
Check the error message carefully and fix the issues. After fixing the current error, **re-check for further errors** until the output is fully valid JSON.

**Task constraints:**
- Fix **only formatting errors** (commas, quotes, colons, brackets, extra/trailing chars, markdown fences, etc.).  
- Do not modify keys or values, only repair structure.
- Clean redundant list items if applicable.  
- Remove any URLs in the string.  
- For any list mixing strings and dicts, output as a JSON array of strings:  
    - Example: input `["Person", "Car":"moving"]` → output `["Person", "Car: moving"]`.  
    - Ensure no colon appears outside of quotes.  
- Replace all `"` with `'` inside string values.  
- Final output must be **valid JSON** and **nothing else** (no explanations).

**Output schema:**
```json
{
    ...
}
```

"""

if __name__ == "__main__":
    print(SUMMARIZE_WITH_LABEL_SYSTEM)