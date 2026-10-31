"""Promo Agent.

Generates a 20-second promotional video storyboard consisting of a 45-50 word voiceover
and 4 synchronized 5-second scene prompts tailored for Vertex AI Veo video generation.
"""

import json
from pathlib import Path
from typing import Any, Optional

from config import MODELS, OUTPUTS_DIR, PROMO_CONFIG, get_run_dir
from utils.llm import call_gemini

PROMO_SYSTEM_PROMPT = """You are an elite creative video director and commercial producer.
Your task is to design a high-converting 20-second promotional storyboard and voiceover based on the provided project idea, script, and tone.

SPECIFICATIONS:
1. 'voiceover_text':
   - Exactly 45 to 50 words total.
   - Paced to be clearly spoken aloud in 20 seconds (~2.3 words per second).
   - Dynamic, engaging, and faithful to the target tone ('serious' vs 'humorous').

2. 'scenes':
   - Exactly 4 scenes (Scene 1 to Scene 4), each representing 5 seconds of footage.
   - Scene 1: An irresistible visual hook capturing the core pain point or intrigue.
   - Scene 2: The technology or magic transformation in action.
   - Scene 3: The triumphant user experience and high-value outcome.
   - Scene 4: An inspiring visual climax ending with a clear Call to Action.

CRITICAL VISUAL PROMPT RULES FOR VEO:
- Describe ONE clear continuous shot per scene (e.g. 'Slow camera push-in on...', 'Low angle tracking shot moving with...').
- Define lighting, color palette, and textures (e.g. 'Cinematic warm amber lighting, shallow depth of field, photorealistic 8k render').
- MAINTAIN THE EXACT SAME VISUAL STYLE AND CHARACTERS ACROSS ALL 4 SCENES for seamless narrative continuity.
- ABSOLUTELY NO TEXT INSIDE THE VIDEO: Never describe text, letters, titles, logos, subtitles, or UI buttons appearing in the visual scene itself (Veo cannot render text and produces artifacts).
- 'caption': Short on-screen subtitle string (4-7 words) for overlay captions.

You must respond ONLY with a valid JSON object matching this schema:
{
  "voiceover_text": "Spoken voiceover text between 45 and 50 words...",
  "scenes": [
    {
      "scene_no": 1,
      "visual_prompt": "Cinematic slow camera tracking shot pushing in on an exhausted college student surrounded by towering stacks of handwritten notebooks under cozy midnight desk lamp glow, shallow depth of field, realistic textures, no text.",
      "caption": "Drowning in endless college lecture notes?"
    },
    {
      "scene_no": 2,
      "visual_prompt": "Smooth lateral camera pan as the student points a smartphone camera toward the pages, soft glowing holographic particles rising from the paper, warm cinematic lighting, consistent character and room, no text.",
      "caption": "Instant AI extraction scans every page."
    },
    {
      "scene_no": 3,
      "visual_prompt": "Medium close-up shot orbiting the student now laughing and tapping interactive flashcards on a tablet with friends in a sunlit modern campus cafe, vibrant colors, same student, no text.",
      "caption": "Master exams with personalized active recall."
    },
    {
      "scene_no": 4,
      "visual_prompt": "Dynamic upward tracking shot of the confident student stepping out into campus daylight, holding up an A-plus exam paper with a triumph smile, golden hour backlight, cinematic lens flare, no text.",
      "caption": "Turn your notes into top grades today."
    }
  ]
}
"""


def build_storyboard(
    idea: str,
    script: str,
    tone: str = "serious",
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a 4-scene, 20-second promo storyboard and voiceover script.

    Args:
        idea: Project idea description.
        script: Full written script.
        tone: Target tone ('serious' or 'humorous').
        run_id: Optional run ID to save output in outputs/<run_id>/storyboard.json.

    Returns:
        Dictionary containing:
        - 'voiceover_text': str (45-50 words)
        - 'scenes': list of 4 scene dicts with 'scene_no', 'visual_prompt', and 'caption'
    """
    user_prompt = f"""Project Idea: {idea}
Target Tone: {tone.upper()}

Core Script Context:
\"\"\"
{script[:1200]}
\"\"\"

Create the 20-second promo storyboard and 45-50 word voiceover.
Remember:
- Exactly 4 scenes (5 seconds each)
- Visual consistency across all 4 scenes
- No text described inside visual prompts
- Scene 1 is a hook, Scene 4 ends with a call to action
"""

    response = call_gemini(
        prompt=user_prompt,
        system=PROMO_SYSTEM_PROMPT,
        json_mode=True,
        model=MODELS.promo,
    )

    voiceover_text = ""
    raw_scenes = []

    if isinstance(response, dict):
        voiceover_text = str(response.get("voiceover_text", "")).strip()
        raw_scenes = response.get("scenes", [])
    elif isinstance(response, list):
        raw_scenes = response

    # Validate and normalize scenes to exactly 4 items
    validated_scenes: list[dict[str, Any]] = []
    for idx in range(1, PROMO_CONFIG.total_scenes + 1):
        scene_item = None
        for s in raw_scenes:
            if isinstance(s, dict) and s.get("scene_no") == idx:
                scene_item = s
                break

        if not scene_item and idx - 1 < len(raw_scenes) and isinstance(raw_scenes[idx - 1], dict):
            scene_item = raw_scenes[idx - 1]

        if scene_item:
            v_prompt = str(scene_item.get("visual_prompt", "")).strip()
            caption = str(scene_item.get("caption", "")).strip()
        else:
            v_prompt = f"Cinematic continuous camera shot for scene {idx} representing {idea}, photorealistic lighting, no text."
            caption = f"Scene {idx} highlight"

        validated_scenes.append({
            "scene_no": idx,
            "visual_prompt": v_prompt,
            "caption": caption,
        })

    # Word count check on voiceover
    word_count = len(voiceover_text.split()) if voiceover_text else 0
    if not voiceover_text or word_count < 25:
        # Fallback voiceover if missing
        voiceover_text = (
            f"Struggling to turn endless study notes into real knowledge? "
            f"Discover the smart new way to transform raw lecture pages into interactive, viral quizzes in seconds. "
            f"Retain more, ace your exams, and make studying engaging. "
            f"Try the quiz revolution today and unlock your potential."
        )

    storyboard = {
        "voiceover_text": voiceover_text,
        "scenes": validated_scenes,
    }

    # Save output if run_id is provided
    if run_id:
        target_dir = get_run_dir(run_id)
    else:
        target_dir = OUTPUTS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

    storyboard_file = target_dir / "storyboard.json"
    with open(storyboard_file, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2, ensure_ascii=False)
    print(f"Saved 20-second storyboard ({len(validated_scenes)} scenes) to: {storyboard_file}")

    return storyboard
