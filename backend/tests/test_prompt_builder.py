from app.services.prompt_builder import PromptBuilder
from app.models.schemas import PromptBuilderOptions

def test_prompt_builder_presets():
    opts = PromptBuilderOptions(
        scene_description="grand theater stage with classical Greek columns and red curtains",
        materials=["aged_oak", "polished_marble"],
        lighting="dramatic_stage",
        render_style="theatrical_set"
    )
    prompt, negative = PromptBuilder.build_prompt(opts)
    
    assert "photorealistic" in prompt.lower()
    assert "greek columns" in prompt.lower()
    assert "oak wood" in prompt.lower()
    assert "marble" in prompt.lower()
    assert "spotlighting" in prompt.lower()
    assert "theatrical set" in prompt.lower()
    assert "blurry" in negative.lower()

def test_prompt_builder_custom():
    opts = PromptBuilderOptions(
        scene_description="modern minimalist brutalist stage",
        materials=["raw concrete"],
        lighting="diffuse skylight",
        render_style="architectural photograph",
        custom_modifiers="award winning, 8k resolution"
    )
    prompt, negative = PromptBuilder.build_prompt(opts)
    assert "brutalist stage" in prompt
    assert "raw concrete" in prompt
    assert "award winning" in prompt
