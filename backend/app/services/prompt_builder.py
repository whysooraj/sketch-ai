from typing import List, Optional
from app.models.schemas import PromptBuilderOptions

class PromptBuilder:
    DEFAULT_NEGATIVE_PROMPT = (
        "low quality, blurry, worst quality, cartoon, anime, 3d render, plastic, drawing, "
        "sketch, pencil, ink, crosshatch, distorted perspective, extra limbs, bad proportions, "
        "overexposed, underexposed, watermark, text, signature"
    )

    MATERIAL_PRESETS = {
        "aged_oak": "aged dark oak wood timber with visible woodgrain texture and authentic wear",
        "polished_marble": "polished Carrara white marble with subtle grey veining and specular reflections",
        "weathered_concrete": "weathered industrial concrete walls with realistic micro-cracks and surface pitting",
        "brushed_steel": "matte brushed stainless steel fixtures with anisotropic metallic highlights",
        "brickwork": "reclaimed exposed red brick masonry with authentic mortar relief and weathering",
        "theatrical_velvet": "heavy deep crimson theatrical velvet drapery with soft volumetric fabric folds",
        "distressed_leather": "vintage distressed cognac leather upholstery with natural patina",
        "glass_metal": "clean architectural plate glass and dark anodized aluminum mullions"
    }

    LIGHTING_PRESETS = {
        "dramatic_stage": "dramatic theatrical stage spotlighting with strong volumetric light beams, sharp shadows, and high contrast chiaroscuro",
        "golden_hour": "warm low-angle golden hour sunlight streaming through openings, casting long soft amber shadows",
        "overcast_diffuse": "soft diffuse natural daylight, even neutral illumination, realistic ambient occlusion in corners",
        "moody_chiaroscuro": "dark moody atmospheric interior lighting, warm tungsten practical lamps, deep shadows, cinematic atmosphere",
        "clean_studio": "professional three-point architectural studio lighting, balanced key and fill, pristine clarity",
        "night_noir": "nocturnal scene with cold moonlight and warm incandescent spill, film noir contrast"
    }

    STYLE_PRESETS = {
        "architectural_photo": "hyperrealistic architectural photograph, captured on Hasselblad H6D-100c, 35mm lens, f/8 aperture, razor-sharp textures",
        "cinematic_still": "cinematic film still from high-budget cinema, 35mm Panavision anamorphic lens, shallow depth of field, authentic film grain",
        "theatrical_set": "authentic full-scale theatrical set design photograph, real physical materials, authentic stage depth and perspective",
        "hyperrealistic_interior": "photorealistic interior design photograph, highly detailed spatial lighting, true-to-life tactile materials, 8k resolution"
    }

    @classmethod
    def build_prompt(cls, options: PromptBuilderOptions) -> tuple[str, str]:
        parts = []

        # 1. Base style definition
        style_desc = cls.STYLE_PRESETS.get(options.render_style, options.render_style)
        parts.append(style_desc)

        # 2. Main scene subject
        clean_scene = options.scene_description.strip()
        parts.append(f"A photorealistic depiction of {clean_scene}")

        # 3. Materials
        if options.materials:
            resolved_materials = []
            for mat in options.materials:
                resolved_materials.append(cls.MATERIAL_PRESETS.get(mat, mat))
            parts.append(f"featuring authentic materials: {', '.join(resolved_materials)}")

        # 4. Lighting and ambiance
        lighting_desc = cls.LIGHTING_PRESETS.get(options.lighting, options.lighting)
        parts.append(f"illuminated by {lighting_desc}")

        # 5. Technical realism modifiers
        parts.append("physically based rendering, authentic micro-textures, photorealistic shadows, volumetric depth, masterpiece")

        # 6. Custom modifiers if any
        if options.custom_modifiers:
            parts.append(options.custom_modifiers.strip())

        full_prompt = ", ".join(parts)
        negative = options.negative_prompt if options.negative_prompt else cls.DEFAULT_NEGATIVE_PROMPT

        return full_prompt, negative
