from typing import Final, get_type_hints, final

from timm.models.vision_transformer import VisionTransformer

from sharp.models.presets import ViTConfig

@final
class TestViTConfig:

    @staticmethod
    def test_constructor_annotations() -> None:
        """The `global_pool` parameters shall match the one in
        timm.models.vision_transformer.VisionTransformer.__init__.
        """

        expected: Final = get_type_hints(VisionTransformer.__init__)['global_pool']
        actual: Final = get_type_hints(ViTConfig)['global_pool']

        assert expected == actual
