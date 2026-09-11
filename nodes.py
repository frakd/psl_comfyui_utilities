from comfy_api.latest import io

from .dsp import suppress_harshness


class AudioHarshnessSuppressor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioHarshnessSuppressor",
            search_aliases=["de-harsh", "de-esser", "sibilance", "刺耳", "齿音"],
            display_name="Audio Harshness Suppressor",
            category="audio/filter",
            description=(
                "Attenuates a frequency band only in the moments where it sounds harsh, "
                "so bright material gets smoother without the whole track getting duller."
            ),
            inputs=[
                io.Audio.Input("audio"),
                io.Float.Input(
                    "harsh_low_hz",
                    default=2500.0,
                    min=200.0,
                    max=20000.0,
                    step=50.0,
                    tooltip="Bottom of the band to watch. 2500 Hz covers presence and sibilance.",
                ),
                io.Float.Input(
                    "harsh_high_hz",
                    default=9000.0,
                    min=200.0,
                    max=20000.0,
                    step=50.0,
                    tooltip="Top of the band to watch. Narrow it to 5000-9000 Hz to target sibilance only.",
                ),
                io.Float.Input(
                    "sensitivity_db",
                    default=-6.0,
                    min=-30.0,
                    max=12.0,
                    step=0.5,
                    tooltip=(
                        "How much louder than the rest of the spectrum the band has to be before it is "
                        "reduced, in dB. A spectrally flat signal measures 0 dB. Lower is more aggressive."
                    ),
                ),
                io.Float.Input(
                    "max_reduction_db",
                    default=9.0,
                    min=0.0,
                    max=30.0,
                    step=0.5,
                    tooltip="Deepest cut applied at the harshest moment. 0 disables processing.",
                ),
                io.Float.Input(
                    "attack_ms",
                    default=5.0,
                    min=0.0,
                    max=200.0,
                    step=0.5,
                    tooltip="How quickly the cut fades in. Keep it short so sibilants are caught at their onset.",
                ),
                io.Float.Input(
                    "release_ms",
                    default=120.0,
                    min=0.0,
                    max=2000.0,
                    step=5.0,
                    tooltip="How quickly the cut fades back out. Too short makes the reduction audible as pumping.",
                ),
                io.Float.Input(
                    "mix",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Blend between the untouched and the processed audio.",
                ),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(
        cls,
        audio,
        harsh_low_hz,
        harsh_high_hz,
        sensitivity_db,
        max_reduction_db,
        attack_ms,
        release_ms,
        mix,
    ) -> io.NodeOutput:
        if audio is None:
            raise ValueError("AudioHarshnessSuppressor: input audio is None (source video may have no audio track).")
        waveform = suppress_harshness(
            audio["waveform"],
            audio["sample_rate"],
            harsh_low_hz=harsh_low_hz,
            harsh_high_hz=harsh_high_hz,
            sensitivity_db=sensitivity_db,
            max_reduction_db=max_reduction_db,
            attack_ms=attack_ms,
            release_ms=release_ms,
            mix=mix,
        )
        return io.NodeOutput({"waveform": waveform, "sample_rate": audio["sample_rate"]})
