from comfy_api.latest import io

from .dehum import suppress_harmonics
from .dsp import suppress_harshness
from .gated_deperc import reference_gated_deperc


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
                    tooltip="Top of the band to watch. Lower it if the top end is not what bothers you.",
                ),
                io.Float.Input(
                    "sensitivity_db",
                    default=-6.0,
                    min=-30.0,
                    max=12.0,
                    step=0.5,
                    tooltip=(
                        "How much denser than the rest of the spectrum the band has to be before it is "
                        "reduced, in dB. A spectrally flat signal measures 0 dB. Lower is more aggressive "
                        "and starts catching ordinary tonal content, not just bright noisy moments."
                    ),
                ),
                io.Float.Input(
                    "max_reduction_db",
                    default=18.0,
                    min=0.0,
                    max=40.0,
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


class AudioHarmonicNotch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioHarmonicNotch",
            search_aliases=["de-hum", "hum", "buzz", "notch", "去嗡", "哼声", "哒哒", "电流声", "谐波"],
            display_name="Audio Harmonic Notch (De-Hum)",
            category="audio/filter",
            description=(
                "Removes a tonal fundamental and its integer harmonics across the whole track "
                "with a bank of narrow notches. Targets periodic low buzz / 哒哒 / mains hum, "
                "leaving the material between the notches untouched."
            ),
            inputs=[
                io.Audio.Input("audio"),
                io.Float.Input(
                    "fundamental_hz",
                    default=50.0,
                    min=1.0,
                    max=2000.0,
                    step=0.1,
                    tooltip=(
                        "Pitch of the hum/buzz. Mains hum is 50 Hz (EU) or 60 Hz (US). For a periodic "
                        "哒哒, read its lowest spectral line. The notches are placed at this frequency "
                        "and every integer multiple of it."
                    ),
                ),
                io.Int.Input(
                    "harmonics",
                    default=8,
                    min=1,
                    max=64,
                    tooltip="How many multiples of the fundamental to notch, including the fundamental itself.",
                ),
                io.Float.Input(
                    "q",
                    default=15.0,
                    min=0.5,
                    max=100.0,
                    step=0.5,
                    tooltip=(
                        "Notch sharpness. Bandwidth = frequency / q, so higher q is narrower and spares "
                        "more of the wanted audio around each harmonic. Lower it if the buzz drifts in pitch."
                    ),
                ),
                io.Float.Input(
                    "reduction_db",
                    default=30.0,
                    min=0.0,
                    max=60.0,
                    step=0.5,
                    tooltip="Effective notch depth. 30 dB leaves ~3% of the tone, 12 dB leaves ~25%. 0 disables processing.",
                ),
                io.Float.Input(
                    "mix",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Blend between the untouched and the notched audio, for A/B comparison.",
                ),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(
        cls,
        audio,
        fundamental_hz,
        harmonics,
        q,
        reduction_db,
        mix,
    ) -> io.NodeOutput:
        if audio is None:
            raise ValueError("AudioHarmonicNotch: input audio is None (source video may have no audio track).")
        waveform = suppress_harmonics(
            audio["waveform"],
            audio["sample_rate"],
            fundamental_hz=fundamental_hz,
            harmonics=harmonics,
            q=q,
            reduction_db=reduction_db,
            mix=mix,
        )
        return io.NodeOutput({"waveform": waveform, "sample_rate": audio["sample_rate"]})


class AudioReferenceDePercuss(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioReferenceDePercuss",
            search_aliases=["de-click", "de-percuss", "哒哒", "参考", "reference", "区间", "gated", "去打击"],
            display_name="Audio De-Percuss (Reference-Gated)",
            category="audio/filter",
            description=(
                "Removes a percussive / 哒哒 sound only in the regions that match a reference clip you "
                "supply. Detects matching regions by spectral similarity, applies HPSS percussive "
                "attenuation inside them with a crossfade, and passes everything else through unchanged."
            ),
            inputs=[
                io.Audio.Input("audio"),
                io.Audio.Input(
                    "reference",
                    tooltip="A short clip containing the target sound (e.g. a second or two of the 哒哒). Its spectral shape is the match template.",
                ),
                io.Float.Input(
                    "match_threshold",
                    default=0.93,
                    min=0.5,
                    max=0.999,
                    step=0.01,
                    tooltip=(
                        "Cosine similarity a frame must reach to count as the target sound. Higher is "
                        "stricter (fewer false regions); lower catches more but may grab similar material."
                    ),
                ),
                io.Float.Input(
                    "min_region_sec",
                    default=0.5,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="Drop detected regions shorter than this. Removes brief false matches; keep above the gap between real pulses.",
                ),
                io.Float.Input(
                    "merge_gap_sec",
                    default=0.3,
                    min=0.0,
                    max=5.0,
                    step=0.05,
                    tooltip="Bridge silent gaps up to this length so one continuous passage stays a single region.",
                ),
                io.Float.Input(
                    "crossfade_ms",
                    default=50.0,
                    min=0.0,
                    max=500.0,
                    step=5.0,
                    tooltip="Fade at region boundaries to avoid clicks. Only this margin around a region is touched besides the region itself.",
                ),
                io.Float.Input(
                    "reduction_db",
                    default=18.0,
                    min=0.0,
                    max=40.0,
                    step=0.5,
                    tooltip="Percussive attenuation depth inside the regions. 0 disables processing.",
                ),
                io.Float.Input(
                    "mix",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Blend between the untouched and the processed audio, for A/B comparison.",
                ),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(
        cls,
        audio,
        reference,
        match_threshold,
        min_region_sec,
        merge_gap_sec,
        crossfade_ms,
        reduction_db,
        mix,
    ) -> io.NodeOutput:
        if audio is None:
            raise ValueError("AudioReferenceDePercuss: input audio is None (source video may have no audio track).")
        if reference is None:
            raise ValueError("AudioReferenceDePercuss: reference audio is None; feed a clip of the target sound.")
        waveform = reference_gated_deperc(
            audio["waveform"],
            audio["sample_rate"],
            reference["waveform"],
            reference["sample_rate"],
            match_threshold=match_threshold,
            min_region_sec=min_region_sec,
            merge_gap_sec=merge_gap_sec,
            crossfade_ms=crossfade_ms,
            reduction_db=reduction_db,
            mix=mix,
        )
        return io.NodeOutput({"waveform": waveform, "sample_rate": audio["sample_rate"]})
