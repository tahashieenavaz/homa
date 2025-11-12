import os
import sys
import ffmpeg
from .Namespace import Namespace


class MediaNamespace(Namespace):
    def _validate_input(self, input_filename: str) -> bool:
        if not os.path.exists(input_filename):
            print(f"Error: Input file not found at {input_filename}", file=sys.stderr)
            return False
        return True

    def _get_output_filename(self, input_filename: str, extension: str) -> str:
        base, _ = os.path.splitext(input_filename)
        return f"{base}.{extension}"

    def _run_ffmpeg(self, input_filename: str, output_filename: str, **output_kwargs):
        try:
            (
                ffmpeg.input(input_filename)
                .output(output_filename, **output_kwargs)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            print(f"✅ Successfully converted and saved to {output_filename}")
        except ffmpeg.Error as e:
            print("FFmpeg Error:", file=sys.stderr)
            print(e.stderr.decode(), file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)

    def _convert_to_webm(self, input_filename: str):
        if not self._validate_input(input_filename):
            return
        output_filename = self._get_output_filename(input_filename, "webm")
        self._run_ffmpeg(
            input_filename,
            output_filename,
            vcodec="libvpx-vp9",
            crf=32,
            acodec="libopus",
            audio_bitrate="96k",
        )

    def _convert_to_gif(self, input_filename: str):
        if not self._validate_input(input_filename):
            return
        output_filename = self._get_output_filename(input_filename, "gif")

        fps = 10
        scale_width = 400
        max_colors = 64
        dither_mode = "bayer"

        complex_filter = (
            f"fps={fps},scale={scale_width}:-1:flags=lanczos,split [a][b]; "
            f"[a] palettegen=stats_mode=diff:max_colors={max_colors} [p]; "
            f"[b][p] paletteuse=dither={dither_mode}:diff_mode=rectangle"
        )

        self._run_ffmpeg(
            input_filename,
            output_filename,
            filter_complex=complex_filter,
            loop=0,
        )

    def convert(self, target_format: str, *input_files):
        convert_map = {
            "web": self._convert_to_webm,
            "webm": self._convert_to_webm,
            "gif": self._convert_to_gif,
        }

        handler = convert_map.get(target_format.lower())
        if not handler:
            print(
                f"Error: Unsupported target format '{target_format}'", file=sys.stderr
            )
            return

        for input_file in input_files:
            handler(input_file)
