import os
import sys
import ffmpeg
from .Namespace import Namespace


class MediaNamespace(Namespace):
    def _change_extension(self, filename: str, extension: str) -> str:
        broken = filename.split(".")
        broken[-1] = extension
        return ".".join(broken)

    def _convert_to_webm(self, input_filename: str):
        if not os.path.exists(input_filename):
            print(f"Error: Input file not found at {input_filename}", file=sys.stderr)
            return

        output_filename = self._change_extension(input_filename)
        try:
            ffmpeg.input(input_filename).output(
                output_filename, vcodec="libvpx", acodec="libopus"
            ).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            print(f"Successfully converted file and saved to {output_filename}")
        except ffmpeg.Error as e:
            print("FFmpeg Error:", file=sys.stderr)
            print(e.stderr.decode(), file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)

    def convert(self, target_format: str, input_file: str):
        convert_map = {
            "web": self._convert_to_webm,
            "webm": self._convert_to_webm,
        }
        handler = convert_map.get(target_format)
        handler(input_file)
