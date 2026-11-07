import os
from pathlib import Path

import librosa
import perth
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()


class AudioWaterMark:
    AUDIO_EXTENTIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    def __init__(self, wm_content: str) -> None:
        self._wm_content = wm_content
        self._watermarker = perth.PerthImplicitWatermarker()

    def extract_in_dir(self, dir: Path):
        dir = dir.absolute()
        if not dir.exists():
            print(f"Target directory does not exist: {dir}")
            return
        if not dir.is_dir():
            print(f"Target path is not a directory: {dir}")
            return
        for file in dir.rglob('*'):
            if file.is_file() and file.suffix.lower() in self.AUDIO_EXTENTIONS:
                self.recognize_audio_watermark(file)

    def watermark_in_dir(self, dir: Path):
        dir = dir.absolute()
        if not dir.exists():
            print(f"Target directory does not exist: {dir}")
            return
        if not dir.is_dir():
            print(f"Target path is not a directory: {dir}")
            return
        backup_files: list[Path] = []
        for file in dir.rglob('*'):
            if file.is_file() and file.suffix.lower() in self.AUDIO_EXTENTIONS:
                if '.back.' in file.name:
                    continue
                bk_fn = file.with_name(file.stem + '.back' + file.suffix)
                if bk_fn.exists():
                    continue
                to_fn = file.absolute()
                bk_fn = file.rename(bk_fn)
                backup_files.append(bk_fn)
                print(f"Renamed: {file.name} -> {bk_fn.name}")
                self.embed_audio_watermark(bk_fn, to_fn, self._wm_content)
                self.recognize_audio_watermark(to_fn)

        print(
            f"Finished. Remove {len(backup_files)} .back files manually, please.")

    def recognize_audio_watermark(self, audio: Path) -> bool:
        assert audio.exists(), f"Audio does not exist: {audio}"
        try:
            wav, sr = librosa.load(str(audio.absolute()), sr=None)
            wm_extract = self._watermarker.get_watermark(wav, sample_rate=sr)
        except Exception as e:
            print(e)
            return False
        print(f"Watermark {audio.name} -> {wm_extract}")
        if not wm_extract:
            return False
        if wm_extract == self._wm_content:
            return True
        else:
            count = 0
            for word in self._wm_content.split():
                if word in wm_extract:
                    count += 1
            return count / len(self._wm_content) > 0.8

    def embed_audio_watermark(self, input_audio: Path, output_audio: Path, content: str):
        assert input_audio.exists() and (not output_audio.exists())

        wav, sr = librosa.load(str(input_audio.absolute()), sr=None)
        watermarked_audio = self._watermarker.apply_watermark(
            wav, watermark=content, sample_rate=sr)
        sf.write(str(output_audio.absolute()),
                 watermarked_audio, sr)  # type: ignore
        print(f"Embedded watermark into {output_audio.name}")


if __name__ == "__main__":
    wm_content: str = str(os.environ['AUDIO_WM_CONTENT'])
    awm = AudioWaterMark(wm_content)
    music_dir = Path("./public/music_wm")
    awm.watermark_in_dir(music_dir)
    awm.extract_in_dir(Path("./public/music_wm"))
