import os
from pathlib import Path

from blind_watermark import WaterMark
from dotenv import load_dotenv
load_dotenv()


class ImageWaterMark:
    IMAGE_EXTENTIONS = {'.jpg', '.jpeg', '.png'}

    def __init__(self, wm_content: str, wm_password: int) -> None:
        self._wm_content = wm_content
        self._bwm1 = WaterMark(password_img=1, password_wm=wm_password)
        self._bwm1.read_wm(self._wm_content, mode='str')
        self._len_wm = len(self._bwm1.wm_bit)  # type: ignore

    def extract_in_dir(self, dir: Path):
        dir = dir.absolute()
        if not dir.exists():
            print(f"Target directory does not exist: {dir}")
            return
        if not dir.is_dir():
            print(f"Target path is not a directory: {dir}")
            return
        for file in dir.rglob('*'):
            if file.is_file() and file.suffix.lower() in self.IMAGE_EXTENTIONS:
                self.recognize_image_watermark(file)

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
            if file.is_file() and file.suffix.lower() in self.IMAGE_EXTENTIONS:
                if '.back.' in file.name:
                    continue
                bk_fn = file.with_name(file.stem + '.back' + file.suffix)
                if bk_fn.exists():
                    continue
                to_fn = file.absolute()
                bk_fn = file.rename(bk_fn)
                backup_files.append(bk_fn)
                print(f"Renamed: {file.name} -> {bk_fn.name}")
                self.embed_image_watermark(bk_fn, to_fn, self._wm_content)
                self.recognize_image_watermark(to_fn)

        print(
            f"Finished. Remove {len(backup_files)} .back files manually, please.")

    def recognize_image_watermark(self, img: Path) -> bool:
        assert img.exists(), f"Image does not exist: {img}"
        try:
            wm_extract = self._bwm1.extract(
                img, wm_shape=self._len_wm, mode='str')
        except Exception as e:
            print(e)
            return False
        print(f"Watermark {img.name} -> {wm_extract}")
        if wm_extract == self._wm_content:
            return True
        else:
            count = 0
            for char in self._wm_content.split():
                if char in wm_extract:
                    count += 1
            return count / len(self._wm_content) > 0.8

    def embed_image_watermark(self, input_img: Path, output_img: Path, content: str):
        assert input_img.exists() and (not output_img.exists())

        self._bwm1.read_img(str(input_img.absolute()))
        self._bwm1.read_wm(content, mode='str')
        self._bwm1.embed(str(output_img.absolute()))
        len_wm = len(self._bwm1.wm_bit)  # type: ignore
        print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))


if __name__ == "__main__":
    wm_content: str = str(os.environ['IMG_WM_CONTENT'])
    wm_pwd: int = int(os.environ['IMG_WM_PASSWORD'])
    iwm = ImageWaterMark(wm_content, wm_pwd)
    arts_dir = Path("./public/arts_wm")
    iwm.watermark_in_dir(arts_dir)
    iwm.extract_in_dir(Path("./public/arts_wm"))
