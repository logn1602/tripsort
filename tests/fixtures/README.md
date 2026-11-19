# Test fixtures

No photos of real people are used in the tests. Most test images are generated at test time
(`tests/factories.py`, `tests/fake_faces.py`). The face-model tests use three public-domain
portrait paintings instead.

## paintings/

Each file is the 1200 px rendition Wikimedia Commons serves for the work, downscaled to
640 px on its long side (Pillow, Lanczos), saved as a JPEG at quality 85 with no metadata
(EXIF, ICC profile, comments).

| File | Work | Source | License |
| --- | --- | --- | --- |
| `mona_lisa.jpg` | Leonardo da Vinci, *Mona Lisa*, between 1503 and 1506 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg) | Public domain |
| `girl_with_a_pearl_earring.jpg` | Johannes Vermeer, *Girl with a Pearl Earring*, 1665 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:1665_Girl_with_a_Pearl_Earring.jpg) | Public domain |
| `george_washington.jpg` | Gilbert Stuart, *George Washington* (Williamstown portrait), 1803 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gilbert_Stuart_Williamstown_Portrait_of_George_Washington.jpg) | Public domain |

The paintings are public domain because of their age, and faithful photographic
reproductions of two-dimensional public-domain works are public domain too (Commons tags
them PD-Art). They were checked on 2026-11-19.

What the tests check with them: YuNet finds exactly one clear face in each painting;
SFace embeds flipped, cropped, darkened, brightened, and shrunk copies of a painting close to
the original and far from the other paintings; and `tripsort run` sorts such copies into one
people folder per painting and names the folders from a refs folder.
