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

## categories/

Photos and scans with no people in them, in one folder per category, for sanity-checking how
CLIP sorts photos into categories. Each file starts from the rendition Wikimedia Commons serves
for a 1200 px wide request (1280 px wide, or the original file when that is narrower),
downscaled to 640 px on its long side (Pillow, Lanczos), converted to RGB (mapped to sRGB
first when the rendition embeds a color profile), and saved as a JPEG at quality 85 with no
metadata (EXIF, ICC profile, comments).

| File | What it shows | Source | License |
| --- | --- | --- | --- |
| `scenery/beach_driftwood.jpg` | Driftwood on an empty beach, Corcovado National Park, Costa Rica | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Corcovado_National_Park,_Costa_RIca_02.jpg) | CC0 |
| `scenery/mountain_river.jpg` | A forested mountain above a rocky river, Jacques-Cartier National Park, Quebec | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Parc_national_de_la_Jacques-Cartier,_Quebec,_Canada_22.jpg) | CC0 |
| `scenery/city_skyline.jpg` | The Frankfurt am Main skyline on a clear day | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Skyline_Frankfurt_am_Main_2019.jpg) | CC0 |
| `scenery/sunset_boats.jpg` | Sunset over a bay with moored fishing boats, Margarita Island, Venezuela | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Juan_Griego_Sunset.jpg) | CC0 |
| `scenery/sunset_tree.jpg` | A lone tree against a sunset over wooded hills, Charlton, Massachusetts | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sunset_from_Dresser_Hill,_Charlton_MA_2026-05-10.jpg) | CC0 |
| `animals/dog_on_pier.jpg` | A dog resting on a concrete pier by the sea | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Dog_in_Chacachacare.jpg) | CC0 |
| `animals/cat_on_bench.jpg` | A tabby cat asleep on a park bench | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Chat_domestique_dormant_sur_un_banc.jpg) | CC0 |
| `animals/kingfisher.jpg` | A common kingfisher perched on a branch | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Eisvogel_kingfisher.jpg) | CC0 |
| `animals/lion.jpg` | A male lion walking along a dirt track, Ngorongoro Conservation Area, Tanzania | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Lion_Ngorongoro_2.jpg) | CC0 |
| `animals/flamingo.jpg` | An American flamingo at the São Paulo Zoo, Brazil | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Phoenicopterus_ruber_in_S%C3%A3o_Paulo_Zoo.jpg) | CC0 |
| `food/curry_rice.jpg` | Japanese beef curry with rice on a plate | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Beef_curry_rice_003.jpg) | CC0 |
| `food/prawn_risotto.jpg` | Prawn risotto served in a restaurant | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Risotto_de_gambas,_restaurant_Danieli_(Vienne,_Autriche).jpg) | CC0 |
| `food/oysters_lemon.jpg` | A plate of oysters beside a plate of sliced lemon | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Plate_of_oysters_with_lemon.jpg) | CC0 |
| `food/dishes_on_table.jpg` | Bowls and plates of vegetables, fruit, bread, and grains on a table | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Good_Food_In_Dishes_-_NCI_Visuals_Online.jpg) | Public domain |
| `food/produce_display.jpg` | Vegetables, fruit, fish, meat, bread, and grains spread out on a table | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Good_Food_Display_-_NCI_Visuals_Online.jpg) | Public domain |
| `documents/city_map_1855.jpg` | Printed plan of the port and town of Brest, France, 1855, with its legend | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Btv1b530849639_plan_du_port_et_de_la_ville_de_Brest_1855_restored.jpg) | Public domain |
| `documents/train_ticket.jpg` | A British Rail cheap day return ticket, Peterborough East to Wakerley and Barrowden | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Railway_ticket_from_Wakerley_station.JPG) | Public domain |
| `documents/theatre_ticket.jpg` | A free admission ticket to the Coliseum theatre, Seattle, 1911 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Coliseum_Theatre_ticket,_1911_(MOHAI_11322).jpg) | Public domain |

Every file is public domain or CC0 according to its Commons page, as reported by the Commons
API (`LicenseShortName`). The licenses were checked on 2026-11-21.
