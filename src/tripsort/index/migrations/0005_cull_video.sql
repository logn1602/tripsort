-- tripsort index, schema version 5: quality scores and rejects, burst keepers, video frames,
-- Live Photos, and video dates.

-- media is rebuilt so its date source can be 'video' (a video container's creation time); a
-- CHECK constraint cannot change in place. Foreign keys are off while a migration runs, so
-- the tables that refer to media keep pointing at it, and nothing else changes.
CREATE TABLE media_v5 (
  id              INTEGER PRIMARY KEY,
  sha256          TEXT NOT NULL UNIQUE,
  kind            TEXT NOT NULL CHECK (kind IN ('image', 'video')),
  format          TEXT,              -- detected from the content: JPEG, PNG, WEBP, HEIF, MP4, MOV
  bytes           INTEGER NOT NULL,
  width           INTEGER,           -- full size, after EXIF orientation (videos: as displayed)
  height          INTEGER,
  duration_s      REAL,              -- videos, read with PyAV
  exif_datetime   TEXT,              -- EXIF capture time, ISO 8601, sub-seconds kept, no zone
  exif_offset     TEXT,              -- OffsetTimeOriginal, e.g. +02:00
  exif_modified   TEXT,              -- EXIF modify date (DateTime), to the second; editors change it
  has_exif        INTEGER NOT NULL DEFAULT 0,  -- capture time or camera make/model present
  camera_make     TEXT,
  camera_model    TEXT,
  lens_model      TEXT,
  exposure_time   REAL,              -- seconds; compared to tell burst shots from copies
  iso             INTEGER,
  gps_lat         REAL,
  gps_lon         REAL,
  phash           TEXT,              -- 64-bit pHash as 16 hex digits; NULL when not computed
  taken_at        TEXT,              -- resolved by analyze from EXIF, video data, names, zip times
  taken_at_source TEXT CHECK (
                    taken_at_source IN ('exif', 'video', 'filename', 'takeout_json', 'zip_mtime')),
  tz_offset       TEXT,
  status          TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'unreadable', 'skipped')),
  note            TEXT,              -- why the file is unreadable or was not fully analyzed
  is_front_camera INTEGER,
  shot            TEXT CHECK (shot IN ('solo', 'group', 'none')),
  is_selfie       INTEGER,
  clip_selfie     REAL,
  video_created   TEXT,              -- a video container's creation time, ISO 8601, no zone
  video_offset    TEXT               -- its offset from UTC, when the container records one
);
INSERT INTO media_v5 (
  id, sha256, kind, format, bytes, width, height, duration_s, exif_datetime, exif_offset,
  exif_modified, has_exif, camera_make, camera_model, lens_model, exposure_time, iso, gps_lat,
  gps_lon, phash, taken_at, taken_at_source, tz_offset, status, note, is_front_camera, shot,
  is_selfie, clip_selfie)
SELECT
  id, sha256, kind, format, bytes, width, height, duration_s, exif_datetime, exif_offset,
  exif_modified, has_exif, camera_make, camera_model, lens_model, exposure_time, iso, gps_lat,
  gps_lon, phash, taken_at, taken_at_source, tz_offset, status, note, is_front_camera, shot,
  is_selfie, clip_selfie
FROM media;
DROP TABLE media;
ALTER TABLE media_v5 RENAME TO media;
CREATE INDEX media_taken ON media(taken_at);

-- Sharpness and exposure from the one pixel pass, per photo. analyze sets reject from them.
CREATE TABLE quality (
  media_id    INTEGER PRIMARY KEY REFERENCES media(id),
  version     TEXT NOT NULL,         -- how the scores were made; a new one scores photos again
  sharpness   REAL NOT NULL,         -- variance of the Laplacian over a gray copy, 512 px long side
  sharp_tiles REAL NOT NULL,         -- the same over its sharpest tiles
  luma_mean   REAL NOT NULL,         -- mean brightness, 0 to 255
  dark_frac   REAL NOT NULL,         -- share of pixels at or below 16
  bright_frac REAL NOT NULL,         -- share of pixels at or above 240
  reject      TEXT CHECK (reject IN ('blurry', 'dark', 'blown_out'))
);

-- Which videos had frames sampled for faces and CLIP, under which models and sampling (also
-- when that failed, or the video is a Live Photo's motion clip, which is never sampled).
CREATE TABLE video_scans (
  media_id INTEGER NOT NULL REFERENCES media(id),
  scan_id  TEXT NOT NULL,
  status   TEXT NOT NULL CHECK (status IN ('ok', 'live_photo', 'not_decoded', 'failed')),
  frames   INTEGER NOT NULL DEFAULT 0,
  detail   TEXT,
  PRIMARY KEY (media_id, scan_id)
);

-- A Live Photo's motion clip and its still, paired by analyze.
CREATE TABLE live_photos (
  motion_id INTEGER PRIMARY KEY REFERENCES media(id),
  still_id  INTEGER NOT NULL REFERENCES media(id)
);

-- exported_files gains the rejects/ and bursts/ roles; nothing refers to it.
CREATE TABLE exported_files_v5 (
  relpath  TEXT PRIMARY KEY,
  fold_key TEXT NOT NULL UNIQUE,
  media_id INTEGER NOT NULL REFERENCES media(id),
  role     TEXT NOT NULL CHECK (
             role IN ('all', 'unreadable', 'dup_keeper', 'dup_copy', 'person', 'shot',
                      'category', 'reject', 'burst')),
  method   TEXT NOT NULL CHECK (method IN ('extract', 'hardlink', 'copy')),
  bytes    INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL
);
INSERT INTO exported_files_v5 (relpath, fold_key, media_id, role, method, bytes, mtime_ns)
  SELECT relpath, fold_key, media_id, role, method, bytes, mtime_ns FROM exported_files;
DROP TABLE exported_files;
ALTER TABLE exported_files_v5 RENAME TO exported_files;
CREATE INDEX exported_media ON exported_files(media_id);

CREATE INDEX video_scans_scan ON video_scans(scan_id);
CREATE INDEX live_photos_still ON live_photos(still_id);
