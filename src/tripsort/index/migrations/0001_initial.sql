-- tripsort index, schema version 1: ingest, dedupe, and export.
-- The migration runner creates and updates the schema_version table itself.
-- Later phases add their own tables in new numbered files.

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- One row per zip, identified by a digest of its central directory, so a moved or
-- renamed zip is recognized and a zip that was fully ingested is never read again.
CREATE TABLE sources (
  id           INTEGER PRIMARY KEY,
  fingerprint  TEXT NOT NULL UNIQUE,
  zip_path     TEXT NOT NULL,        -- absolute path where the zip was last seen
  zip_name     TEXT NOT NULL,        -- file name only; the manifest shows this, never the path
  label        TEXT NOT NULL UNIQUE, -- e.g. 'sam'; defaults to the zip file stem
  bytes        INTEGER NOT NULL,
  members      INTEGER NOT NULL,     -- file entries, directories excluded
  status       TEXT NOT NULL DEFAULT 'partial' CHECK (status IN ('partial', 'complete')),
  added_at     TEXT NOT NULL,
  completed_at TEXT
);

-- One row per distinct file content (SHA-256). Exact duplicates share a row.
CREATE TABLE media (
  id              INTEGER PRIMARY KEY,
  sha256          TEXT NOT NULL UNIQUE,
  kind            TEXT NOT NULL CHECK (kind IN ('image', 'video')),
  format          TEXT,              -- detected from the content: JPEG, PNG, WEBP, HEIF, MP4, MOV
  bytes           INTEGER NOT NULL,
  width           INTEGER,           -- full size, after EXIF orientation
  height          INTEGER,
  duration_s      REAL,              -- videos, once video analysis exists
  exif_datetime   TEXT,              -- EXIF capture time, ISO 8601, sub-seconds kept, no zone
  exif_offset     TEXT,              -- OffsetTimeOriginal, e.g. +02:00
  has_exif        INTEGER NOT NULL DEFAULT 0,  -- capture time or camera make/model present
  camera_make     TEXT,
  camera_model    TEXT,
  lens_model      TEXT,
  exposure_time   REAL,              -- seconds; compared to tell burst shots from copies
  iso             INTEGER,
  gps_lat         REAL,
  gps_lon         REAL,
  phash           TEXT,              -- 64-bit pHash as 16 hex digits; NULL when not computed
  taken_at        TEXT,              -- resolved by analyze from EXIF, file names, zip times
  taken_at_source TEXT CHECK (taken_at_source IN ('exif', 'filename', 'takeout_json', 'zip_mtime')),
  tz_offset       TEXT,
  status          TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'unreadable', 'skipped')),
  note            TEXT               -- why the file is unreadable or was not fully analyzed
);

-- Where each content was found. A zip may hold the same name twice, so entries are keyed
-- by their position in the zip's central directory.
CREATE TABLE occurrences (
  source_id    INTEGER NOT NULL REFERENCES sources(id),
  member_index INTEGER NOT NULL,
  media_id     INTEGER NOT NULL REFERENCES media(id),
  member_path  TEXT NOT NULL,        -- decoded name with '/' separators
  member_raw   BLOB,                 -- raw name bytes when the zip did not mark them UTF-8
  zip_mtime    TEXT,                 -- the entry's timestamp, ISO 8601, no zone
  PRIMARY KEY (source_id, member_index)
);

-- Zip entries that were not ingested, and why.
CREATE TABLE skipped_members (
  source_id    INTEGER NOT NULL REFERENCES sources(id),
  member_index INTEGER NOT NULL,
  member_path  TEXT NOT NULL,
  member_raw   BLOB,
  bytes        INTEGER NOT NULL,
  reason       TEXT NOT NULL,
  detail       TEXT,
  PRIMARY KEY (source_id, member_index)
);

CREATE TABLE media_groups (          -- avoids GROUPS, a window-function keyword
  id   INTEGER PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('copy', 'burst'))
);

CREATE TABLE group_members (
  group_id  INTEGER NOT NULL REFERENCES media_groups(id) ON DELETE CASCADE,
  media_id  INTEGER NOT NULL REFERENCES media(id),
  is_keeper INTEGER NOT NULL DEFAULT 0,
  distance  INTEGER,                 -- pHash distance to the keeper, for copies
  PRIMARY KEY (group_id, media_id)
);

-- What the last export wrote, so the next export only writes what changed and never
-- touches files that tripsort did not write.
CREATE TABLE exported_files (
  relpath  TEXT PRIMARY KEY,         -- '/'-separated, relative to the project folder
  fold_key TEXT NOT NULL UNIQUE,     -- NFC + casefold of relpath, for case-insensitive drives
  media_id INTEGER NOT NULL REFERENCES media(id),
  role     TEXT NOT NULL CHECK (role IN ('all', 'unreadable', 'dup_keeper', 'dup_copy')),
  method   TEXT NOT NULL CHECK (method IN ('extract', 'hardlink', 'copy')),
  bytes    INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL
);

CREATE TABLE runs (
  id             INTEGER PRIMARY KEY,
  started_at     TEXT NOT NULL,
  finished_at    TEXT,
  tool_version   TEXT,
  config_hash    TEXT,
  stage          TEXT,
  items          INTEGER,
  bytes          INTEGER,
  seconds        REAL,
  peak_rss_bytes INTEGER
);

CREATE INDEX occ_media           ON occurrences(media_id);
CREATE INDEX media_taken         ON media(taken_at);
CREATE INDEX group_members_media ON group_members(media_id);
CREATE INDEX exported_media      ON exported_files(media_id);
