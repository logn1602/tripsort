-- tripsort index, schema version 2: faces, people, and shot types.

-- Which photos were scanned for faces, and with which models, so a photo without faces is
-- never scanned again, while a new face model id rescans every photo.
CREATE TABLE face_scans (
  media_id INTEGER NOT NULL REFERENCES media(id),
  model_id TEXT NOT NULL,
  status   TEXT NOT NULL CHECK (status IN ('ok', 'not_decoded', 'failed')),
  faces    INTEGER NOT NULL DEFAULT 0,
  detail   TEXT,                        -- why the scan did not run or failed
  PRIMARY KEY (media_id, model_id)
);

-- One row per person. person_07 keeps its number for good, and numbers are never reused.
CREATE TABLE people (
  id       INTEGER PRIMARY KEY,
  number   INTEGER NOT NULL UNIQUE,
  name     TEXT,                        -- NULL until named from refs/ or by `people rename`
  name_key TEXT UNIQUE,                 -- folder-safe, casefolded name: 'Sam' and 'sam' are one
  named_by TEXT CHECK (named_by IN ('refs', 'user')),
  pinned   INTEGER NOT NULL DEFAULT 0,  -- renamed or merged into by the user: kept together
  centroid BLOB                         -- mean embedding of its clustered faces, L2-normalized
);

-- Every face detected above a low score floor. Boxes and landmarks are fractions of the
-- EXIF-oriented image, so they do not depend on the size the image was decoded at. The
-- real score and foreground thresholds apply in analyze, so tuning them reads no pixels.
CREATE TABLE faces (
  id            INTEGER PRIMARY KEY,
  media_id      INTEGER NOT NULL REFERENCES media(id),
  model_id      TEXT NOT NULL,
  frame_ts      REAL NOT NULL DEFAULT -1, -- -1 for photos; seconds into a video
  face_index    INTEGER NOT NULL,         -- 0 is the largest face in the photo
  x             REAL NOT NULL,
  y             REAL NOT NULL,
  w             REAL NOT NULL,
  h             REAL NOT NULL,
  score         REAL NOT NULL,
  landmarks     TEXT NOT NULL,            -- JSON: five [x, y] points (eyes, nose tip, mouth corners)
  embedding     BLOB NOT NULL,            -- 128 x float32, L2-normalized
  -- Set by analyze. person_id is the last person the face belonged to and survives a face
  -- dropping out of the clustering for a while; match says whether it belongs to them now.
  is_foreground INTEGER NOT NULL DEFAULT 0,
  cluster       INTEGER,                  -- label from the last clustering, -1 for noise
  person_id     INTEGER REFERENCES people(id) ON DELETE SET NULL,
  match         TEXT CHECK (match IN ('cluster', 'attached', 'background')),
  UNIQUE (media_id, model_id, frame_ts, face_index)
);

-- The photos in refs/ (one clear photo per person), embedded once per model.
CREATE TABLE ref_images (
  relpath   TEXT NOT NULL,              -- '/'-separated, relative to the refs folder
  model_id  TEXT NOT NULL,
  name      TEXT NOT NULL,              -- from refs/<name>.jpg or refs/<name>/<any>.jpg
  sha256    TEXT NOT NULL,
  status    TEXT NOT NULL CHECK (status IN ('ok', 'no_face', 'unreadable')),
  faces     INTEGER NOT NULL DEFAULT 0,
  embedding BLOB,                       -- the largest face; NULL unless status is 'ok'
  PRIMARY KEY (relpath, model_id)
);

-- Shot types, set by analyze from the foreground faces and the lens model.
ALTER TABLE media ADD COLUMN is_front_camera INTEGER;
ALTER TABLE media ADD COLUMN shot TEXT CHECK (shot IN ('solo', 'group', 'none'));
ALTER TABLE media ADD COLUMN is_selfie INTEGER;

-- exported_files gains the people/ and shots/ roles. SQLite cannot change a CHECK
-- constraint in place, so the table is rebuilt; no other table references it.
CREATE TABLE exported_files_v2 (
  relpath  TEXT PRIMARY KEY,
  fold_key TEXT NOT NULL UNIQUE,
  media_id INTEGER NOT NULL REFERENCES media(id),
  role     TEXT NOT NULL CHECK (
             role IN ('all', 'unreadable', 'dup_keeper', 'dup_copy', 'person', 'shot')),
  method   TEXT NOT NULL CHECK (method IN ('extract', 'hardlink', 'copy')),
  bytes    INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL
);
INSERT INTO exported_files_v2 (relpath, fold_key, media_id, role, method, bytes, mtime_ns)
  SELECT relpath, fold_key, media_id, role, method, bytes, mtime_ns FROM exported_files;
DROP TABLE exported_files;
ALTER TABLE exported_files_v2 RENAME TO exported_files;

CREATE INDEX exported_media   ON exported_files(media_id);
CREATE INDEX faces_media      ON faces(media_id);
CREATE INDEX faces_person     ON faces(person_id);
CREATE INDEX face_scans_model ON face_scans(model_id);
