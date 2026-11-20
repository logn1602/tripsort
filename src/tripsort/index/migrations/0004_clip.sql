-- tripsort index, schema version 4: CLIP embeddings, categories, and bursts.

-- Which photos were embedded with which CLIP model (also when that failed, or the photo had
-- no pixels), so a photo is never embedded twice, while a new CLIP model id embeds every
-- photo again.
CREATE TABLE clip_scans (
  media_id INTEGER NOT NULL REFERENCES media(id),
  model_id TEXT NOT NULL,
  status   TEXT NOT NULL CHECK (status IN ('ok', 'not_decoded', 'failed')),
  detail   TEXT,
  PRIMARY KEY (media_id, model_id)
);

CREATE TABLE clip_embeddings (
  media_id  INTEGER NOT NULL REFERENCES media(id),
  frame_ts  REAL NOT NULL DEFAULT -1,  -- -1 for photos; seconds into a video
  model_id  TEXT NOT NULL,
  embedding BLOB NOT NULL,            -- 512 x float32, L2-normalized
  PRIMARY KEY (media_id, frame_ts, model_id)
);

-- Text embeddings of prompts, per text model. The default prompts' embeddings ship with
-- tripsort; others are embedded once and kept here.
CREATE TABLE prompt_cache (
  model_id  TEXT NOT NULL,
  prompt    TEXT NOT NULL,
  embedding BLOB NOT NULL,            -- 512 x float32, L2-normalized
  PRIMARY KEY (model_id, prompt)
);

-- Every category's score (and other's) for each kept photo, from the last analysis, so a
-- new threshold needs no CLIP at all. is_primary marks the category the photo is sorted by.
CREATE TABLE categories (
  media_id   INTEGER NOT NULL REFERENCES media(id),
  label      TEXT NOT NULL,
  score      REAL NOT NULL,
  is_primary INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (media_id, label)
);

-- How much CLIP takes a photo for a selfie (0 to 1), used when the lens model does not say.
ALTER TABLE media ADD COLUMN clip_selfie REAL;

-- exported_files gains the categories/ role. SQLite cannot change a CHECK constraint in
-- place, so the table is rebuilt; no other table references it.
CREATE TABLE exported_files_v4 (
  relpath  TEXT PRIMARY KEY,
  fold_key TEXT NOT NULL UNIQUE,
  media_id INTEGER NOT NULL REFERENCES media(id),
  role     TEXT NOT NULL CHECK (
             role IN ('all', 'unreadable', 'dup_keeper', 'dup_copy', 'person', 'shot',
                      'category')),
  method   TEXT NOT NULL CHECK (method IN ('extract', 'hardlink', 'copy')),
  bytes    INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL
);
INSERT INTO exported_files_v4 (relpath, fold_key, media_id, role, method, bytes, mtime_ns)
  SELECT relpath, fold_key, media_id, role, method, bytes, mtime_ns FROM exported_files;
DROP TABLE exported_files;
ALTER TABLE exported_files_v4 RENAME TO exported_files;

CREATE INDEX exported_media    ON exported_files(media_id);
CREATE INDEX clip_scans_model  ON clip_scans(model_id);
CREATE INDEX categories_primary ON categories(is_primary, label);
