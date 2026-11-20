-- tripsort index, schema version 3: merge suggestions the user dismissed.

-- Two people the user said are not one person (`tripsort people dismiss`), so tripsort
-- never suggests merging them again. The lower id comes first. Dismissing also pins both
-- people, as a merge pins the person merged into, so later clusterings keep them apart.
-- A pair goes away with either person, for example when the user merges them after all.
CREATE TABLE dismissed_suggestions (
  person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
  other_id  INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
  PRIMARY KEY (person_id, other_id),
  CHECK (person_id < other_id)
);
