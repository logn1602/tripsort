-- tripsort index, schema version 6: each burst shot's place in its burst's ranking, so
-- export can give a person the best shot of a burst that shows them.

-- bursts: 1 is the best shot; NULL for copy groups, and until analyze ranks the burst
ALTER TABLE group_members ADD COLUMN shot_rank INTEGER;
