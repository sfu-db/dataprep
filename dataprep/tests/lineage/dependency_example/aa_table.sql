CREATE VIEW aa_table AS
SELECT a.subject_id, b.gender
FROM `physionet-data.mimiciii_derived.a_table` a, `physionet-data.mimiciii_derived.no_dob` b;
CREATE TABLE a_table AS
SELECT subject_id, gender
FROM `physionet-data.mimiciii_derived.no_dob`;
