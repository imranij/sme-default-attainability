# Changes from the pipeline that produced the results

The scripts in `code/` and `lib/` are the ones that produced the files in `results/`, with two
kinds of change made for the public copy, neither of which alters a computation.

## Path handling

In the working copy, `code/common.py` found the library through a folder two levels above the
script, and eleven scripts built the path to `data/processed/` for themselves. The public copy
finds the repository root once, in `common.py`, from the environment variable `SME_ATTAIN_ROOT`
when it is set and from the location of `common.py` otherwise. Every path is then built from
that root, and the eleven scripts import `PROC` from `common.py`.
`0001-repo-relative-root.patch` shows these edits line by line, and applying it in reverse
gives back the original path code.

## Comments, docstrings and text fields

Comments and docstrings were rewritten where they pointed to internal notes or working files
that are not part of the release. Section numbers were updated to match the paper. A few
descriptive text fields that the scripts write into the JSON files (for example the `reason`
note in `gate_status_sba_arm.json`) were reworded in the script and in the committed file
together, so a re-run still writes the same text. No number in any results file was edited.

## Which simulator run is shipped

`code/02_selection_mechanisms.py` is the configuration that produced
`results/Q2_sim_mechanisms.json` and `results/Q2_sim_summary.json`. It uses 12,000 simulated
firms, one seed and 80 boosting rounds. A larger configuration with 40,000 firms and three seeds
existed in the working copy, but it did not produce any reported number, so it is not included.
