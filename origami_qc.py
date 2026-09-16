"""Read-only audits of saved shared-candidate digital classifications."""
from __future__ import annotations

import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd


MEASUREMENTS = ("digital_pixel_probabilities", "digital_group_localization_evidence",
                "digital_group_prominences")


def expected_template_fractions(names):
    """Sample composition: code3 has twice the concentration of other templates."""
    weights = np.array([2.0 if str(name).lower().replace("_", "").replace(" ", "").replace("-", "") == "code3" else 1.0 for name in names])
    return weights / weights.sum() if len(weights) else weights


def expected_group_fractions(results, group_ids):
    """Recompute from loaded patterns so saved equal-mixture references cannot persist."""
    patterns = []
    for payload in results.values():
        model = payload.get("params", {}).get("logical_model", {})
        ids = tuple(map(str, model.get("bit_ids", ())))
        bits = np.asarray(model.get("active_bits", ()), dtype=float)
        if len(set(ids)) != len(ids) or bits.shape != (len(ids),) or set(ids) != set(group_ids):
            return np.empty(0)
        patterns.append(bits[[ids.index(bit) for bit in group_ids]])
    return expected_template_fractions(list(results)) @ np.asarray(patterns) if patterns else np.empty(0)


def build_classification_audits(results):
    """Return CSV-ready tables; nearest patterns never change assignments.

    Candidate IDs are one-based indices in the saved shared candidate arrays.
    Invalid measurements are excluded from pattern distances and denominators.
    """
    if not results:
        raise ValueError("Run Step 4 with digital-group templates before opening the audits.")
    names = list(results)
    first = results[names[0]]
    params = first.get("params", {})
    ids = tuple(map(str, params.get("digital_pixel_ids", ())))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Digital-group IDs are missing or duplicated. Rerun Steps 3 and 4.")
    probability = np.asarray(params.get(MEASUREMENTS[0], ()), dtype=float)
    if probability.ndim != 2 or probability.shape[1] != len(ids):
        raise ValueError("Digital-group measurements are missing. Rerun Steps 3 and 4.")
    n, g = probability.shape
    arrays = [np.asarray(params.get(key, ()), dtype=float) for key in MEASUREMENTS]
    if any(a.shape != (n, g) for a in arrays):
        raise ValueError("Digital-group measurements are incomplete. Rerun Steps 3 and 4.")
    probability, evidence, prominence = arrays
    support_min = float(params.get("min_site_localizations", 0))
    prominence_min = float(params.get("min_site_evidence", 0))
    if not np.isfinite([support_min, prominence_min]).all() or support_min < 0 or not 0 <= prominence_min <= 1:
        raise ValueError("Saved digital-group thresholds are invalid.")
    picks = first.get("picks")
    regions = getattr(picks, "regions", ())
    if len(regions) != n:
        raise ValueError("Saved candidates and measurements differ. Rerun Steps 3 and 4.")
    centers = np.asarray([np.mean(r, axis=0)[:2] if len(r) else [np.nan, np.nan]
                          for r in regions], dtype=float).reshape(n, 2)
    expected, accepted = [], []
    for name in names:
        payload = results[name]
        p = payload.get("params", {})
        other_ids = tuple(map(str, p.get("digital_pixel_ids", ())))
        model = p.get("logical_model", {})
        model_ids = tuple(map(str, model.get("bit_ids", ())))
        bits = np.asarray(model.get("active_bits", ()), dtype=bool)
        if (len(other_ids) != g or set(other_ids) != set(ids)
                or len(model_ids) != g or set(model_ids) != set(ids) or bits.shape != (g,)):
            raise ValueError("Templates must use the same measured digital-group IDs.")
        order = [other_ids.index(bit) for bit in ids]
        for key, reference in zip(MEASUREMENTS, arrays):
            values = np.asarray(p.get(key, ()), dtype=float)
            if values.shape != (n, g) or not np.array_equal(values[:, order], reference, equal_nan=True):
                raise ValueError("Audits require shared candidate measurements. Rerun Steps 3 and 4 with shared alignment.")
        if (float(p.get("min_site_localizations", 0)) != support_min
                or float(p.get("min_site_evidence", 0)) != prominence_min):
            raise ValueError("Saved templates use different digital-group thresholds. Rerun Step 4.")
        other_picks = payload.get("picks")
        other_regions = getattr(other_picks, "regions", ())
        if len(other_regions) != n or any(
            a is not b and not np.array_equal(a, b, equal_nan=True)
            for a, b in zip(regions, other_regions)
        ):
            raise ValueError("Audits require the same candidate ordering for every template.")
        mask = np.asarray(getattr(other_picks, "accepted_mask", ()), dtype=bool)
        if mask.shape != (n,):
            raise ValueError("Saved assignment masks are incomplete. Rerun Step 4.")
        accepted.append(mask)
        expected.append(bits[[model_ids.index(bit) for bit in ids]])
    expected = np.asarray(expected)
    accepted = np.asarray(accepted).T
    if np.any(accepted.sum(axis=1) > 1):
        raise ValueError("A candidate has multiple saved assignments. Rerun Step 4.")
    valid = np.logical_and.reduce([np.isfinite(a).all(axis=1) for a in arrays])
    support_pass = evidence >= support_min
    prominence_pass = prominence >= prominence_min
    probability_pass = probability >= .5
    states = support_pass & prominence_pass & probability_pass
    distances = np.sum(states[:, None, :] != expected[None, :, :], axis=2)
    matches = (distances == 0) & valid[:, None]
    if np.any(accepted & ~matches):
        raise ValueError("Saved assignments disagree with the digital calls. Rerun Step 4.")
    assigned = accepted.any(axis=1)
    statuses = np.where(~valid, "invalid", np.where(assigned, "assigned",
        np.where(matches.sum(axis=1) == 0, "unmatched",
        np.where(matches.sum(axis=1) == 1, "rejected_exact", "ambiguous_exact"))))
    labels = [names[np.argmax(row)] if row.any() else "" for row in accepted]
    patterns = ["".join("1" if b else "0" for b in row) if ok else ""
                for row, ok in zip(states, valid)]
    candidate = pd.DataFrame(dict(candidate_id=np.arange(1, n + 1), status=statuses,
        assigned_template=labels, pattern=patterns, x_nm=centers[:, 0], y_nm=centers[:, 1],
        localization_count=[len(r) for r in regions]))
    for attribute in ("rectangle_confidence", "crop_retained_fractions", "footprint_overlap_fraction"):
        values = np.asarray(getattr(picks, attribute, ()), dtype=float)
        candidate[attribute] = values if values.shape == (n,) else np.nan
    # This is a support-only diagnostic, not a calibrated class probability.
    support_score = np.full((n, g), np.nan)
    if support_min > 0:
        support_score = 1 / (1 + np.exp(-np.clip(np.log(19) * (evidence / support_min - 1), -700, 700)))
    group_rows = []
    distance_rows = []
    for i in range(n):
        for j, bit in enumerate(ids):
            group_rows.append(dict(candidate_id=i + 1, status=statuses[i], assigned_template=labels[i],
                group_id=bit, valid_candidate=bool(valid[i]), evidence=evidence[i, j],
                support_threshold=support_min, support_only_score=support_score[i, j],
                stored_on_score=probability[i, j], probability_threshold=.5,
                relative_prominence=prominence[i, j], prominence_threshold=prominence_min,
                support_pass=bool(support_pass[i, j]), probability_pass=bool(probability_pass[i, j]),
                prominence_pass=bool(prominence_pass[i, j]),
                support_pass_prominence_fail=bool(valid[i] and support_pass[i, j] and not prominence_pass[i, j]),
                final_on=bool(states[i, j]) if valid[i] else None))
        if not valid[i]:
            continue
        for t, name in enumerate(names):
            missing = expected[t] & ~states[i]
            extra = ~expected[t] & states[i]
            distance_rows.append(dict(candidate_id=i + 1, status=statuses[i], template=name,
                distance=int(distances[i, t]), nearest=bool(distances[i, t] == distances[i].min()),
                missing_on_count=int(missing.sum()), extra_on_count=int(extra.sum()),
                missing_groups=";".join(np.asarray(ids)[missing]), extra_groups=";".join(np.asarray(ids)[extra])))
    summary = []
    cohorts = [("All valid", valid), ("Assigned", valid & assigned),
               ("Unmatched", valid & (statuses == "unmatched"))]
    cohorts += [("Assigned: " + name, valid & accepted[:, t]) for t, name in enumerate(names)]
    for cohort, mask in cohorts:
        count = int(mask.sum())
        for j, bit in enumerate(ids):
            summary.append(dict(cohort=cohort, group_id=bit, candidates=count,
                on_count=int(states[mask, j].sum()), on_fraction=float(states[mask, j].mean()) if count else np.nan,
                mixture_expected_on_fraction=float(expected_template_fractions(names) @ expected[:, j]),
                support_fail=int((~support_pass[mask, j]).sum()),
                probability_fail=int((~probability_pass[mask, j]).sum()),
                prominence_fail=int((~prominence_pass[mask, j]).sum()),
                support_pass_prominence_fail=int((support_pass[mask, j] & ~prominence_pass[mask, j]).sum())))
    pattern_rows = []
    for pattern, frame in candidate[candidate.status == "unmatched"].groupby("pattern"):
        i = int(frame.iloc[0].candidate_id) - 1
        nearest = np.flatnonzero(distances[i] == distances[i].min())
        pattern_rows.append(dict(pattern=pattern, count=len(frame), distance=int(distances[i].min()),
            nearest_templates=";".join(names[t] for t in nearest),
            candidate_ids=";".join(map(str, frame.candidate_id))))
    pattern_table = pd.DataFrame(pattern_rows, columns=["pattern", "count", "distance", "nearest_templates", "candidate_ids"])
    pattern_table = pattern_table.sort_values("count", ascending=False, kind="stable")
    dropout = []
    for f in np.flatnonzero(expected.all(axis=1)):
        for t, name in enumerate(names):
            if t == f:
                continue
            missing = expected[f] & ~expected[t]
            dropout.append(dict(full_template=names[f], comparison_template=name,
                absent_groups=";".join(np.asarray(ids)[missing]), absent_group_count=int(missing.sum()),
                observed_exact_pattern=int(matches[:, t].sum()), assigned_count=int(accepted[:, t].sum())))
    return dict(candidates=candidate, candidate_groups=pd.DataFrame(group_rows, columns=[
            "candidate_id", "status", "assigned_template", "group_id", "valid_candidate", "evidence",
            "support_threshold", "support_only_score", "stored_on_score", "probability_threshold",
            "relative_prominence", "prominence_threshold", "support_pass", "probability_pass",
            "prominence_pass", "support_pass_prominence_fail", "final_on"]),
        group_summary=pd.DataFrame(summary),
        candidate_template_distances=pd.DataFrame(distance_rows, columns=["candidate_id", "status", "template", "distance", "nearest", "missing_on_count", "extra_on_count", "missing_groups", "extra_groups"]),
        unmatched_patterns=pattern_table,
        full_dropout_comparisons=pd.DataFrame(dropout, columns=["full_template", "comparison_template", "absent_groups", "absent_group_count", "observed_exact_pattern", "assigned_count"]),
        metadata=dict(group_order=ids, template_names=names, candidate_count=n,
            expected_template_fractions=dict(zip(names, expected_template_fractions(names).tolist())),
            invalid_count=int((~valid).sum()), support_threshold=support_min, prominence_threshold=prominence_min,
            notes="Read-only audit. Nearest patterns and full-dropout comparisons are hypotheses, not truth labels. Mixture reference gives code3 twice the weight of other templates; it is not a classifier prior. Gate failure counts can overlap. Candidate IDs index the saved shared candidates; undetected objects are not included. Support-only scores are not calibrated class probabilities."))


def export_classification_audits(audit, path):
    """Write a single reviewable ZIP containing CSV tables and interpretation notes."""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, table in audit.items():
            if name != "metadata":
                archive.writestr(name + ".csv", table.to_csv(index=False))
        archive.writestr("metadata.json", json.dumps(audit["metadata"], indent=2))


def plot_classification_audit(figure, audit, view):
    """Render audits without depending on a Tk application."""
    figure.clear()
    ids = audit["metadata"]["group_order"]
    if view == "Digital-group threshold audit":
        left, right = figure.subplots(1, 2)
        summary = audit["group_summary"]
        rows = summary[summary.cohort == "All valid"]
        y = np.arange(len(ids))
        left.barh(y - .18, rows.on_fraction, height=.35, label="Observed final ON")
        left.barh(y + .18, rows.mixture_expected_on_fraction, height=.35, label="Mixture reference (code3 ×2)")
        left.set(yticks=y, yticklabels=ids, xlim=(0, 1), xlabel="Fraction of valid candidates", title="Final calls versus mixture reference")
        left.legend(fontsize=8)
        keys = ["support_fail", "probability_fail", "prominence_fail", "support_pass_prominence_fail"]
        values = rows[keys].to_numpy()
        right.imshow(values, aspect="auto", cmap="YlOrRd")
        right.set(yticks=y, yticklabels=ids, xticks=range(4),
            xticklabels=["Support\nfail", "ON score\nfail", "Prominence\nfail", "Support pass\nprominence fail"],
            title="Gate counts (failures may overlap)")
        for j in range(len(ids)):
            for k in range(4):
                right.text(k, j, str(values[j, k]), ha="center", va="center",
                           color="white" if values[j, k] > values.max() * .65 else "black")
        total = int(rows.iloc[0].candidates)
        figure.suptitle(f"Digital-group threshold audit: {total:,} valid candidates; {audit['metadata']['invalid_count']:,} invalid excluded")
    else:
        top, bottom = figure.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
        patterns = audit["unmatched_patterns"].head(20)
        if len(patterns):
            pixels = np.array([[int(bit) for bit in p] for p in patterns.pattern])
            top.imshow(pixels, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            labels = [f"{r.count} candidates | d={r.distance} | {r.nearest_templates}" for r in patterns.itertuples()]
            top.set(xticks=range(len(ids)), xticklabels=ids, yticks=range(len(patterns)), yticklabels=labels,
                    title="Most frequent unmatched patterns: blue = ON; d = number of differing groups")
        else:
            top.text(.5, .5, "No unmatched patterns among valid candidates", ha="center", transform=top.transAxes)
            top.set_axis_off()
        bottom.set_axis_off()
        drop = audit["full_dropout_comparisons"]
        text = "Groups absent relative to the all-ON template (possible dropout, not proven origin):\n"
        text += "\n".join(f"{r.full_template} → {r.comparison_template}: {r.absent_groups or '(none)'}; observed exact pattern: {r.observed_exact_pattern}" for r in drop.itertuples()) if len(drop) else "No all-ON template is present."
        bottom.text(0, 1, text, va="top", transform=bottom.transAxes, fontsize=9)
        figure.suptitle(f"Unmatched-pattern audit: {int(audit['unmatched_patterns']['count'].sum()):,} candidates; nearest ties retained")
    figure.tight_layout(rect=(0, 0, 1, .94))
