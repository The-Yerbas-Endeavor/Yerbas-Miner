#!/usr/bin/env python3
import argparse
import csv
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

TARGET_FACTOR = 65536.0


def parse_iso(value):
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "accepted"}


def parse_miner_log(path):
    text = Path(path).read_text(errors="replace")
    start_match = re.search(
        r"Script started on (\d{4}-\d{2}-\d{2}) "
        r"(\d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})",
        text,
    )
    if not start_match:
        raise RuntimeError("Could not find script start timestamp in miner log")

    date_text, time_text, offset_text = start_match.groups()
    base = datetime.fromisoformat(f"{date_text}T{time_text}{offset_text}")

    lines = text.splitlines()
    mining_marker = None
    for i, line in enumerate(lines):
        if "Starting Stratum miner." in line:
            mining_marker = i
            break
    if mining_marker is None:
        raise RuntimeError("Could not find 'Starting Stratum miner.' in miner log")

    time_re = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")
    start_hms = None
    for line in lines[mining_marker:mining_marker + 40]:
        match = time_re.search(line)
        if match:
            start_hms = match.group(1)
            break
    if not start_hms:
        raise RuntimeError("Could not find first mining timestamp after Stratum start")

    def local_datetime_for(hms, reference):
        hour, minute, second = map(int, hms.split(":"))
        candidate = reference.replace(
            hour=hour, minute=minute, second=second, microsecond=0
        )
        if candidate < reference - timedelta(hours=12):
            candidate += timedelta(days=1)
        return candidate

    mining_start = local_datetime_for(start_hms, base)

    last_local = mining_start
    seen_after_start = False
    for line in lines[mining_marker:]:
        match = time_re.search(line)
        if not match:
            continue
        candidate = local_datetime_for(match.group(1), last_local)
        if candidate < last_local:
            candidate += timedelta(days=1)
        last_local = candidate
        seen_after_start = True

    done_match = re.search(
        r"Script done on (\d{4}-\d{2}-\d{2}) "
        r"(\d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})",
        text,
    )
    if done_match:
        date_text, time_text, offset_text = done_match.groups()
        mining_end = datetime.fromisoformat(
            f"{date_text}T{time_text}{offset_text}"
        )
    elif seen_after_start:
        mining_end = last_local
    else:
        mining_end = mining_start

    avg_values = [
        float(value) * (1000.0 if unit.lower().startswith("k") else 1.0)
        for value, unit in re.findall(r"\bAVG\s+([0-9.]+)\s+([kM]?H/s)", text)
    ]
    final_avg = avg_values[-1] if avg_values else None

    return (
        mining_start.astimezone(timezone.utc),
        mining_end.astimezone(timezone.utc),
        final_avg,
    )


def effective_hashrate(rows, start, end):
    accepted = 0
    rejected = 0
    blocks = 0
    work = 0.0

    for row in rows:
        submitted = parse_iso(row.get("share_submitted_utc", ""))
        if submitted is None or submitted < start or submitted > end:
            continue

        is_accepted = truthy(row.get("share_accepted", ""))
        result = str(row.get("share_result", "")).strip().lower()
        if is_accepted:
            accepted += 1
            try:
                difficulty = float(row.get("share_difficulty", "") or 0)
            except ValueError:
                difficulty = 0.0
            work += difficulty * TARGET_FACTOR
            try:
                blocks += int(float(row.get("share_block_candidate", "") or 0))
            except ValueError:
                pass
        elif result or row.get("share_id"):
            rejected += 1

    seconds = max((end - start).total_seconds(), 1.0)
    return accepted, rejected, blocks, work / seconds


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare YERB-Pool share-derived effective hash with "
            "Yerbas Miner telemetry."
        )
    )
    parser.add_argument("csv", help="YERB-Pool worker CSV export")
    parser.add_argument("--miner-log", help="Foreground Yerbas Miner script log")
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("CSV contains no rows")

    meta = rows[0]
    exported = parse_iso(meta.get("exported_utc", ""))

    print("YERB-Pool report")
    for label, key in (
        ("Worker", "worker"),
        ("Pool current", "current_hashrate_hs"),
        ("Pool 24h average", "average_hashrate_24h_hs"),
        ("Pool accepted", "accepted"),
        ("Pool rejected", "rejected"),
        ("Pool efficiency", "efficiency_percent"),
        ("Pool blocks", "blocks_found"),
        ("Exported UTC", "exported_utc"),
    ):
        value = meta.get(key, "")
        if "hashrate" in key and value:
            try:
                value = f"{float(value):,.2f} H/s"
            except ValueError:
                pass
        elif key == "efficiency_percent" and value:
            try:
                value = f"{float(value):.4f}%"
            except ValueError:
                pass
        print(f"  {label:16}: {value}")

    if exported:
        print("\nTrailing share-derived effective hash")
        for hours in (1, 3, 6, 12, 24):
            start = exported - timedelta(hours=hours)
            accepted, rejected, blocks, hashrate = effective_hashrate(
                rows, start, exported
            )
            print(
                f"  {hours:>2}h: {hashrate:9.2f} H/s | "
                f"accepted={accepted} rejected={rejected} blocks={blocks}"
            )

    if args.miner_log:
        start, end, miner_avg = parse_miner_log(args.miner_log)
        accepted, rejected, blocks, hashrate = effective_hashrate(rows, start, end)
        duration = end - start

        print("\nExact miner/pool overlap")
        print(f"  Mining start UTC : {start.isoformat()}")
        print(f"  Mining end UTC   : {end.isoformat()}")
        print(f"  Duration         : {duration}")
        print(f"  Pool effective   : {hashrate:,.2f} H/s")
        print(f"  Accepted/rejected: {accepted}/{rejected}")
        print(f"  Blocks           : {blocks}")
        if miner_avg is not None:
            delta = miner_avg - hashrate
            percent = (miner_avg / hashrate - 1.0) * 100.0 if hashrate > 0 else 0.0
            print(f"  Miner final AVG  : {miner_avg:,.2f} H/s")
            print(f"  Miner-pool delta : {delta:+,.2f} H/s ({percent:+.2f}%)")


if __name__ == "__main__":
    main()
