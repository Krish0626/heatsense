"""GET /api/v1/reports/{id}.csv -- zone/ward summary CSV export (FR-08).

report_id maps to a planning-area summary for the given season. The response
is a streaming CSV so large grids (~45k cells, ~1000 planning areas) don't
block the server.
"""
import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.v1.deps import get_db
from app.schemas.common import Season
from app.services import report_service

router = APIRouter(prefix="/reports", tags=["reports"])

# Column order for the exported CSV.
CSV_COLUMNS = [
    "area_id",
    "area_name",
    "zone_name",
    "cell_count",
    "mean_predicted_lst_c",
    "hotspot_count",
    "top_intervention",
]


@router.get("/{report_id}.csv")
def export_report_csv(
    report_id: int,
    season: Season = Query(default=Season.pre_monsoon, description="Season to summarise"),
    db: Session = Depends(get_db),
):
    """Stream a planning-area summary as CSV.

    report_id is currently used only to validate the request and name the
    file; the actual filter is by `season`. A future version could persist
    named report snapshots and serve them by ID.
    """
    if report_id <= 0:
        raise HTTPException(404, "report_id must be a positive integer")

    rows = report_service.zone_summary_for_season(db, season)
    # If no predictions exist yet, return an empty CSV rather than an error.

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)

    filename = f"heatsense_report_{report_id}_{season.value}.csv"
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
