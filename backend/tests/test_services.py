from backend import db_models
from backend.services import get_speaker_trends


def test_get_speaker_trends_daily(temp_db_session):
    session = temp_db_session()
    try:
        a = db_models.Analysis(source_filename="file.txt")
        session.add(a)
        session.flush()
        session.add_all([
            db_models.Utterance(
                analysis_id=a.id,
                speaker="Alice",
                date="2024-09-01",
                predictions={"comm_Pausing": 3},
                aggregated_scores={}
            ),
            db_models.Utterance(
                analysis_id=a.id,
                speaker="Alice",
                date="2024-09-01",
                predictions={"comm_Pausing": 5},
                aggregated_scores={}
            ),
            db_models.Utterance(
                analysis_id=a.id,
                speaker="Bob",
                date="2024-09-02",
                predictions={"comm_Pausing": 4},
                aggregated_scores={}
            ),
        ])
        session.commit()

        result = get_speaker_trends(db=session, metric="comm_Pausing", period="daily")
        assert "labels" in result and "datasets" in result
        assert len(result["labels"]) >= 1
        # One dataset per speaker
        labels = result["labels"]
        speakers = [ds["label"] for ds in result["datasets"]]
        assert set(speakers) == {"Alice", "Bob"}
    finally:
        session.close()

