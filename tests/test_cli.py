import json

from rag_assistant.cli import main


def test_cli_demo_question_evaluation_and_bad_input(tmp_path, capsys):
    prefix = ["--data-dir", str(tmp_path)]
    assert main([*prefix, "demo"]) == 0
    capsys.readouterr()
    assert main([*prefix, "ask", "How does Atlas handle Redis?"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["status"] == "supported" and answer["claims"]
    assert not any(claim["text"].startswith("#") for claim in answer["claims"])
    assert main(["evaluate"]) == 0
    evaluation = json.loads(capsys.readouterr().out)
    assert evaluation["recall_at_5"] == 1 and evaluation["unknown_question_abstained"]
    assert main([*prefix, "ingest", str(tmp_path / "missing.pdf")]) == 1


def test_cli_import_real_file(tmp_path, capsys):
    path = tmp_path / "notes.md"
    path.write_text("Atlas uses Postgres to persist orders.")
    assert main(["--data-dir", str(tmp_path / "index"), "ingest", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["filename"] == "notes.md"
