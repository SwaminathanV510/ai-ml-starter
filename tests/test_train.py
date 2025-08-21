from src.train import main


def test_train_runs(monkeypatch, tmp_path, capsys):
    # Run training in a tmp dir to avoid polluting workspace
    monkeypatch.chdir(tmp_path)
    main()
    # Should have produced a saved model and printed accuracy
    out = capsys.readouterr().out
    assert "Accuracy:" in out
