from source.main import load_data

def test_data_loading():
    data = load_data()
    assert not data.empty
    assert "target" in data.columns
