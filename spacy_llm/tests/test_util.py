import os

import pytest

from ..util import consume_parameter


class TestConsumeParameter:
    def test_read_env(self):
        config = {"bar": 42}
        os.environ["FOO_BAR"] = "12"
        val = consume_parameter(config, "bar", prefix="FOO_", type_=int)
        assert val == 12

        config = {"baz": 42}
        val = consume_parameter(config, "bar", prefix="FOO_", type_=int)
        assert val == 12

    def test_read_config(self):
        config = {"baz": 42}
        os.environ["FOO_BAR"] = "12"
        val = consume_parameter(config, "baz", prefix="FOO_", type_=int)
        assert val == 42

    def test_read_default(self):
        config = {"bar": 42}
        os.environ["FOO_BAR"] = "12"
        val = consume_parameter(config, "baz", prefix="FOO_", default=1, type_=int)
        assert val == 1

    def test_missing_fails(self):
        config = {"baz": 42}
        os.environ.pop("FOO_BAR", None)

        with pytest.raises(ValueError):
            consume_parameter(config, "bar", prefix="FOO_")
