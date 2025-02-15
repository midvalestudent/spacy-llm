import json
from pathlib import Path

import pytest
from confection import Config
from pytest import FixtureRequest
from spacy.tokens import Span
from spacy.training import Example

from spacy_llm.pipeline import LLMWrapper
from spacy_llm.tasks.rel import RelationItem, RELTask
from spacy_llm.ty import Labeled, LLMTask
from spacy_llm.util import assemble_from_config, split_labels

from ..compat import has_openai_key

EXAMPLES_DIR = Path(__file__).parent / "examples"


@pytest.fixture
def zeroshot_cfg_string():
    return """
    [nlp]
    lang = "en"
    pipeline = ["ner", "llm"]
    batch_size = 128

    [components]

    [components.ner]
    source = "en_core_web_md"

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.REL.v1"
    labels = "LivesIn,Visits"

    [components.llm.backend]
    @llm_backends = "spacy.REST.v1"
    api = "OpenAI"

    [initialize]
    vectors = "en_core_web_md"
    """


@pytest.fixture
def fewshot_cfg_string():
    return f"""
    [nlp]
    lang = "en"
    pipeline = ["ner", "llm"]
    batch_size = 128

    [components]

    [components.ner]
    source = "en_core_web_md"

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.REL.v1"
    labels = ["LivesIn", "Visits"]

    [components.llm.task.examples]
    @misc = "spacy.FewShotReader.v1"
    path = {str(EXAMPLES_DIR / "rel_examples.jsonl")}

    [components.llm.backend]
    @llm_backends = "spacy.REST.v1"
    api = "OpenAI"

    [initialize]
    vectors = "en_core_web_md"
    """


@pytest.fixture
def noop_config():
    return """
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    batch_size = 128

    [components]

    [components.llm]
    factory = "llm"

    [components.llm.task]
    @llm_tasks = "spacy.REL.v1"
    labels = ["LivesIn", "Visits"]

    [components.llm.task.normalizer]
    @misc = "spacy.LowercaseNormalizer.v1"

    [components.llm.backend]
    @llm_backends = "test.NoOpBackend.v1"
    """


@pytest.fixture
def task():
    text = "Joey rents a place in New York City."
    gold_relations = [RelationItem(dep=0, dest=1, relation="LivesIn")]
    return text, gold_relations


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_rel_config(cfg_string, request: FixtureRequest):
    """Simple test to check if the config loads properly given different settings"""
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = assemble_from_config(orig_config)
    assert nlp.pipe_names == ["ner", "llm"]

    pipe = nlp.get_pipe("llm")
    assert isinstance(pipe, LLMWrapper)
    assert isinstance(pipe.task, LLMTask)

    task = pipe.task
    labels = orig_config["components"]["llm"]["task"]["labels"]
    labels = split_labels(labels)
    assert isinstance(task, Labeled)
    assert task.labels == tuple(labels)
    assert pipe.labels == task.labels
    assert nlp.pipe_labels["llm"] == list(task.labels)


@pytest.mark.external
@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.parametrize("cfg_string", ["zeroshot_cfg_string", "fewshot_cfg_string"])
def test_rel_predict(task, cfg_string, request):
    """Use OpenAI to get REL results.
    Note that this test may fail randomly, as the LLM's output is unguaranteed to be consistent/predictable
    """
    cfg_string = request.getfixturevalue(cfg_string)
    orig_config = Config().from_str(cfg_string)
    nlp = assemble_from_config(orig_config)

    text, _ = task
    doc = nlp(text)

    assert doc.ents
    assert doc._.rel


def test_rel_init(noop_config):

    RELTask._check_rel_extention()

    config = Config().from_str(noop_config)
    del config["components"]["llm"]["task"]["labels"]
    nlp = assemble_from_config(config)

    examples = []

    for text, rel in [
        ("Alice travelled to London.", "Visits"),
        ("Bob lives in Manchester.", "LivesIn"),
    ]:
        predicted = nlp.make_doc(text)
        reference = predicted.copy()

        # We might want to set those on the predicted example as well...
        reference.ents = [
            Span(reference, 0, 1, label="PER"),
            Span(reference, 3, 4, label="LOC"),
        ]

        reference._.rel = [RelationItem(dep=0, dest=1, relation=rel)]

        examples.append(Example(predicted, reference))

    _, llm = nlp.pipeline[0]
    task: RELTask = llm._task

    assert set(task._label_dict.values()) == set()
    nlp.initialize(lambda: examples)
    assert set(task._label_dict.values()) == {"LivesIn", "Visits"}


def test_rel_serde(noop_config, tmp_path: Path):

    config = Config().from_str(noop_config)
    del config["components"]["llm"]["task"]["labels"]

    nlp1 = assemble_from_config(config)
    nlp2 = assemble_from_config(config)
    nlp3 = assemble_from_config(config)

    labels = {"livesin": "LivesIn", "visits": "Visits"}

    task1: RELTask = nlp1.get_pipe("llm")._task
    task2: RELTask = nlp2.get_pipe("llm")._task
    task3: RELTask = nlp3.get_pipe("llm")._task

    # Artificially add labels to task1
    task1._label_dict = labels

    assert task1._label_dict == labels
    assert task2._label_dict == dict()
    assert task3._label_dict == dict()

    path = tmp_path / "model"

    nlp1.to_disk(path)

    cfgs = list(path.rglob("cfg"))
    assert len(cfgs) == 1

    cfg = json.loads(cfgs[0].read_text())
    assert cfg["_label_dict"] == labels

    nlp2.from_disk(path)
    nlp3.from_bytes(nlp1.to_bytes())

    assert task1._label_dict == task2._label_dict == task3._label_dict == labels
