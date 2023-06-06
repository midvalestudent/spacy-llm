# Using GPT models from Azure OpenAI

This example shows how you can use a model from Azure-hosted OpenAI for
categorizing texts in zero- or few-shot settings. Here, we perform binary text
classification to determine if a given text is an `INSULT` or a `COMPLIMENT`.

First, create an OpenAI resource, a deployment of GPT-3.5-turbo, and an API key
from your Azure portal or fetch an existing one. Make the deployment particulars
available as environment variables:

```sh
export AZURE_OPENAI_RESOURCE="resource-name"
export AZURE_OPENAI_DEPLOYMENT="deployment-name"
export AZURE_OPENAI_API_VERSION="api-version"
export AZURE_OPENAI_API_KEY="api-key"
```
(The parameters specifying your deployment can also be set in your config file:
 see the commented lines within this example's configs.)

Then, you can run the pipeline on a sample text via:

```sh
python run_pipeline.py [SAMPLE TEXT] [PATH TO CONFIG] [PATH TO FILE WITH EXAMPLES]
```

For example:

```sh
python run_pipeline.py "You look great today! Nice shirt!" ./zeroshot.cfg
```
or, for few-shot:
```sh
python run_pipeline.py "You look great today! Nice shirt!" ./fewshot.cfg ./examples.jsonl
```

You can also include examples to perform few-shot annotation. To do so, use the 
`fewshot.cfg` file instead. You can find the few-shot examples in
the `examples.jsonl` file. Feel free to change and update it to your liking.
We also support other file formats, including `.yml`, `.yaml` and `.json`.
