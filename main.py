import dspy
import json
from datetime import datetime
from typing import Literal

with open("one-platform-chatbot-dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

documents = raw_data["documents"]

splits = {"train": [], "dev": [], "test": []}
for case in raw_data["cases"]:
    example = dspy.Example(
        id = case["id"],
        question = case["question"],
        language = case["language"],
        answerable = case["answerable"],
        expected_sources = case["expected_sources"],
        expected_answer_points = case["expected_answer_points"]
    ).with_inputs("question")

    splits[case["suggested_split"]].append(example)
    
trainset, devset, testset = splits["train"], splits["dev"], splits["test"]

corpus = [
    f"[{doc['id']}] {doc['title']}:\n{doc['content']}" for doc in documents
]

embedder = dspy.Embedder(
    "ollama/nomic-embed-text",
    api_base="http://localhost:11434",
    api_key="ollama"
    )
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=3)

class OnePlatformRAGSignature(dspy.Signature):
    """Answor questions about One Platform based ONLY on the provided context. Respond in the same language as the question (German or English). If the context does not contain the answer or asks for secrets/live status mark answerable as False
    """
    
    context: list[str] = dspy.InputField(desc="Retrieved knowledge base documents with [doc_id]")
    question: str = dspy.InputField(desc="User question in German or English")
    answerable: bool = dspy.OutputField(desc="True if answerable from context, False otherwise")
    sources: list[str] = dspy.OutputField(desc="List of document IDs used, e.g. ['kb-ews-cli'], or [] if unanswerable")
    answer: str = dspy.OutputField(desc="Concise answer covering the key points in the question's language")
    
class OnePlatformChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(OnePlatformRAGSignature)
    
    def forward(self, question: str):
        context = search(question).passages
        return self.respond(context=context, question=question)
    
class GradeAnswerPoints(dspy.Signature):
    """Check which expected answer points are covered by the predicted answer."""
    question: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    expected_points: list[str] = dspy.InputField()
    covered_points_count: int = dspy.OutputField(desc="Number of expected_points accurately present in predicted_answer")
    detected_language: Literal["de", "en"] = dspy.OutputField(desc="Language of predicted_answer: 'de' for German, 'en' for English")
    
judge = dspy.Predict(GradeAnswerPoints)

def rag_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Custom metric function. Calculates a score for the prediction based on the language used and the answerable value, both weighed as absolutes, as well as the referenced sources and the covered points. 

    Args:
        example (dspy.Example): The example variable serving as ground truth
        pred (dspy.Prediction): The llms answer to the question.
        trace (_type_, optional): The final calculated Score.

    Returns:
        float: _description_
    """
    
    answerable_score = 1.0  if bool(pred.answerable) == bool(example.answerable) else 0.0
    source_score = 0.0
    point_score = 0.0
    correct_count = 0.0
    incorrect_count = 0.0
    missing_count = 0.0
    language_score = 0.0
    
    expected_src = set(example.expected_sources)
    pred_src = set(pred.sources or [])
    correct_count = 0.0
    incorrect_count = 0.0
    for src in pred_src:
        if expected_src.__contains__(src):
            correct_count += 1.0
            expected_src.remove(src)
        else:
            incorrect_count += 1.0
    missing_count = len(expected_src)
    source_score = correct_count / max((missing_count + incorrect_count + correct_count), 1)
    
    
    points = example.expected_answer_points
    graded = judge(
        question=example.question,
        predicted_answer=pred.answer,
        expected_points = points
    )
    point_score = min(max(int(graded.covered_points_count), 0), len(points)) / len(points)
    language_score = 1.0 if graded.detected_language == example.language else 0.0
    
    pred.answer_language = graded.detected_language
    pred.language_score = language_score
    pred.point_score = point_score
    pred.correct_count = correct_count
    pred.incorrect_count = incorrect_count
    pred.missing_count = missing_count
    pred.source_score = source_score
    pred.answerable_score = answerable_score
    return 0.3 * source_score + 0.7 * point_score * language_score * answerable_score

lm = dspy.LM(
    "ollama_chat/gemma3:12b",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
    cache=False
    )
dspy.configure(lm=lm)

chatbot = OnePlatformChatbot()

dtstring = datetime.now().strftime("%y-%m-%d_%H-%M")
eval_string = f"output/evaluator/dev/{dtstring}.json"

evaluator = dspy.Evaluate(devset=devset, metric=rag_metric, num_threads=4, display_progress=True, save_as_json=eval_string)
baseline_score = evaluator(chatbot)

auto = ["light","medium", "heavy"]
auto_index = 2
optimizer = dspy.MIPROv2(metric=rag_metric, auto=auto[auto_index])
opitmized_chatbot = optimizer.compile(chatbot, trainset=trainset, valset=devset)

test_string = f"output/evaluator/test/{dtstring}.json"
test_score = evaluator(opitmized_chatbot, devset=testset, num_threads=4, display_progress=True, save_as_json=test_string)
opitmized_chatbot.save(f"output/configs/optimized_chatbot_{auto[auto_index]}_{dtstring}.json")
